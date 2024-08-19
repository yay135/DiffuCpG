import os
import math
import random
import numpy as np
from inspect import isfunction
from functools import partial
from tqdm import tqdm
import pandas as pd

from torch.optim import AdamW
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.nn.parallel import DataParallel

from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

import argparse


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (p l) -> b (c p) l", p=2),
        nn.Conv1d(dim * 2, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1",
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) l -> b h c l", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b (h d) l", l=l)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) l -> b h c l", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c l -> b (h c) l", h=self.heads, l=l)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        # changed to 1 and 0 from 7,3
        self.init_conv = nn.Conv1d(input_channels, init_dim, 1, padding=0)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv1d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out,
                                    time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out,
                                    time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


timesteps = 2000

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


# mse loss ignore -1 locs in y_true
class CustomMSELoss(nn.Module):
    def forward(self, y_true, y_pred, x_start):
        # Mask for tensor locations where true values are not -1
        mask = (x_start != -1)

        # create two mask to seperate loss of TCGA channels and methy channels
        mask_copy_loss0 = torch.full(mask.size(), False, device=mask.device)
        mask_copy_loss1 = torch.full(mask.size(), False, device=mask.device)

        mask_copy_loss0[:, :-1] = mask[:, :-1]
        mask_copy_loss1[:, -1] = mask[:, -1]

        # Apply the mask to true values and predictions
        masked_true0 = y_true[mask_copy_loss0]
        masked_pred0 = y_pred[mask_copy_loss0]

        # Calculate the squared difference
        squared_diff0 = (masked_true0 - masked_pred0) ** 2

        # Calculate the mean of squared differences
        mse0 = torch.mean(squared_diff0)

        # Apply the mask to true values and predictions
        masked_true1 = y_true[mask_copy_loss1]
        masked_pred1 = y_pred[mask_copy_loss1]

        # Calculate the squared difference
        squared_diff1 = (masked_true1 - masked_pred1) ** 2

        # Calculate the mean of squared differences
        mse1 = torch.mean(squared_diff1)
        # different weights for tcga channel and methy channel
        return mse0*0.1 + mse1 * 0.9


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    custom_mse_loss = CustomMSELoss()

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    elif loss_type == 'custom':
        loss = custom_mse_loss(noise, predicted_noise, x_start)
    else:
        raise NotImplementedError()

    return loss


def iter_sample(data):
    def iter():
        for sample in data:
            yield torch.tensor(sample.values, dtype=torch.float)
    return iter


class IterDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self.generator = generator

    def __iter__(self):
        return self.generator()


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# custom inpainting (imputation)


@torch.no_grad()
def inpainting(model, x_p,  imp_mask):
    device = next(model.parameters()).device
    x_p = x_p.clone().to(device)
    imp_mask = imp_mask.clone().to(device)
    assert (x_p.size() == imp_mask.size())

    b = x_p.shape[0]

    t = torch.full((b,), timesteps-1, device=device, dtype=torch.long)
    noise = torch.randn_like(x_p, device=device)
    x_noisy = q_sample(x_start=x_p, t=t, noise=noise)
    img = x_noisy

    for i in tqdm(reversed(range(0, timesteps)), desc='imputing progress', total=timesteps):
        img = p_sample(model, img, torch.full(
            (b,), i, device=device, dtype=torch.long), i)
        # user original values where values are not missing
        img[imp_mask] = x_p[imp_mask]

    return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train a DDPM model')
    parser.add_argument('-t', '--train_folder', type=str, required=True)
    parser.add_argument('-f', '--model_folder', type=str, required=True)
    parser.add_argument('-w', '--win_size', type=int, default=1000)
    parser.add_argument('-c', '--channel', type=int, required=True)
    parser.add_argument('-d', '--cuda_device', type=int, default=0)
    parser.add_argument("-e", "--epoch", type=int, default=2000)
    parser.add_argument("-s", "--earlystop", type=bool, default=False)
    parser.add_argument("-p", "--patience", type=int, default=15)

    args = parser.parse_args()

    if torch.cuda.is_available():
        cuda_device = int(args.cuda_device)
        if 0 <= cuda_device < torch.cuda.device_count():
            device = torch.device(f"cuda:{cuda_device}")
        else:
            device = torch.device("cuda:0")

    else:
        device = torch.device('cpu')

    print(f"using device {device}")
    # load sequence and methylation data
    win_size = int(args.win_size)
    model_folder = args.model_folder
    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    patience = int(args.patience)
    earlystop = bool(args.earlystop)
    if earlystop:
        print(f"using early stop, patience set to {patience}, may lead to insufficient training.")

    epochs = int(args.epoch)
    mid_channels = 32
    channels = int(args.channel)
    batch_size = 64
    learning_rate = 1e-3
    train_data_folder = args.train_folder

    train_samples = os.listdir(train_data_folder)
    train_samples, test_samples = train_test_split(
        train_samples, test_size=0.1)

    train_samples = random.sample(
        train_samples, k=min(2000, len(train_samples)))
    test_samples = random.sample(test_samples, k=min(200, len(test_samples)))

    sample_paths_train = [f"{train_data_folder}/{p}" for p in train_samples]
    sample_paths_test = [f"{train_data_folder}/{p}" for p in test_samples]

    model_save_path = os.path.join(model_folder, 'diffusion_1d')
    train_total = math.ceil(len(train_samples)/batch_size)
    test_total = math.ceil(len(test_samples)/batch_size)

    model = Unet(
        dim=mid_channels,
        channels=channels,
        dim_mults=(1, 2, 4,)
    )

    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    best_loss = float("inf")

    print("loading data ...")

    train_data = list(map(lambda x: pd.read_csv(
        x, header=None), tqdm(sample_paths_train)))
    test_data = list(map(lambda x: pd.read_csv(
        x, header=None), tqdm(sample_paths_test)))

    train_dataloader = torch.utils.data.DataLoader(
        IterDataset(iter_sample(train_data)), batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(
        IterDataset(iter_sample(test_data)), batch_size=batch_size)

    for epoch in range(epochs):
        epoch_losses = []
        pbar = tqdm(enumerate(train_dataloader), total=train_total)
        for step, batch in pbar:
            optimizer.zero_grad()
            assert (win_size == batch.shape[2])
            batch_size = batch.shape[0]
            batch = batch.to(device)
            t = torch.randint(0, timesteps, (batch_size,)).long().to(device)
            loss = p_losses(model, batch, t, loss_type="custom")
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())
            avg_loss = round(sum(epoch_losses)/len(epoch_losses), 4)

            pbar.set_description(
                (f"epoch:{epoch}/{epochs}, "
                 f"step: {step + 1}/{train_total}, "
                 f"train loss:{avg_loss}, "
                 f"best loss: {round(best_loss,4)}, ")
            )

        # test steps
        with torch.no_grad():
            pbar = tqdm(test_dataloader, total=test_total)
            test_losses = []
            for batch in pbar:
                batch_size = batch.shape[0]
                batch = batch.to(device)
                t = torch.randint(0, timesteps, (batch_size,)).long().to(device)
                loss = p_losses(model, batch, t, loss_type="custom")
                test_losses.append(loss.item())
                test_loss = round(sum(test_losses)/len(test_losses), 4)
                pbar.set_description(
                    f"test loss: {test_loss}")

        test_loss_avg = sum(test_losses)/len(test_losses)
        if test_loss_avg >= best_loss:
            if earlystop:
                patience -= 1
                if patience <= 0:
                    print(f"did not improve for {int(args.patience)} epochs.")
                    break
            
        else:
            best_loss = test_loss_avg
            torch.save(model, model_save_path)
            patience = int(args.patience)

    print("finished training.")