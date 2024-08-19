from diffusion import *

mse_loss = nn.MSELoss()
# generate a mask for all missing cpg locs
def mask_gen_missing(batch):
    batch = batch.numpy()
    res = []
    for sample in batch:
        c = np.where(sample[2] == 1)[0]
        g = np.where(sample[3] == 1)[0] - 1
        cpg = np.intersect1d(c, g)
        cpg_avai = np.where(sample[4] != -1)[0]
        missing = cpg[np.logical_not(np.isin(cpg, cpg_avai))]
        mask = torch.full(sample.shape, False)
        mask[-1, missing] = True
        mask = torch.unsqueeze(mask, 0)
        res.append(mask)
    mask = torch.concat(res, 0)
    return mask



if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='impute using a DDPM model')
    parser.add_argument('-t', '--test_folder', type=str, required=True)
    parser.add_argument('-o', '--out_folder', type=str, default="inpainting_out")
    parser.add_argument('-f', '--model_folder', type=str, required=True)
    parser.add_argument('-w', '--win_size', type=int, required=True)
    parser.add_argument('-c', '--channel', type=int, required=True)
    parser.add_argument('-d', '--cuda_device', type=int, default=0)

    args = parser.parse_args()

    win_size = args.win_size
    batch_size = 64

    if torch.cuda.is_available():
        cuda_device = int(args.cuda_device)
        if 0 <= cuda_device < torch.cuda.device_count():
            device = torch.device(f"cuda:{cuda_device}")
        else:
            device = torch.device("cuda:0")

    else:
        device = torch.device('cpu')

    print(f"using device: {device}")

    out_folder = args.out_folder
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    test_folder = args.test_folder
    vali_samples = os.listdir(test_folder)
    vali_total = math.ceil(len(vali_samples)/batch_size)
    sample_paths_vali = [f"{test_folder}/{p}" for p in vali_samples]
    
    model_folder = args.model_folder
    model_save_path = os.path.join(model_folder, "diffusion_1d")

    model = torch.load(model_save_path, map_location="cpu")
    model = model.to(device)
    model.eval()

    def iter_sample(data, fid):
        assert(len(data) == len(fid))
        def iter():
            for i, sample in enumerate(data):
                yield torch.tensor(sample.values, dtype=torch.float), torch.tensor(fid[i])
        return iter

    print("loading data ...")
    vali_data = list(map(lambda x: pd.read_csv(x, header=None), tqdm(sample_paths_vali)))
    vali_dataloader = torch.utils.data.DataLoader(\
        IterDataset(iter_sample(vali_data, [i for i in range(len(vali_samples))])), batch_size=batch_size)

    for step, (batch, fid) in enumerate(vali_dataloader):
        prog = step*batch_size + len(batch)
        print(f"{prog}/{len(vali_samples)}")
        assert(win_size == batch.shape[2])
        mask = mask_gen_missing(batch)
        imp_mask = torch.logical_not(mask)
        impainted = inpainting(model, batch, imp_mask)

        for i, idx in enumerate(fid):
            fname = vali_samples[idx]
            imp_sample = impainted.cpu().numpy()[i]
            np.savetxt(os.path.join(out_folder, f"imputed_{fname}"), imp_sample, delimiter=",", fmt="%.2f")
