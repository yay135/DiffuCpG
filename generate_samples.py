from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import random
import pickle
import re
import os


chrs = ["chr1", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17",
        "chr18", "chr19", "chr2", "chr20", "chr21", "chr22", "chr3", "chr4", "chr5", "chr6",
        "chr7", "chr8", "chr9", "chrX", "chrY"]


# combine sequence data
def rbind_seq_methy(seq, start, end, channels):
    # form base vector
    res_seq = str(seq[start-1: end])
    # onehot encode ATCG
    # A
    res_seq0 = np.zeros((len(res_seq),))
    A_loc = [m.start() for m in re.finditer("A", res_seq)]
    res_seq0[A_loc] = 1
    res_seq0 = res_seq0.reshape(1, -1)

    # T
    res_seq1 = np.zeros((len(res_seq),))
    T_loc = [m.start() for m in re.finditer("T", res_seq)]
    res_seq1[T_loc] = 1
    res_seq1 = res_seq1.reshape(1, -1)

    # C

    res_seq2 = np.zeros((len(res_seq),))
    C_loc = [m.start() for m in re.finditer("C", res_seq)]
    res_seq2[C_loc] = 1
    res_seq2 = res_seq2.reshape(1, -1)

    # G
    res_seq3 = np.zeros((len(res_seq),))
    G_loc = [m.start() for m in re.finditer("G", res_seq)]
    res_seq3[G_loc] = 1
    res_seq3 = res_seq3.reshape(1, -1)

    res = np.concatenate([res_seq0, res_seq1, res_seq2,
                         res_seq3] + channels, axis=0)

    return res


def binary_search_smaller_index(arr, target):
    low = 0
    high = len(arr) - 1
    result = -1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            result = mid
            low = mid + 1
        else:
            high = mid - 1

    return result

# sliding window scan for qualified 1kb windows within methylation data points
def scan_for_window(all_cpgs_locs, df_chr, leng, methyl_col):
    cpgs = df_chr.iloc[:, 1].values
    cpgs = cpgs[df_chr.iloc[:, methyl_col] != -1]   
    skips = all_cpgs_locs[~np.isin(all_cpgs_locs, cpgs)]

    skips_count = len(skips[(skips >= 1) & (skips <= 1+len_win-1)])
    cpgs_count = len(cpgs[(cpgs >= 1) & (cpgs <= 1+len_win-1)])

    skips = set(skips)
    cpgs = set(cpgs)

    ans = []

    pbar = tqdm(range(leng), total=leng)
    # sliding window find qaulified win
    for i in pbar:
        start = i + 1
        if skips_count <= tol and cpgs_count >= min_cpg:
            ans.append((start, start + len_win - 1))
        nx = start + len_win

        if nx <= leng:
            if nx in skips:
                skips_count += 1
            elif nx in cpgs:
                cpgs_count += 1
        else:
            break
        if start in skips:
            skips_count -= 1
        elif start in cpgs:
            cpgs_count -= 1

    return ans


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='generate samples')
    parser.add_argument('-d', '--folder', type=str, required=True)
    #specify which column is the methylation level for your data, default to the last column
    parser.add_argument('-i', '--index', type=int, default=-1)
    parser.add_argument('-t', '--tol', type=int, default=0)
    parser.add_argument('-c', '--chr', choices=chrs + ["chr#"], default="chr#")
    parser.add_argument('-w', '--winsize', type=int, default=1000)
    parser.add_argument('-m', '--mincpg', type=int, default=10)
    parser.add_argument('-n', '--nsample', type=int, default=100)
    parser.add_argument('-p', '--output', type=str, default="out")

    args = parser.parse_args()
    
    # limit chromosome, if chr is chr#, include all chrs
    chr_select = args.chr
    # raw data folder
    folder = args.folder
    files = [os.path.join(folder, file) for file in os.listdir(folder)]
    # number of sample to generate per chromosome
    n_sample = int(args.nsample)
    # negative indicates select as much samples as possible for each chromosome
    if n_sample < 0:
        n_sample = float("inf")
    # the length of (window) each sample in base pairs
    len_win = int(args.winsize)
    # minum number of cpg methyl required to select the window
    min_cpg = int(args.mincpg)
    # max number of missing cpg methyl allowed to select the window
    tol = int(args.tol) 
    # indicates can have any number of missing methylation in a window
    if tol < 0:
        tol = float('inf')
    
    methyl_col = int(args.index)
    
    with open("data/seq_dict.pkl", "rb") as f:
        chrs_rec = pickle.load(f)

    with open("data/cpgs_dict.pkl", "rb") as f:
        cpgs_rec = pickle.load(f)
    
    # process samples using sequence data, ci (confidence interval) data
    _folder = args.output
    if not os.path.exists(_folder):
        os.mkdir(_folder)
    for i, path in enumerate(files):    
        print(f"{i+1}/{len(files)} {path} ...")
        # methyl_df must have the following structure
        # 0:chr, 1:loc, 3:methylation, 4+ : additional biological data, missing methylation must be filled with -1 or skipped
        methyl_df = pd.read_csv(path)

        if chr_select == "chr#":
            chx = chrs
        else:
            chx = [chr_select]

        for ch in chx:
            print(ch)
            methyl_df_chr = methyl_df.loc[methyl_df.iloc[:, 0] == ch, ]
            seq = chrs_rec[ch]
            wins = scan_for_window(np.array(cpgs_rec[ch]), methyl_df_chr, len(seq), methyl_col)
            wins = random.sample(wins, k=min(len(wins), n_sample))
            
            for start, end in tqdm(wins):
  
                df_win = methyl_df.loc[(methyl_df.iloc[:, 1] >= start) & (
                    methyl_df.iloc[:, 1] <= end), ]
                # form methyl channel                
                methyl = np.zeros((end-start+1,)) - 1
                methyl[df_win.iloc[:, 1] - start] = df_win.iloc[:, methyl_col]
                methyl = methyl.reshape(1, -1)
                # form other channels
                if methyl_df.shape[1] > 3:
                    bio_cols = np.delete(np.arange(methyl_df.shape[1]), (0, 1, methyl_col))
                    bio = np.zeros((end-start+1, len(bio_cols))) - 1
                    bio[df_win.iloc[:, 1] - start] = df_win.iloc[:, bio_cols]
                    bio = bio.T
                    chns = [bio, methyl]
                else:
                    chns = [methyl]

                res = rbind_seq_methy(seq, start, end, chns)

                pd.DataFrame(res).to_csv(
                    f"{_folder}/record_{ch}_{start}_{os.path.basename(path).split('.')[0]}_chn{len(res)}.csv", index=False, header=False)

