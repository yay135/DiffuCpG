from generate_samples import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='concat long range samples')
    parser.add_argument('-r', '--raw', type=str, required=True)
    parser.add_argument('-w', '--winsize', type=int, default=1000)
    parser.add_argument('-l', '--longrange', type=str, required=True)
    parser.add_argument('-i', '--index', type=int, default=-1)
    parser.add_argument('-p', '--output', type=str, default="out_longrange")
    args = parser.parse_args()

    with open("data/seq_dict.pkl", "rb") as f:
        chrs_rec = pickle.load(f)

    longrange_file = args.longrange
    raw_file = args.raw
    win_size = int(args.winsize)
    _folder = args.output
    if not os.path.exists(_folder):
        os.mkdir(_folder)
    methyl_col = int(args.index)

    raw_df = pd.read_csv(raw_file)

    ranges = pd.read_csv(args.longrange, dtype={"c0": str, "c1": str})
    for _, row in tqdm(ranges.iterrows(), total=len(ranges)):
        c0, c1, s0, s1 = row["c0"], row["c1"], row["s0"], row['s1']
        e0, e1 = s0 + win_size - 1, s1 + win_size - 1
        sub_df0 = raw_df.loc[(raw_df["chr"] == f"chr{c0}") & (
            raw_df["locs"] >= s0) & (raw_df['locs'] <= e0), ]
        sub_df1 = raw_df.loc[(raw_df["chr"] == f"chr{c1}") & (
            raw_df["locs"] >= s1) & (raw_df['locs'] <= e1), ]

        def gen_sample_from_win(df_win, start, end, methyl_col, seq):
            methyl = np.zeros((end-start+1,)) - 1
            methyl[df_win.iloc[:, 1] - start] = df_win.iloc[:, methyl_col]
            methyl = methyl.reshape(1, -1)
            # form other channels
            ncol = df_win.shape[1]
            if ncol > 3:
                bio_cols = np.delete(np.arange(ncol), (0, 1, methyl_col))
                bio = np.zeros((end-start+1, len(bio_cols))) - 1
                bio[df_win.iloc[:, 1] - start] = df_win.iloc[:, bio_cols]
                bio = bio.T
                chns = [bio, methyl]
            else:
                chns = [methyl]

            res = rbind_seq_methy(seq, start, end, chns)
            return res
        
        sample0 = gen_sample_from_win(sub_df0, s0, e0, methyl_col, chrs_rec[f"chr{c0}"])
        sample1 = gen_sample_from_win(sub_df1, s1, e1, methyl_col, chrs_rec[f"chr{c1}"])
        res = np.concatenate((sample0, sample1), axis=1)

        pd.DataFrame(res).to_csv(
            f"{_folder}/record_chr{c0}_{s0}_chr{c1}_{s1}_chn{len(res)}.csv", index=False, header=False)
