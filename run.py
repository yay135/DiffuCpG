import subprocess

# # get raw methylation data
subprocess.run(["bash", "gdown.sh"])

# # install dependencies
subprocess.run(["python", "-m", "pip", "install", "-r", "req.txt"])

# # get sequence data
subprocess.run(["python", "preprocess_seq_cpgs.py"])

# generate some training samples
# window size defaults to 1kb
# "raw" is the raw methylation data folder
# -i sets which column is the methylation level in the raw files, the rest of the columns besides 0(chromo), 1(loc) and i are extra feature channels. 
# -t sets when generating a window(sample), how many missing methylation states are allowed
subprocess.run(["python", "generate_samples.py", "-d", "raw", "-i", "3", "-t", "0", "-p", "train"])

# generate some samples for imputation
# windows size is default to 1kb
# "raw" is the methylation raw files folder
# -i sets which column is the methylation level in the raw files, the rest of the columns besides 0(chromo), 1(loc) and i are extra feature channels. 
# -t sets when generating a window(sample) how many missing methylation states are allowed, set t = -1 to allow as many missing as possible for testing imputation.
subprocess.run(["python", "generate_samples.py", "-d", "raw", "-i", "3", "-t", "-1", "-p", "test"])


# you can also generate long range interaction samples by combine two 1kb sample together
# -r specify which raw file to generate from
# -l specify the interacting ranges
subprocess.run(["python", "generate_samples_concat.py", "-r", 
                "raw/54T1B.R1.Cutadapt_bismark_bt2_pe.CpG_report.tx_processed.csv", 
               "-l", "data/long_range_interaction.csv"])


# training 
# -t sets the folder containing training samples
# -w sets the window size to 1000 base pairs
# -c sets the channel size to 6, A + T + C + G + Methyl + island
# the raw files contains an extra feature column indicates whether a cpg is in a cpg island.
subprocess.run(["python", "diffusion.py", "-t", "train", "-f", "model", "-w", "1000", "-c", "6"])

# testing
# -t sets the folder containing testing samples
# -w sets the window size to 1000 base pairs
# -c sets the channel size to 6,  A + T + C + G + Methyl + island
# the script generates a default output folder
# this script will use trained diffusion model and a inpainting alogrithm to impute missing methylation states for all cpg sites in a 1kb window
subprocess.run(["python", "diffusion_inpainting.py", "-t", "test", "-f", "model", "-w", "1000", "-c", "6"])