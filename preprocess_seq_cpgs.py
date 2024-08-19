
from download_assembly import *
from Bio import SeqIO
import pickle
import os

if not os.path.exists("data"):
    os.mkdir("data")

url = "https://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/dna/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
local_file_path = f"data/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"
    
if not os.path.exists("data/Homo_sapiens.GRCh37.dna.primary_assembly.fa.gz"):
    print("downlaoding required data ...")
    # raw data url https://ftp.ensembl.org/pub/grch37/current/fasta/homo_sapiens/dna/
    download_grch37_primary_assembly(url, local_file_path)    
    gunzip_file(local_file_path, local_file_path[:len(local_file_path)-3])

local_file_path = local_file_path[:len(local_file_path)-3]
    
records = SeqIO.parse(local_file_path, format="fasta")

chrs = ["chr1", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15", "chr16", "chr17",
        "chr18", "chr19", "chr2", "chr20", "chr21", "chr22", "chr3", "chr4", "chr5", "chr6",
        "chr7", "chr8", "chr9", "chrX", "chrY"]

chrs_rec = {}

def find_cg_indices(sequence):
    indices = []
    for i in tqdm(range(len(sequence) - 1), total=len(sequence)):
        if sequence[i:i+2] == "CG":
            indices.append(i+1)
    return indices

# extract sequence data
for rec in records:
    key = "chr"+str(rec.description).split()[0]
    if key in chrs:
        chrs_rec[key] = rec.seq
        
with open("data/seq_dict.pkl", "wb") as f:
    pickle.dump(chrs_rec, f)

# extract cpg sites
merged = {ch:find_cg_indices(str(chrs_rec[ch])) for ch in chrs}

with open("data/cpgs_dict.pkl", "wb") as f:
    pickle.dump(merged, f)