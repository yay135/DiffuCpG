# DiffuCpG
## 1. Introduction

![Arch figure](https://github.com/yay135/DiffuCpG/blob/main/methylation_imputation_arch.jpg?raw=true)

In this study, we used a generative AI diffusion model to address missing methylation data. We trained the model with Whole-Genome Bisulfite Sequencing data from 26 acute myeloid leukemia samples and validated it with Reduced Representation Bisulfite Sequencing data from 93 myelodysplastic syndrome and 13 normal samples. Additional testing included data from the Illumina 450k methylation array and Single-Cell Reduced Representation Bisulfite Sequencing on HepG2 cells. Our model, DiffuCpG, outperformed previous methods by integrating a broader range of genomic features, utilizing both short- and long-range interactions without increasing input complexity. It demonstrated superior accuracy, scalability, and versatility across various tissues, diseases, and technologies, providing predictions in both binary and continuous methylation states.

In this repository, we deposit the code used to build the diffusion models along with necessary example datasets to train and test a diffusion model for methylation imputation purposes.

## Docker builds of our best models are now available!
### Install Docker
Install Docker using the following link:  
https://docs.docker.com/engine/install/  
Recommended system specs: Debian 12 bookworm with 16GB RAM or more.  
### Run the Models  
``docker pull yay135/diffucpg``   
prepare your input data directory, use the following command to print a example input data directory   
``docker run --rm yay135/diffucpg -e true``   
assume your data directory name is "input_data"   
in windows   
``docker run --gpus all -v .\input_data\:/data --rm yay135/diffucpg``  
in unix or linux
``docker run --gpus all -v ./input_data:/data --rm yay135/diffucpg``  

### other docker options
-d or --device : select which cuda device to run with, default is 0      
-m or --mingcpg : scan your methyl array, limit only imputing windows with at least m non-missing methyl values, default is 10   
-o or --overlap : set number of impute epochs, shift window locations between epochs, get mean imputed values for each CpG location, default is 2   
example:   
``docker run --gpus all -v ./input_data:/data --rm yay135/diffucpg -d 1 -m 5 -o 3``   
use cuda device 1, min number of non-missing methyl values in a window is 5, overlap epochs 3

**The following tutorials are for non-docker usages.**

## 2. Data and Models
Example datasets are available for download using "gdown.sh". The example datasets only contain WGBS methylation data. The model is the DDPM diffusion model, the repository contains a complete implementation for 1-dimensional input. Please refer to https://arxiv.org/abs/2006.11239  and https://huggingface.co/blog/annotated-diffusion for more details.   
## 3. How to use this repo
### 3.1 System Requirements
The number of steps in the diffusion process is set to 2000. Imputing a sample requires 2000 steps. Gpu acceleration is prefered. 16GB of RAM is required. The code is fully tested and operational on the following platform:   

Distributor ID: Debian   
Description:    Debian GNU/Linux 12 (bookworm)   
Release:        12   
Codename:       bookworm   

### 3.2 Clone the Current Project
Run the following command to clone the project.  
``git clone https://github.com/yay135/DiffuCpG.git``  
### 3.4 Configure Environment
Make sure you have the following software installed in your system:   
Python 3.9+   
Pytorch 2.0.1+ 
### 3.4 Run Training and Testing
``python run.py``   
The script will download necessary data and install dependencies automatically.   

## 4 Data ans script Details
### 4.1 RAW Data
The methylation arrays downloaded are in the folder "raw", each file is a methylation array. The first 2 columns are "chromosome" and "location". The assembly used for mapping in our project is the "GRCH37 primary assembly". It is also downloaded automatically. The rest of the columns in each file are methylation levels(required) and other biological data (optional) you wish to incorporate to enhance the model. These files in the raw folder are the initial inputs for pipeline,if you wish to use your own data, it must be configured as such before running the pipeline. 

### 4.2 Generate sample
Use script "generate_samples.py" to generate samples for training and testing.   
The model can not directly read and impute a methylation array file. Instead, each methylation array is divided into windows, each window is 1kb (1000 base pairs) in length, and each training testing sample is generated from a window. Each sample contains at least 5 channels. the first 4 is the sequence one-hot encoding, the 5th is the methylation data. If a base pair location is not a CpG location, the methylation data value for it is "-1". If a CpG's methylation data is missing or waiting for imputaion, its value is also "-1". Other biological data can be added as extra channels.   

'-d' or '--folder': specify raw data folder   
'-i' or '--index' : which column in a raw file is the methylation array   
'-t' or '--tol' : how many missing methylation value is tolerated   
(we recommend 0 for generating training samples and -1 for generating testing samples, 0 will force the script to only select from windows with no missings, -1 will tolerate missing as much as possible.)     
'-c' or '--chr' : limit which chromosome to use, default is "chr#" to use all chromosomes   
'-w' or '--winsize' : what window size to use, default is 1000 
'-m' or '--mincpg': force generate from window to have a minimum number of CpGs, default is 10
'-n' or '--nsample': number of samples to generate per chromosome
'-p' or '--output': samples output folder, default is "out"

### 4.3 Training script
Use diffusion.py to train and test a DDPM model using the generated samples  
'-t' or '--train_folder' : the folder containing the training samples   
'-f' or '--model_folder' : the model folder, will be created if it does not exist      
'-w' or '--win_size' : window size of each sample, default is 1000   
'-c' or '--channel': channel size of each sample   
'-d' or '--cuda_device' : if you have multiple cuda gpus, select which gpu to use, default is 0   
"-e" or "--epoch" : how many epochs for training, default is 2000   
"-s" or "--earlystop" : whether to use "early stopping" during training, default is False   
"-p" or "--patience" : patience for early stopping, default is 10   

### 4.4 Imputation 
Use diffusion_inpainting.py to perform imputation on generated samples.   
'-t' or '--test_folder' : the folder containing samples for imputation   
'-o' or '--out_folder': imputed output folder name, default="inpainting_out"   
'-w' or '--win_size' : window size of each sample, default is 1000   
'-c' or '--channel': channel size of each sample   
'-d' or '--cuda_device' : if you have multiple cuda gpus, select which gpu to use, default is 0   
 
## Team
If you have any questions or concerns about the project, please contact the following team member:
Fengyao Yan fxy134@miami.edu 
