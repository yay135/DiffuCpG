export FILEID='1CRvLc_Z6O0WSS9u4PAy_Ps_qve8fZkWC'
export FILENAME='raw_DiffuCpG.tar.gz'

wget https://bootstrap.pypa.io/get-pip.py

python -m pip install gdown
gdown $FILEID

echo Download complete. Extracting ... 
tar -xvf $FILENAME
