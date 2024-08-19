import requests
from tqdm import tqdm
import gzip
import shutil

def download_grch37_primary_assembly(url: str, local_file_path: str):
    """
    Downloads the GRCh37 primary assembly file from the given URL and saves it to the specified local file path.

    Parameters:
    - url: str : The URL to download the file from.
    - local_file_path: str : The local path to save the downloaded file.
    """
    # Send a HEAD request to get the file size
    response = requests.head(url)
    file_size = int(response.headers.get('content-length', 0))

    # Start the download with a progress bar
    with requests.get(url, stream=True) as r, open(local_file_path, 'wb') as f, tqdm(
        total=file_size, unit='B', unit_scale=True, desc=local_file_path) as pbar:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    print(f"Downloaded {local_file_path}")

# unzip a gz file
def gunzip_file(input_file: str, output_file: str):
    """
    Decompresses a .gz file.

    Parameters:
    - input_file: str : The path to the .gz file to be decompressed.
    - output_file: str : The path where the decompressed file will be saved.
    """
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Decompressed {input_file} to {output_file}")