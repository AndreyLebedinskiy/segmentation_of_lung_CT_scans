import os
import requests
from tqdm import tqdm
import zipfile
import tarfile
import bz2


def download_file(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    if response.status_code != 200:
        raise Exception(f"Failed to download {url}: {response.status_code}")

    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def unzip_file(path, output_dir):
    try:
        with zipfile.ZipFile(path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
    except zipfile.BadZipFile:
        print("Not a valid zip file")


def untar_file(path, output_dir):
    try:
        with tarfile.open(path, 'r:*') as tar:
            tar.extractall(path=output_dir)
    except tarfile.TarError:
        print("Not a valid tar file")


def unbz2_file(path, output_dir):
    tar_path = os.path.join(output_dir, os.path.basename(path)[:-4])
    with bz2.BZ2File(path, 'rb') as file, open(tar_path, 'wb') as out_file:
        out_file.write(file.read())
    
    if tar_path.endswith('.tar'):
        untar_file(tar_path, output_dir)
        os.remove(tar_path)


def load_and_open(urls, output_dir):
    for url in urls:
        filename = os.path.basename(url)
        file_path = os.path.join(output_dir, filename)
        print(f"Downloading {filename}")
        download_file(url, file_path)

        if filename.lower().endswith('.zip'):
            unzip_file(file_path, output_dir)
        elif filename.lower().endswith('.bz2'):
            unbz2_file(file_path, output_dir)
        else:
            print(f"{filename} not an archive")
        os.remove(file_path)