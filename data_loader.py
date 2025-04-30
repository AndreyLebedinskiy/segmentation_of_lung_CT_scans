import os
import requests
from tqdm import tqdm
import zipfile
import tarfile
import bz2


# Statics
luna_urls = [
    "https://zenodo.org/records/3723295/files/annotations.csv",
    "https://zenodo.org/records/3723295/files/candidates.csv",
    "https://zenodo.org/records/3723295/files/candidates_V2.zip",
    "https://zenodo.org/records/3723295/files/evaluationScript.zip",
    "https://zenodo.org/records/3723295/files/sampleSubmission.csv",
    "https://zenodo.org/records/3723295/files/seg-lungs-LUNA16.zip",
    "https://zenodo.org/records/3723295/files/subset0.zip",
    "https://zenodo.org/records/3723295/files/subset1.zip",
    "https://zenodo.org/records/3723295/files/subset2.zip",
    "https://zenodo.org/records/3723295/files/subset3.zip",
    "https://zenodo.org/records/3723295/files/subset4.zip",
    "https://zenodo.org/records/3723295/files/subset5.zip",
    "https://zenodo.org/records/3723295/files/subset6.zip",
    "https://zenodo.org/records/4121926/files/subset7.zip",
    "https://zenodo.org/records/4121926/files/subset8.zip",
    "https://zenodo.org/records/4121926/files/subset9.zip"
]

vessel_urls = [
#    "https://zenodo.org/records/8055066/files/VESSEL12_01-05.tar.bz2",
#    "https://zenodo.org/records/8055066/files/VESSEL12_01-20_Lungmasks.tar.bz2",
#    "https://zenodo.org/records/8055066/files/VESSEL12_06-10.tar.bz2",
#    "https://zenodo.org/records/8055066/files/VESSEL12_11_15.tar.bz2",
#    "https://zenodo.org/records/8055066/files/VESSEL12_16-20.tar.bz2",
    "https://zenodo.org/records/8055066/files/VESSEL12_ExampleScans.tar.bz2"
]

luna_output_dir = "segmentation_of_lung_CT_scans/data/luna16_data"
vessel_output_dir = "segmentation_of_lung_CT_scans/data/vessel12_data"


# Functions
def download_file(url, output_path):
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
    output_dir_bz2 = os.path.join(output_dir, os.path.basename(path)[:-4])
    with bz2.BZ2File(path, 'rb') as file:
        content = file.read()
    with open(output_dir_bz2, 'wb') as f_out:
        f_out.write(content)
   
    if output_dir.endswith('.tar'):
        tar_output_path = output_dir[:-4]
        untar_file(output_dir_bz2, output_dir)


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


# Mian
#load_and_open(luna_urls, luna_output_dir)
load_and_open(vessel_urls, vessel_output_dir)