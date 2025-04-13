import os
import requests
from tqdm import tqdm
import zipfile

# Statics
urls = [
#    "https://zenodo.org/records/3723295/files/annotations.csv",
#    "https://zenodo.org/records/3723295/files/candidates.csv",
#    "https://zenodo.org/records/3723295/files/candidates_V2.zip",
#    "https://zenodo.org/records/3723295/files/evaluationScript.zip",
#    "https://zenodo.org/records/3723295/files/sampleSubmission.csv",
#    "https://zenodo.org/records/3723295/files/seg-lungs-LUNA16.zip",
#    "https://zenodo.org/records/3723295/files/subset0.zip",
#    "https://zenodo.org/records/3723295/files/subset1.zip",
    "https://zenodo.org/records/3723295/files/subset2.zip",
    "https://zenodo.org/records/3723295/files/subset3.zip",
    "https://zenodo.org/records/3723295/files/subset4.zip",
    "https://zenodo.org/records/3723295/files/subset5.zip",
    "https://zenodo.org/records/3723295/files/subset6.zip",
    "https://zenodo.org/records/4121926/files/subset7.zip",
    "https://zenodo.org/records/4121926/files/subset8.zip",
    "https://zenodo.org/records/4121926/files/subset9.zip"
]

output_dir = "luna16_data"

# Functions
def download_file(url, output_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
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

def unzip_file(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    except zipfile.BadZipFile:
        print(f"Skipped (not a valid zip file): {zip_path}")

# Mian
for url in urls:
    filename = os.path.basename(url)
    file_path = os.path.join(output_dir, filename)
    print(f"Downloading {filename}...")
    download_file(url, file_path)
    if filename.lower().endswith('.zip'):
        print(f"Unzipping {filename}...")
        unzip_file(file_path, output_dir)
    else:
        print(f"Skipped unzip (not a zip): {filename}")