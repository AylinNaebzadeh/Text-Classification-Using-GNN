import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil
import numpy as np 
import pandas as pd


CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'arxiv-cs-papers-multi-label-classification-200k-v1:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F3372297%2F5864998%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240225%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240225T060706Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D2e28c8912c28a40d02aef686aea5ed4954336b4a6effae6a150b99781f164b25e3622947563f213bd401e189db50c6df99dd0e78fb62e70d37897c9a7724b53b758815abe58deebcc6475efc1f5ff035bd919945e6a732ff8b6c5ab55b37b694b1ae5f73d34277ed58f889a4f82fdd94ad66faf8ec04a2e81c7b929184d2709c5c42ffaa6f6da57715fffab00be0dce0cc081925a28e9cbd539606dc700668099aef6e71d880b6b9c915bcd40054047a6f8900d98910c38c29905e81fbd1f59380dfd79e9cdd179c1423409bd31da3b7f7e409728e1b18e751104830b372faa6d4880969057be836e8b362047f91b999278ffd6cd841563ed6f364d081e58306'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

# umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
    os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
    pass
try:
    os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
    pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
                with ZipFile(tfile) as zfile:
                    zfile.extractall(destination_path)
            else:
                with tarfile.open(tfile.name) as tarfile:
                    tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue


print('Data source import complete.')



