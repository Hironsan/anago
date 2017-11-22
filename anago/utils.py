import io
import os
import zipfile

import requests


def download(url, save_dir='.'):
    """Downloads trained weights, config and preprocessor.

    Args:
        url (str): target url.
        save_dir (str): store directory.
    """
    print('Downloading...')
    r = requests.get(url, stream=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as f:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        f.extractall(save_dir)
    print('Complete!')
