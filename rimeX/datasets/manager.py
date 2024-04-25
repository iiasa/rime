"""Handle remote datasets to be downloaded
"""
import os
from pathlib import Path
import json
import copy
import datetime
import tqdm
import shutil

import rimeX
from rimeX.config import CACHE_FOLDER, CONFIG
from rimeX.logs import logger

import rimeX_datasets
_DEFAULTDATADIR = rimeX_datasets.__path__[0]
ALL_DATASETS = ['werning2024']

def get_downloadpath(relpath=''):
    return Path(CONFIG.get('downloaddir', CACHE_FOLDER / "download")) / relpath


def get_datapath(relpath=''):
    return Path(CONFIG.get('datadir', _DEFAULTDATADIR)) / relpath

MEGABYTES = 1024*1024

def download(url, destination, chunk_size=MEGABYTES):
    partial = Path(str(destination) + ".download")

    if partial.exists():
        response = _resume_download(url, partial, chunk_size)

    else:
        response = _download(url, partial, chunk_size)

    shutil.move(partial, destination)
    return response


def _download(url, destination, chunk_size=MEGABYTES):
    """
    ref: https://realpython.com/python-download-file-from-url/#using-the-third-party-requests-library
    """
    logger.info(f"Download {url} to {destination}")

    import requests
    with requests.get(url, stream=True) as response:
        Path(destination).parent.mkdir(parents=True, exist_ok=True) # create folder if it does not exist
        total = int(response.headers.get('content-length', 0))
        with open(destination, mode="wb") as file, tqdm.tqdm(total=round(total/MEGABYTES,2), unit='MB') as bar :
            for chunk in response.iter_content(chunk_size=chunk_size):
                size = file.write(chunk)
                bar.update(round(size/MEGABYTES,2))

    assert response.ok
    return response


def _resume_download(url, destination, chunk_size=MEGABYTES):
    """
    ref: https://stackoverflow.com/a/22894873/2192272
    """
    import requests

    logger.info(f"Resume download {url} to {destination}")

    with open(destination, mode="ab") as file:
        resume_byte_pos = file.tell()
        resume_header = {'Range': 'bytes=%d-' % resume_byte_pos}

        with requests.get(url, stream=True, headers=resume_header) as response:
            if response.headers.get('content-range'):
                total = int(response.headers.get('content-range', "/0").split("/")[-1])
            else:
                total = int(response.headers.get('content-length', 0)) + int(resume_byte_pos)

            with tqdm.tqdm(
                initial=round(int(resume_byte_pos)/MEGABYTES, 2),
                total=round(total/MEGABYTES, 2),
                unit='MB',
                ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    size = file.write(chunk)
                    bar.update(round(size/MEGABYTES, 2))

    assert response.ok
    return response


def _get_extension(downloaded):
    stripped = archive.strip().split("?")[0]
    if stripped.endswith(".tar.gz"):
        ext = ".tar.gz"
    else:
        basename, ext = os.path.splitext(stripped)
    return ext


def extract_archive(downloaded, path, ext=None, members=None, recursive=False, delete_archive=False):
    # Extract (ref: https://ioflood.com/blog/python-unzip)
    archive = str(downloaded)

    if not ext:
        ext = _get_extension(archive)

    if not ext:
        if archive != path:
            logger.info(f"mv {archive} {path}")
            shutil.move(archive, path)  
        return 

    logger.info(f"Extract {archive} to {path}")

    if ext == ".zip":
        import zipfile
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(path, members=members)
            extract_zip(archive, path, recursive=recursive, members=members)
            if members is None: members = zip_ref.namelist()

    elif ext in (".tar", ".tar.gz"):
        import tarfile
        with tarfile.open(archive, 'r:gz' if ext == ".tar.gz" else 'r') as tar_ref:
            tar_ref.extractall(path, members=members)
            if members is None: members = tar_ref.getmembers()

    elif ext in (".gz"):
        import gzip
        with gzip.open(archive, 'rb') as f_in:
            with open(path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            if members is None: members = [] # unknown method

    elif ext:
        raise NotImplementedError(f"Unknown extension {ext}")

    if recursive:
        for member in members:
            extracted_path = Path(path) / member
            if os.path.isfile(extracted_path) and extracted_path.endswith(('.zip', '.tar', '.gz')):
                ext = _get_extension(extracted_path)
                extract_archive(extracted_path, extracted_path[:-len(ext)], ext=ext, recursive=True, delete_archive=True)

    # delete if everything above went fine
    if delete_archive:
        os.remove(archive)

def get_filename_from_url(url):
    import urllib
    return os.path.basename(urllib.parse.urlparse(url).path)

def require_dataset(name, url=None, extract=True, force_download=None, extract_name=None, members=None, recursive=False, skip_download=False, **metadata):

    filepath = get_datapath(name)
    dataset_json = get_datapath("datasets.json")

    download_folder = get_downloadpath()

    if not skip_download and (not filepath.exists() or force_download):

        download_name = get_filename_from_url(url)
        downloaded = download_folder / download_name
        download_folder.mkdir(exist_ok=True, parents=True)

        if not (downloaded).exists() or force_download:
            downloaded.parent.mkdir()
            download(url, downloaded)

        if extract:
            extract_archive(downloaded, get_datapath(extract_name or name), ext=ext, members=members, recursive=recursive)

        else:
            logger.info(f"mv {downloaded} {extract_path}")
            shutil.move(downloaded, extract_path)

        # also keep a centralized .json that can be git-tracked
        metadata.update({"url": url, "date": str(datetime.datetime.now()), "extract_name": str(extract_name) if extract_name else None, "name": str(name), "ext": ext, "members": members, "recursive": recursive})

        if dataset_json.exists():
            logger.info(f"Update {dataset_json}")
            all_data_info = json.load(open(dataset_json))
        else:
            logger.info(f"Create {dataset_json}")
            all_data_info = { "records": [] }

        # re-arrange as dict to ease update
        by_key = {r['name']:r for r in all_data_info["records"]}
        by_key[metadata['name']] = metadata

        # but save as list of records to make editing less redundant
        all_data_info["records"]  = sorted(by_key.values(), key=lambda r: r['name'])

        # remove redundant fields
        for r in all_data_info["records"]:
            if 'recursive' in r and not r['recursive']: r.pop('recursive')
            if 'members' in r and not r['members']: r.pop('members')
            if 'extract_name' in r and (not r['extract_name'] or not r.get('ext') or r['extract_name'] == r['name']): r.pop('extract_name')
            if 'ext' in r and not r['ext']: r.pop('ext')

        with open(dataset_json, "w") as f:
            json.dump(all_data_info, f, indent=4, sort_keys=True)

    return filepath

def maybe_download_module_data(name):
    " just import a module (which presumably contains a require_dataset statement or download function) "
    from importlib import import_module
    m = import_module(rimeX.datasets.__name__ + "." + name)
    if hasattr(m, "download"):
        m.download()

def main():
    import argparse
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument("--name", nargs='+', default=[], choices=ALL_DATASETS)
    parser.add_argument("--json")
    parser.add_argument("--all", action='store_true')

    o = parser.parse_args()

    # download from json file
    if o.json:
        js = json.load(open(o.json))    
        datasets = js["records"]

        # download select only one out of several
        if o.name:
            datasets = [ dataset for dataset in datasets if dataset[name] in o.name ]

        # download all
        for dataset in datasets:
            require_dataset(**dataset)

    else:
        if o.all:
            o.name = ALL_DATASETS
        if not o.name:
            print("Available datasets are {' '.join(ALL_DATASETS)}. Use the --name NAME or --all flag.")
            parser.exit(1)

        for name in o.name:
            maybe_download_module_data(name)