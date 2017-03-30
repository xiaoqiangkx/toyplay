# -*- coding: utf-8 -*-


"""
@version: 1.0
@author: clark
@file: fetch_house.py
@time: 2017/3/30 21:53
@change_time:
1.2017/3/30 21:53
"""
import os
import tarfile
import requests


EXTRACT_HOUSING_PATH = "../data/housing"
HOUSING_URL = "https://github.com/ageron/handson-ml/raw/master/datasets/housing/housing.tgz"


def download_file(url):
    print url

    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    with open(local_filename, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:   # filter out keep-alive new chunks
                f.write(chunk)

    return local_filename


def fetch_housing_data(housing_url=HOUSING_URL, extract_path=EXTRACT_HOUSING_PATH):
    if not os.path.isdir(extract_path):
        os.makedirs(extract_path)

    filename = download_file(housing_url)
    housing_tgz = tarfile.open(filename)
    housing_tgz.extractall(path=extract_path)
    housing_tgz.close()
    os.remove(filename)


if __name__ == '__main__':
    fetch_housing_data()
