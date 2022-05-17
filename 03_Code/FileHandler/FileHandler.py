import os
import json
import pandas as pd
from glob import glob


def read_csv(file_path):
    return pd.read_csv(file_path)


def read_txt(file_path):
    return set(open(file_path, encoding='utf-8').readlines())


def read_arabic_csv(file_path):
    df = pd.read_csv(file_path, lineterminator='\n', encoding='utf-8')
    df.columns = [col.replace('\r', '') for col in df.columns]
    return df.replace({r'\r': ''}, regex=True)


def save_json_file(dictionary, folder_directory, file_name):
    file_path = os.path.join(folder_directory, file_name)
    with open(file_path, 'w') as fp:
        json.dump(dictionary, fp)


def load_json_file(file_path):
    with open(file_path, 'r') as fp:
        return json.load(fp)


def read_all_directory(directory, extensions):
    return glob("{}/*.{}".format(directory, extensions))