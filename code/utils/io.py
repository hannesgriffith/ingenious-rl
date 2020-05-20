import os
import json

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def make_dir_if_not_exists(path):
    if not os.path.isdir(path):
        os.mkdir(path)
