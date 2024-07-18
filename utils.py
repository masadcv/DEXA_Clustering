import json 
import os


def save_json(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved data to {file_path}")


def load_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

