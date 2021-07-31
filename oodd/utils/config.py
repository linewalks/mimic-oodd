import json


def load_config(file_path="config.json"):
  with open("config.json", "r") as f:
    return json.load(f)
