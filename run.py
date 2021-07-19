import json

from oodd.data.mimic import MIMIC3


if __name__ == "__main__":

  with open("config.json", "r") as f:
    config = json.load(f)

  data_loader = MIMIC3(config["DB_URI"])
  data_loader.get_merged_data()
