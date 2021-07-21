import json

from oodd.data.mimic import MIMIC3


if __name__ == "__main__":

  with open("config.json", "r") as f:
    config = json.load(f)

  data_loader = MIMIC3(config["DB_URI"])
  x, y, data_key_df, seq_len_list = data_loader.get_rnn_inputs()
  print(x.shape)
  print(y.shape)

  data_loader._split_by_gender(
    x,
    y,
    data_key_df,
    seq_len_list,
    train_gender="F"
  )

  data_loader.close()
