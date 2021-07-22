import json

from oodd.data.mimic import MIMIC3
from oodd.model.rnn import RNNModel


if __name__ == "__main__":

  with open("config.json", "r") as f:
    config = json.load(f)

  data_loader = MIMIC3(config["DB_URI"])
  x, y, data_key_df, seq_len_list = data_loader.get_rnn_inputs()
  print(x.shape)
  print(y.shape)

  train_data, test_data = data_loader._split_by_gender(
    x,
    y,
    data_key_df,
    seq_len_list,
    train_gender="F"
  )

  model = RNNModel(
    x.shape[-1],
    y.shape[-1]
  )
  model.train(
    train_data["x"],
    train_data["y"],
    epochs=1
  )

  y_pred = model.predict(
    test_data["x"]
  )

  data_loader.close()
