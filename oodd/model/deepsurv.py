import pandas as pd

from tfdeepsurv import dsnn
from tfdeepsurv.datasets import survival_df


class DeepSurvModel:
  def __init__(
    self,
    num_features,
    dense_nodes=[32, 32, 1]
  ):
    self.num_features = num_features
    self.dense_nodes = dense_nodes

    self.build_model()

  def build_model(self):
    nn_config = {
        "learning_rate": 5e-4, #0.1,
        "learning_rate_decay": 1.0,
        "activation": "relu",
        "L1_reg": 0.0, # 3.4e-5, 
        "L2_reg": 0.0, # 8.8e-5, 
        "optimizer": "adam",
        "dropout_keep_prob": 0.5,
        "seed": 1
    }

    self.model = dsnn(
      self.num_features, 
      self.dense_nodes,
      nn_config
    )
    self.model.build_graph()

  def _make_input(self, x, y, x_cols):
    df = pd.concat([
      pd.DataFrame(x, columns=x_cols),
      pd.DataFrame(y, columns=["E", "T"])
    ], axis=1)
    return survival_df(df, t_col="T", e_col="E")

  def train(
    self,
    x,
    y,
    x_cols
  ):
    df = self._make_input(x, y, x_cols)

    watch_list = self.model.train(
      df[x_cols],
      df[["Y"]],
      num_steps=2000,
      num_skip_steps=100,
    )
    print("Last metric", watch_list["metrics"][-1])

  def score(
    self,
    x,
    y,
    x_cols
  ):
    df = self._make_input(x, y, x_cols)
    return self.model.evals(df[x_cols], df[["Y"]])
