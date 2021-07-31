import pandas as pd

from lifelines import CoxPHFitter


class CoxModel:
  def __init__(self):
    self.model = CoxPHFitter()

  def _make_input(self, x, y, x_cols):
    return pd.concat([
      pd.DataFrame(x, columns=x_cols),
      pd.DataFrame(y, columns=["E", "T"])
    ], axis=1)

  def train(
    self,
    x,
    y,
    x_cols
  ):
    df = self._make_input(x, y, x_cols)

    self.model.fit(df, "T", "E")
    self.model.print_summary()

  def score(
    self,
    x,
    y,
    x_cols
  ):
    df = self._make_input(x, y, x_cols)
    return {
      "log_likelihood": self.model.score(df, "log_likelihood"),
      "concordance_index": self.model.score(df, "concordance_index")
    }
