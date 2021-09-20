from lightgbm import LGBMClassifier


class LGBMModel:
  def __init__(self):
    self.build_model()

  def build_model(self):
    self.model = LGBMClassifier()

  def train(
    self,
    x,
    y,
    validation_split=0.2,
    epochs=10,
    batch_size=32
  ):
    self.model.fit(
      x,
      y
    )

  def predict(
    self,
    x
  ):
    return self.model.predict_proba(x)[:, 1]
