from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense


class RNNModel:
  def __init__(
    self,
    num_features,
    num_classes,
    rnn_nodes=[64, 64]
  ):
    self.num_features = num_features
    self.num_classes = num_classes
    self.rnn_nodes = rnn_nodes

    self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(Input(shape=(None, self.num_features)))

    for node in self.rnn_nodes:
      model.add(LSTM(node, return_sequences=True))

    last_activation = "sigmoid" if self.num_classes == 1 else "softmax"
    model.add(Dense(self.num_classes, activation=last_activation))

    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    self.model = model
    self.ood_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

  def train(
    self,
    x,
    y,
    validation_split=0.2,
    epochs=1,
    batch_size=32
  ):
    self.model.fit(
      x,
      y,
      epochs=epochs,
      validation_split=validation_split,
      batch_size=batch_size
    )

  def predict(
    self,
    x
  ):
    return self.model.predict(x)

  def predict_ood(
    self,
    x
  ):
    return self.ood_model.predict(x)
