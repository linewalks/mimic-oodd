
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Input, LSTM, Dense, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam


class RNNModel:
  def __init__(
    self,
    num_features,
    num_classes,
    rnn_nodes=[64, 64],
    return_sequences=True
  ):
    self.num_features = num_features
    self.num_classes = num_classes
    self.rnn_nodes = rnn_nodes
    self.return_sequences = return_sequences

    self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(Input(shape=(None, self.num_features)))

    for idx, node in enumerate(self.rnn_nodes):
      is_last = idx == len(self.rnn_nodes) - 1
      return_sequences = self.return_sequences or not is_last
      model.add(LSTM(node, return_sequences=return_sequences))

    last_activation = "sigmoid" if self.num_classes == 1 else "softmax"
    model.add(Dense(
      self.num_classes,
      activation=last_activation,
      kernel_initializer=TruncatedNormal(stddev=0.01)
    ))

    optimizer = Adam(lr=0.0002)
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    model.summary()

    self.model = model
    self.ood_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

  def train(
    self,
    x,
    y,
    validation_split=0.2,
    epochs=5,
    batch_size=64
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
