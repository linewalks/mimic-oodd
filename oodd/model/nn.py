from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.optimizers import Adam


class NNModel:
  def __init__(
    self,
    num_features,
    num_classes,
    dense_nodes=[16]
  ):
    self.num_features = num_features
    self.num_classes = num_classes
    self.dense_nodes = dense_nodes

    self.build_model()

  def build_model(self):
    model = Sequential()
    model.add(Input(shape=(self.num_features,)))

    initializer = TruncatedNormal(stddev=0.01)

    for node in self.dense_nodes:
      model.add(Dense(node, activation="linear", kernel_initializer=initializer))
      model.add(Dropout(0.5))

    last_activation = "sigmoid" if self.num_classes == 1 else "softmax"
    model.add(Dense(self.num_classes, activation=last_activation))

    adam = Adam(lr=0.0005)
    model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
    model.summary()

    self.model = model
    self.ood_model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

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
