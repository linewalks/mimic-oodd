import argparse
import json


def parse_args():
  argparser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawTextHelpFormatter
  )
  argparser.add_argument(
    "--scenario_type",
    dest="scenario_type",
    type=str,
    default="gender",
    help="Type of the OODD Scenario. (gender, age)"
  )
  argparser.add_argument(
    "--scenario_param",
    dest="scenario_param",
    type=str,
    default=None,
    help="""Parameters of the Scenario.
Differs by scenario_type.    

gender: single character (M or F)
age: comma splited string {min_age},{max_age} (ex: 20,30)
    """
  )
  argparser.add_argument(
    "--model",
    dest="model",
    type=str,
    default="rnn",
    help="Model to use. (rnn)"
  )
  argparser.add_argument(
    "--random_state",
    dest="random_state",
    type=int,
    default=1234,
    help="Random state to use."
  )
  return argparser.parse_args()


class Runner:
  def __init__(
    self,
    config,
    scenario_type,
    scenario_param,
    model,
    random_state=1234
  ):
    from oodd.data.mimic import MIMIC3
    self.data_loader = MIMIC3(config["DB_URI"])

    self.scenario_type = scenario_type
    self.scenario_param = scenario_param
    self.model_type = model
    self.random_state = random_state

    self._define_func()

  def close(self):
    self.data_loader.close()

  def _define_func(self):
    self._define_load_func()
    self._define_split_func()
    self._define_model()
    self._define_evaluator()

  def _define_load_func(self):
    if self.model_type == "rnn":
      self.load_func = self.data_loader.get_rnn_inputs

  def _define_split_func(self):
    if self.scenario_type == "gender":
      self.split_func = self.data_loader._split_by_gender
      self.split_param = {}
      if self.scenario_param is not None:
        self.split_param.update({
          "train_gender": self.scenario_param
        })
    elif self.scenario_type == "age":
      self.split_func = self.data_loader._split_by_age
      self.split_param = {}
      if self.scenario_param is not None:
        parma_ary = self.scenario_param.split(",")
        self.split_param.update({
          "train_age_min": int(parma_ary[0]),
          "train_age_max": int(parma_ary[1])
        })

  def _define_model(self):
    if self.model_type == "rnn":
      from oodd.model.rnn import RNNModel
      self.model_cls = RNNModel

  def _define_evaluator(self):
    if self.model_type == "rnn":
      from oodd.evaluation.prediction import RNNPredictionEvaluator
      from oodd.evaluation.oodd import RNNOODDEvaluator
      self.prediction_evaluator = RNNPredictionEvaluator()
      self.oodd_evaluator = RNNOODDEvaluator()

  def run(self):
    x, y, data_key_df, seq_len_list = self.load_func()
    train_data, test_data = self.split_func(
      x,
      y,
      data_key_df,
      seq_len_list,
      random_state=self.random_state,
      **self.split_param
    )

    model = self.model_cls(
      x.shape[-1],
      y.shape[-1]
    )
    model.train(
      train_data["x"],
      train_data["y"],
      epochs=1
    )

    pred_y = model.predict(
      test_data["x"]
    )

    prediction_result = self.prediction_evaluator.evaluate(
      test_data["y"],
      pred_y,
      test_data["seq_len_list"]
    )
    print("Prediction Result", prediction_result)

    train_ood = model.predict_ood(train_data["x"])
    test_ood = model.predict_ood(test_data["x"])

    oodd_result = self.oodd_evaluator.evaluate(
      train_ood,
      test_ood,
      test_data["oodd_label"],
      train_data["seq_len_list"],
      test_data["seq_len_list"]
    )
    print("OODD Result", oodd_result)


def main():
  args = parse_args()

  with open("config.json", "r") as f:
    config = json.load(f)

  runner = Runner(
    config,
    args.scenario_type,
    args.scenario_param,
    args.model,
    args.random_state
  )
  runner.run()
  runner.close()


if __name__ == "__main__":
  main()
