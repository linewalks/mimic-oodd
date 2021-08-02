from oodd.evaluation.prediction import PredictionEvaluator
from oodd.evaluation.oodd import OODDEvaluator
from oodd.model.nn import NNModel
from oodd.utils.args import get_common_args, parse_common_args
from oodd.utils.config import load_config
from oodd.utils.runner import RunnerBase


def parse_args():
  argparser = get_common_args()
  argparser.add_argument(
    "--time_to_use",
    dest="time_to_use",
    type=int,
    default=24,
    help="When to make prediction. (defualt :24)"
  )
  argparser.add_argument(
    "--time_to_observe",
    dest="time_to_observe",
    type=int,
    default=100,
    help="""Max time to observe the data.
Septic shock after this time will labeled as 0. (default: 100)
    """
  )
  argparser.add_argument(
    "--data_type",
    dest="data_type",
    type=str,
    default="last",
    help="""Data prperocess type. (last, stat)
last:
  use data of last time point only.
stat:
  use aggregated data. (mean, min, max, std)
    """
  )
  return parse_common_args(argparser)


class NNRunner(RunnerBase):
  def __init__(
    self,
    config,
    feature_list,
    scenario_type,
    scenario_param,
    time_to_use,
    time_to_observe,
    data_type,
    random_state=1234
  ):
    super().__init__(
      config,
      feature_list,
      scenario_type,
      scenario_param,
      random_state
    )

    self.time_to_use = time_to_use
    self.time_to_observe = time_to_observe
    self.data_type = data_type

  def close(self):
    self.data_loader.close()

  def run(self):
    x, y, data_key_df, seq_len_list, x_cols = self.data_loader.get_survival_inputs(
      self.time_to_use,
      self.time_to_observe,
      self.data_type
    )
    x = x / x.mean(axis=0)

    train_data, test_data = self.split_func(
      x,
      y,
      data_key_df,
      seq_len_list,
      random_state=self.random_state,
      **self.split_param
    )

    model = NNModel(
      x.shape[-1],
      1
    )
    model.train(
      train_data["x"],
      train_data["y"][:, 0]
    )

    pred_y = model.predict(
      test_data["x"]
    )

    prediction_result = PredictionEvaluator().evaluate(
      test_data["y"][:, 0],
      pred_y
    )
    print("Prediction Result", prediction_result)

    train_ood = model.predict_ood(train_data["x"])
    test_ood = model.predict_ood(test_data["x"])

    oodd_result = OODDEvaluator().evaluate(
      train_ood,
      test_ood,
      test_data["oodd_label"]
    )
    print("OODD Result", oodd_result)


def main():
  args = parse_args()
  config = load_config()

  runner = NNRunner(
    config,
    args.feature_list,
    args.scenario_type,
    args.scenario_param,
    args.time_to_use,
    args.time_to_observe,
    args.data_type,
    args.random_state
  )
  runner.run()
  runner.close()


if __name__ == "__main__":
  main()
