from oodd.evaluation.prediction import RNNPredictionEvaluator
from oodd.evaluation.oodd import RNNOODDEvaluator
from oodd.model.rnn import RNNModel
from oodd.utils.args import get_common_args
from oodd.utils.config import load_config
from oodd.utils.runner import RunnerBase


def parse_args():
  argparser = get_common_args()
  argparser.add_argument(
     "--window_size",
     dest="window_size",
     type=int,
     default=25,
     help="Size of the window to use for the RNN. (default: 25)"
  )
  argparser.add_argument(
    "--sequence_length",
    dest="sequence_length",
    type=int,
    default=50,
    help="Max length of the sequence to use for the RNN. (default: 50)"
  )
  return argparser.parse_args()


class RNNRunner(RunnerBase):
  def __init__(
    self,
    config,
    scenario_type,
    scenario_param,
    window_size,
    sequence_length,
    random_state=1234
  ):
    super().__init__(
      config,
      scenario_type,
      scenario_param,
      random_state
    )

    self.window_size = window_size
    self.sequence_length = sequence_length

    self._define_split_func()

  def close(self):
    self.data_loader.close()

  def run(self):
    x, y, data_key_df, seq_len_list, _ = self.data_loader.get_rnn_inputs(
      window_size=self.window_size,
      sequence_length=self.sequence_length
    )
    train_data, test_data = self.split_func(
      x,
      y,
      data_key_df,
      seq_len_list,
      random_state=self.random_state,
      **self.split_param
    )

    model = RNNModel(
      x.shape[-1],
      y.shape[-1]
    )
    model.train(
      train_data["x"],
      train_data["y"]
    )

    pred_y = model.predict(
      test_data["x"]
    )

    prediction_result = RNNPredictionEvaluator().evaluate(
      test_data["y"],
      pred_y,
      test_data["seq_len_list"]
    )
    print("Prediction Result", prediction_result)

    train_ood = model.predict_ood(train_data["x"])
    test_ood = model.predict_ood(test_data["x"])

    oodd_result = RNNOODDEvaluator().evaluate(
      train_ood,
      test_ood,
      test_data["oodd_label"],
      train_data["seq_len_list"],
      test_data["seq_len_list"]
    )
    print("OODD Result", oodd_result)


def main():
  args = parse_args()
  config = load_config()

  runner = RNNRunner(
    config,
    args.scenario_type,
    args.scenario_param,
    args.window_size,
    args.sequence_length,
    args.random_state
  )
  runner.run()
  runner.close()


if __name__ == "__main__":
  main()
