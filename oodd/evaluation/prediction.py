import numpy as np

from sklearn.metrics import (
  roc_auc_score,
  average_precision_score,
  f1_score
)


class PredictionEvaluator:
  def evaluate(
    self,
    test_y,
    pred_y
  ):
    return {
      "auroc": roc_auc_score(test_y, pred_y),
      "auprc": average_precision_score(test_y, pred_y),
      "f1": f1_score(test_y, pred_y > 0.5)
    }


class RNNPredictionEvaluator(PredictionEvaluator):
  def evaluate(
    self,
    test_y,
    pred_y,
    seq_len_list
  ):
    new_pred_y = np.concatenate([
      each_pred[-seq_len:]
      for seq_len, each_pred in zip(seq_len_list, pred_y)
    ])
    new_test_y = np.concatenate([
      each_test[-seq_len:]
      for seq_len, each_test in zip(seq_len_list, test_y)
    ])

    return super().evaluate(
      new_test_y,
      new_pred_y
    )
