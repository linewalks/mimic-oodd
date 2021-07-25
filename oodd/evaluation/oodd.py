import numpy as np

from scipy.spatial import distance
from sklearn.metrics import roc_auc_score


class OODDEvaluator:
  def evaluate(
    self,
    train_ood,
    test_ood,
    oodd_label
  ):
    train_mean = train_ood.mean(axis=0)

    cov = np.cov(train_ood.T)
    vi = np.linalg.inv(cov)
    test_dist = np.array([distance.mahalanobis(train_mean, each, vi) for each in test_ood])

    return {
      "auroc": roc_auc_score(oodd_label, test_dist)
    }


class RNNOODDEvaluator(OODDEvaluator):
  def evaluate(
    self,
    train_ood,
    test_ood,
    oodd_label,
    train_seq_len_list,
    test_seq_len_list,
  ):
    new_train_ood = np.concatenate([
      each_ood[-seq_len:]
      for seq_len, each_ood in zip(train_seq_len_list, train_ood)
    ])
    new_test_ood = np.concatenate([
      each_ood[-seq_len:]
      for seq_len, each_ood in zip(test_seq_len_list, test_ood)
    ])
    print(oodd_label.shape)
    new_oodd_label = np.concatenate([
      [each_target] * seq_len
      for seq_len, each_target in zip(test_seq_len_list, oodd_label)
    ])
    print(new_oodd_label.shape)
    return super().evaluate(
      new_train_ood,
      new_test_ood,
      new_oodd_label
    )
