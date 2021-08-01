from oodd.data.mimic import MIMIC3


class RunnerBase:
  def __init__(
    self,
    config,
    feature_list,
    scenario_type,
    scenario_param,
    random_state
  ):
    data_loader_param = {}
    if feature_list:
      data_loader_param["feature_list"] = feature_list

    self.data_loader = MIMIC3(
      config["DB_URI"],
      **data_loader_param
    )
    self.scenario_type = scenario_type
    self.scenario_param = scenario_param
    self.random_state = random_state

    self._define_split_func()

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
