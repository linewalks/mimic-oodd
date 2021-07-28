import os
import numpy as np
import pandas as pd

from collections import Iterable
from sqlalchemy import create_engine
from tensorflow.keras.preprocessing.sequence import pad_sequences


class MIMIC3:
  def __init__(
    self,
    db_uri: str,
    feature_list: list = [
      "bun",
      "bun_cr_ratio",
      "gcs",
      "pao2",
      "ph_art",
      "resprate",
      "shock_idx",
      "sysbp",
      "urineoutput_6hr"
    ],
    mimic_schema: str = "mimiciii",
    sepsis_schema: str  ="mimiciii_sepsis",
    derived_schema: str = "mimiciii_derived",
    trewscore_feature_schema: str = "trewscore_feat",
    data_save_path: str = "./files"
  ):
    self.engine = create_engine(db_uri)
    self.conn = self.engine.connect()

    self.feature_list = feature_list

    self.mimic_schema = mimic_schema
    self.sepsis_schema = sepsis_schema
    self.derived_schema = derived_schema
    self.trewscore_feature_schema = trewscore_feature_schema

    self.data_save_path = data_save_path
    os.makedirs(self.data_save_path, exist_ok=True)

  def close(self):
    self.conn.close()

  def get_file_path(self, filename: str):
    return os.path.join(self.data_save_path, filename)

  def save_df(self, df: pd.DataFrame, filename: str) -> None:
    df.to_pickle(self.get_file_path(filename))

  def load_df(self, filename: str) -> pd.DataFrame:
    return pd.read_pickle(self.get_file_path(filename))

  def save_npy(self, ary: np.ndarray, filename: str) -> None:
    np.save(self.get_file_path(filename), ary)

  def load_npy(self, filename: str) -> np.ndarray:
    return np.load(self.get_file_path(filename))

  def check_file(self, filename: str) -> bool:
    return os.path.exists(self.get_file_path(filename))

  def _get_merged_filename(self) -> str:
    return "_".join(self.feature_list) + ".pkl"

  def get_merged_data(
    self,
  ) -> pd.DataFrame:
    print("Loading Merged Data")
    filename = self._get_merged_filename()
    if self.check_file(filename):
      return self.load_df(filename)

    key_cols = ["icustay_id", "hr", "endtime"]
    target_df = self.get_target()
    key_df = target_df[key_cols]

    feature_df_list = []
    for feature in self.feature_list:
      if isinstance(feature, str):
        table_name, col_name = feature, None
      else:
        table_name, col_name = feature

      each_df = pd.merge(
        key_df,
        self.get_feature(table_name, col_name),
        on=key_cols,
        how="left"
      ).drop(key_cols, axis=1)
      feature_df_list.append(each_df)

    merged_df = pd.concat([
      target_df,
    ] + feature_df_list, axis=1)

    self.save_df(merged_df, filename)
    return merged_df

  def get_feature(
    self,
    table_name: str,
    col_name: str = None,
  ):
    if col_name is None:
      col_name = table_name
    print("Loading Feature", table_name, col_name)

    df = pd.read_sql(f"""
      SELECT
        icustay_hours.icustay_id,
        hr,
        endtime,
        {col_name}
      FROM
        {self.derived_schema}.icustay_hours AS icustay_hours
      LEFT JOIN
        (
          -- 시간 단위로 묶어서 처리
          -- (1시간내 2회 이상 측정된 노이즈 제거)
          SELECT
            icustay_id,
            DATE_TRUNC('hour', charttime) AS charttime,
            AVG({col_name}) AS {col_name}
          FROM
            {self.trewscore_feature_schema}.{table_name}
          GROUP BY
            icustay_id,
            DATE_TRUNC('hour', charttime)
        ) AS feature
      ON
        icustay_hours.icustay_id = feature.icustay_id AND
        icustay_hours.endtime = feature.charttime
      WHERE
        hr > 0
      ORDER BY
        icustay_hours.icustay_id,
        endtime
    """, self.conn)

    df[col_name] = df[col_name].fillna(method="ffill").fillna(method="bfill")
    print("Loading Feature", table_name, col_name, "Done", df.shape)
    return df

  def get_target(self):
    print("Loading Target")
    df = pd.read_sql(f"""
      SELECT
        icustays.hadm_id,
        icustay_hours.icustay_id,
        hr,
        endtime,
        charttime,
        COALESCE((charttime - endtime) BETWEEN INTERVAL '0 DAY' AND INTERVAL '1 DAY', false) AS target
      FROM
        {self.derived_schema}.icustay_hours AS icustay_hours
      LEFT JOIN
        {self.sepsis_schema}.septic_shock_occurrence_v2 AS septic_shock
      ON
        icustay_hours.icustay_id = septic_shock.icustay_id
      LEFT JOIN
        {self.mimic_schema}.icustays AS icustays
      ON
        icustay_hours.icustay_id = icustays.icustay_id
      WHERE
        hr > 0
      ORDER BY
        icustays.hadm_id,
        endtime
    """, self.conn)

    # Septic Shock이 발생헀으면 (Charttime이 있으면)
    # 그 이후의 데이터는 날린다.
    df = df[
        pd.isnull(df.charttime) |
        (df.charttime > df.endtime)
    ].reset_index(drop=True)

    use_icustay_list = df.drop_duplicates(["hadm_id"], keep="first").icustay_id.unique()
    df = df[df.icustay_id.isin(use_icustay_list)].reset_index(drop=True)
    print("Load Target Done", df.shape)

    return df

  def _convert_merged_to_rnn_input(
    self,
    merged_df: pd.DataFrame,
    window_size: int,
    sequence_length: int
  ):
    key_cols = ["hadm_id", "icustay_id", "hr", "endtime"]
    y_cols = ["target"]
    x_cols = merged_df.columns.drop(key_cols + y_cols + ["charttime", "target"])

    key_list = []
    seq_key_list = []
    x = []
    y = []
    for icustay_id, group_df in merged_df.groupby(["hadm_id", "icustay_id"]):
        group_key_df = group_df[key_cols]
        group_x_ary = group_df[x_cols].values
        group_y_ary = group_df[y_cols].values

        for start_idx in range(0, group_df.shape[0], window_size):
            end_idx = start_idx + sequence_length
            key_list.append(group_key_df[["hadm_id", "icustay_id"]].values[0])
            seq_key_list.append(group_key_df[start_idx:end_idx])
            x.append(group_x_ary[start_idx:end_idx])
            y.append(group_y_ary[start_idx:end_idx])

    data_key_df = pd.DataFrame(key_list, columns=["hadm_id", "icustay_id"])
    seq_len_list = np.array([len(seq_key) for seq_key in seq_key_list])
    x = pad_sequences(x)
    y = pad_sequences(y)

    return x, y, data_key_df, seq_len_list

  def get_rnn_input_filename(self, window_size, sequence_length, postfix):
    feature_list_str = "_".join(map(str, self.feature_list + [window_size, sequence_length]))
    return f"{feature_list_str}_{postfix}"

  def get_rnn_inputs(
    self,
    window_size: int = 25,
    sequence_length: int = 50
  ):
    merged_df = self.get_merged_data()

    filename_dict = {
      "x": self.get_rnn_input_filename(window_size, sequence_length, "x.npy"),
      "y": self.get_rnn_input_filename(window_size, sequence_length, "y.npy"),
      "data_key_df": self.get_rnn_input_filename(window_size, sequence_length, "data_key_df.pkl"),
      "seq_len_list": self.get_rnn_input_filename(window_size, sequence_length, "seq_len_list.npy")
    }

    if all(self.check_file(filename) for filename in filename_dict.values()):
      x = self.load_npy(filename_dict["x"])
      y = self.load_npy(filename_dict["y"])
      data_key_df = self.load_df(filename_dict["data_key_df"])
      seq_len_list = self.load_npy(filename_dict["seq_len_list"])
      return x, y, data_key_df, seq_len_list

    x, y, data_key_df, seq_len_list = self._convert_merged_to_rnn_input(
      merged_df,
      window_size,
      sequence_length
    )

    self.save_npy(x, filename_dict["x"])
    self.save_npy(y, filename_dict["y"])
    self.save_df(data_key_df, filename_dict["data_key_df"])
    self.save_npy(seq_len_list, filename_dict["seq_len_list"])
    return x, y, data_key_df, seq_len_list

  def _split_by_df(
    self,
    x,
    y,
    data_key_df,
    seq_len_list,
    condition_df: pd.DataFrame,
    train_ratio: float,
    random_state: int = 1234
  ):
    np.random.seed(random_state)
    selected_patients = condition_df[condition_df.condition].hadm_id.unique()
    other_patients = condition_df[~condition_df.condition].hadm_id.unique()

    num_train = int(selected_patients.size * train_ratio)
    train_patients = np.random.choice(selected_patients, num_train, replace=False)

    test_selected_patients = set(selected_patients) - set(train_patients)
    test_other_patients = set(other_patients)

    test_patients = test_selected_patients | test_other_patients

    train_idx = data_key_df.hadm_id.isin(train_patients)
    test_idx = data_key_df.hadm_id.isin(test_patients)

    train_x, train_y = x[train_idx], y[train_idx]
    train_data_key_df = data_key_df[train_idx]
    train_seq_key_list = seq_len_list[train_idx]

    test_x, test_y = x[test_idx], y[test_idx]
    test_data_key_df = data_key_df[test_idx]
    test_seq_key_list = seq_len_list[test_idx]

    oodd_label = np.zeros(len(test_y))
    oodd_label[test_data_key_df.hadm_id.isin(test_selected_patients)] = 1

    print("Train Patients", len(train_patients))
    print("Test Patients", len(test_patients))

    print("Train", train_x.shape, train_y.shape)
    print("Test", test_x.shape, test_y.shape)

    print("OODD Label True", oodd_label.sum(), len(oodd_label))
    return {
      "x": train_x,
      "y": train_y,
      "data_key_df": train_data_key_df,
      "seq_len_list": train_seq_key_list,
    }, {
      "x": test_x,
      "y": test_y,
      "data_key_df": test_data_key_df,
      "seq_len_list": test_seq_key_list,
      "oodd_label": oodd_label
    }

  def _split_by_gender(
    self,
    x,
    y,
    data_key_df,
    seq_len_list,
    train_gender: str = "M",
    train_ratio: float = 0.5,
    random_state: int = 1234
  ):
    visit_demo_df = pd.read_sql(f"""
      SELECT
        hadm_id,
        gender = '{train_gender}' AS condition
      FROM
        {self.mimic_schema}.admissions
      LEFT JOIN
        {self.mimic_schema}.patients
      ON
        admissions.subject_id = patients.subject_id
    """, self.conn)
    return self._split_by_df(
      x,
      y,
      data_key_df,
      seq_len_list,
      visit_demo_df,
      train_ratio,
      random_state
    )

  def _split_by_age(
    self,
    x,
    y,
    data_key_df,
    seq_len_list,
    train_age_min=40,
    train_age_max=9999,
    train_ratio: float = 0.5,
    random_state=1234
  ):
    visit_demo_df = pd.read_sql(f"""
      SELECT
        hadm_id,
        (
          AGE(admittime, dob) BETWEEN
          INTERVAL '{train_age_min} years' AND
          INTERVAL '{train_age_max} years'
        ) AS condition
      FROM
        {self.mimic_schema}.admissions
      LEFT JOIN
        {self.mimic_schema}.patients
      ON
        admissions.subject_id = patients.subject_id
    """, self.conn)
    return self._split_by_df(
      x,
      y,
      data_key_df,
      seq_len_list,
      visit_demo_df,
      train_ratio,
      random_state
    )
