import hashlib
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from tensorflow.keras.preprocessing.sequence import pad_sequences
from typing import Optional


class MIMIC3:
  def __init__(
    self,
    db_uri: str,
    feature_list: list = [
      # "heartrate",
      # "pao2",
      # "ph_art",
      # "platelet",
      # "resprate",
      # "sysbp",
      # "icd_diag|chron_liver",
      # "icd_diag|immunocomp",
      # "icd_diag|hema_malig",
      # "icd_diag|chron_heart",
      # "icd_diag|chron_organ",
      # "icd_diag|diabetes",

      "bun",
      "bun_cr_ratio",
      "fio2",
      "gcs",
      "shock_idx",
      "sofa_score|hepatic",
      "sofa_score|renal",
      "sofa_score|neurologic",
      "urineoutput_6hr",
      "wbc",
      "prescriptions|formulary_drug_cd|5000",
      "demographic",
      "icustays"
    ],
    mimic_schema: str = "mimiciii",
    sepsis_schema: str  ="mimiciii_sepsis",
    derived_schema: str = "mimiciii_derived",
    trewscore_feature_schema: str = "trewscore_feat",
    data_save_path: str = "./files"
  ):
    self.engine = create_engine(db_uri)
    self.conn = self.engine.connect()

    self.feature_list = list(sorted(feature_list))

    self.mimic_schema = mimic_schema
    self.sepsis_schema = sepsis_schema
    self.derived_schema = derived_schema
    self.trewscore_feature_schema = trewscore_feature_schema

    self.data_save_path = data_save_path
    os.makedirs(self.data_save_path, exist_ok=True)

  def close(self):
    self.conn.close()

  def get_file_path(self, filename: str):
    filename, file_ext = filename.split(".")
    enc = hashlib.md5()
    enc.update(filename.encode("utf-8"))
    filename = enc.hexdigest()
    return os.path.join(self.data_save_path, f"{filename}.{file_ext}")

  def save_df(self, df: pd.DataFrame, filename: str) -> None:
    df.to_pickle(self.get_file_path(filename))

  def load_df(self, filename: str) -> pd.DataFrame:
    return pd.read_pickle(self.get_file_path(filename))

  def save_npy(self, ary: np.ndarray, filename: str) -> None:
    np.save(self.get_file_path(filename), ary)

  def load_npy(self, filename: str) -> np.ndarray:
    return np.load(self.get_file_path(filename), allow_pickle=True)

  def check_file(self, filename: str) -> bool:
    return os.path.exists(self.get_file_path(filename))

  def _get_merged_filename(self) -> str:
    return "_".join(self.feature_list) + ".pkl"

  def get_dummies_with_nested_column_names(self, series, parent_col_name):
    df = pd.get_dummies(series)
    df.columns = [(parent_col_name, col) for col in df.columns]
    return df

  def get_merged_data(
    self,
  ) -> pd.DataFrame:
    filename = self._get_merged_filename()
    print("Loading Merged Data", self.get_file_path(filename))
    if self.check_file(filename):
      return self.load_df(filename)

    key_cols = ["icustay_id", "hr", "endtime"]
    target_df = self.get_target()
    key_df = target_df[key_cols]

    feature_df_list = []
    for feature in self.feature_list:
      ary = feature.split("|")
      table_name = ary[0]

      load_func = self.get_charttime_feature
      if table_name == "sofa_score":
        load_func = self.get_hr_feature
      elif table_name == "prescriptions":
        load_func = self.get_prescriptions_feature
      elif table_name == "demographic":
        load_func = self.get_demographic_feature
      elif table_name == "icustays":
        load_func = self.get_icustays_feature
      elif table_name == "icd_diag":
        load_func = self.get_hadm_feature

      each_df = pd.merge(
        key_df,
        load_func(*ary),
        on=key_cols,
        how="left"
      ).drop(key_cols, axis=1)
      feature_df_list.append(each_df)

    merged_df = pd.concat([
      target_df,
    ] + feature_df_list, axis=1)

    self.save_df(merged_df, filename)
    return merged_df

  def get_hadm_feature(
    self,
    table_name: str,
    col_name: Optional[str] = None
  ):
    if col_name is None:
      col_name = table_name
    print("Loading HADM_ID Feature", table_name, col_name)

    df = pd.read_sql(f"""
      SELECT
        icustay_hours.icustay_id,
        icustay_hours.hr,
        endtime,
        {col_name}
      FROM
        {self.derived_schema}.icustay_hours AS icustay_hours
        LEFT JOIN
          {self.mimic_schema}.icustays AS icustays
        ON
          icustay_hours.icustay_id = icustays.icustay_id
        LEFT JOIN
          {self.trewscore_feature_schema}.{table_name} AS feature
        ON
          icustays.hadm_id = feature.hadm_id
      WHERE
        icustay_hours.hr > 0
      ORDER BY
        icustay_hours.icustay_id,
        endtime
    """, self.conn)

    df[col_name] = df[col_name].fillna(method="ffill").fillna(method="bfill")
    print("Loading HADM_ID Feature", table_name, col_name, "Done", df.shape)
    return df

  def get_prescriptions_feature(
    self,
    table_name: str,
    col_name: str,
    min_count: int = 500
  ):
    print("Loading Prescription", table_name, col_name)

    df = pd.read_sql(f"""
      SELECT
        icustay_hours.icustay_id,
        icustay_hours.hr,
        icustay_hours.endtime,
        prescriptions.{col_name},
        prescriptions.cnt
      FROM
        {self.derived_schema}.icustay_hours AS icustay_hours
      LEFT JOIN
        (
          SELECT
            icustay_id,
            startdate,
            {col_name},
            COUNT({col_name}) AS cnt
          FROM
            {self.mimic_schema}.prescriptions prescriptions
          WHERE
            {col_name} IN (
              SELECT
                {col_name}
              FROM
                {self.mimic_schema}.prescriptions
              GROUP BY
                {col_name}
              HAVING
                COUNT(*) > {min_count}
            )
          GROUP BY
            icustay_id,
            startdate,
            {col_name}
        ) prescriptions
      ON
        icustay_hours.icustay_id = prescriptions.icustay_id AND
        DATE_TRUNC('day', icustay_hours.endtime) = prescriptions.startdate
      WHERE
        icustay_hours.hr > 0 AND
        {col_name} IS NOT NULL
      ORDER BY
        icustay_hours.icustay_id,
        endtime
    """, self.conn)

    df = pd.pivot_table(
      data=df[["icustay_id", "hr", "endtime", col_name, "cnt"]],
      index=["icustay_id", "hr", "endtime"],
      columns=[col_name],
      aggfunc="sum",
      fill_value=None
    )

    df = df.clip(0, 1)
    df[df.eq(0)] = None

    df = df.sort_values(["icustay_id", "hr"])
    filled_df = df.groupby("icustay_id").fillna(method="ffill")

    df = pd.concat([
      df.reset_index()[["icustay_id", "hr", "endtime"]],
      filled_df.reset_index(drop=True)
    ], axis=1)

    return df

  def get_demographic_feature(
    self,
    table_name: str
  ):
    print("Loading Demographic")
    df = pd.read_sql(f"""
      SELECT
        icustays.subject_id,
        icustay_hours.icustay_id,
        icustay_hours.hr,
        icustay_hours.endtime
      FROM
        {self.derived_schema}.icustay_hours AS icustay_hours
        JOIN
          {self.mimic_schema}.icustays AS icustays
        ON
          icustays.icustay_id = icustay_hours.icustay_id
      WHERE
        icustay_hours.hr > 0
      ORDER BY
        icustay_hours.icustay_id,
        endtime
    """, self.conn)

    demo_df = pd.read_sql(f"""
      SELECT
        *
      FROm
        {self.mimic_schema}.patients
    """, self.conn)

    age_df = pd.merge(
      df[["subject_id", "endtime"]],
      demo_df[["subject_id", "dob"]],
      on="subject_id",
      how="left",
      validate="many_to_one"
    )

    age_df["agey"] = ((age_df.endtime.subtract(age_df.dob)) / 365).dt.days
    age_df["agem"] = ((age_df.endtime.subtract(age_df.dob)) / 30).dt.days
    age_df.agem.loc[age_df.agem >= (300 * 12)] -= 211 * 12
    age_df.agey.loc[age_df.agey >= 300] -= 211

    gender_df = pd.merge(
      df[["subject_id"]],
      demo_df[["subject_id", "gender"]],
      how="left",
      on="subject_id",
      validate="many_to_one"
    )
    gender_df = self.get_dummies_with_nested_column_names(gender_df.gender, "gender")

    return pd.concat([
      df[["icustay_id", "hr", "endtime"]],
      age_df[["agey", "agem"]],
      gender_df
    ], axis=1)

  def get_icustays_feature(
    self,
    table_name: str
  ):
    print("Loading ICUStays")
    df = pd.read_sql(f"""
      SELECT
        icustays.subject_id,
        icustay_hours.icustay_id,
        icustay_hours.hr,
        icustay_hours.endtime,
        icustays.first_careunit
      FROM
        {self.derived_schema}.icustay_hours AS icustay_hours
        JOIN
          {self.mimic_schema}.icustays AS icustays
        ON
          icustays.icustay_id = icustay_hours.icustay_id
      WHERE
        icustay_hours.hr > 0
      ORDER BY
        icustay_hours.icustay_id,
        endtime
    """, self.conn)

    return pd.concat([
      df[["icustay_id", "hr", "endtime"]],
      self.get_dummies_with_nested_column_names(df.first_careunit, "careunit")
    ], axis=1)

  def get_hr_feature(
    self,
    table_name: str,
    col_name: Optional[str] = None
  ):
    if col_name is None:
      col_name = table_name
    print("Loading HR Feature", table_name, col_name)

    df = pd.read_sql(f"""
      SELECT
        icustay_hours.icustay_id,
        icustay_hours.hr,
        endtime,
        {col_name}
      FROM
        {self.derived_schema}.icustay_hours AS icustay_hours
      LEFT JOIN
        (
          -- 한 icustay_id, hr에 중복된 값을 가지는 경우가 있음 
          -- 평균으로 묶어서 처리
          SELECT
            icustay_id,
            hr,
            AVG({col_name}) AS {col_name}
          FROM
            {self.trewscore_feature_schema}.{table_name}
          GROUP BY
            icustay_id,
            hr
        ) AS feature
      ON
        icustay_hours.icustay_id = feature.icustay_id
        AND icustay_hours.hr = feature.hr
      WHERE
        icustay_hours.hr > 0
      ORDER BY
        icustay_hours.icustay_id,
        endtime
    """, self.conn)

    df[col_name] = df[col_name].fillna(method="ffill").fillna(method="bfill")
    print("Loading HR Feature", table_name, col_name, "Done", df.shape)
    return df

  def get_charttime_feature(
    self,
    table_name: str,
    col_name: Optional[str] = None
  ):
    if col_name is None:
      col_name = table_name
    print("Loading Charttime Feature", table_name, col_name)

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
    print("Loading Charttime Feature", table_name, col_name, "Done", df.shape)
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
    print("Converting to RNN Input", window_size, sequence_length)
    key_cols = ["hadm_id", "icustay_id", "hr", "endtime"]
    y_cols = ["target"]
    x_cols = merged_df.columns.drop(key_cols + y_cols + ["charttime", "target"])

    key_list = []
    seq_key_list = []
    x = []
    y = []
    for _, group_df in merged_df.groupby(["hadm_id", "icustay_id"]):
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

    return x, y, data_key_df, seq_len_list, list(x_cols)

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
      "seq_len_list": self.get_rnn_input_filename(window_size, sequence_length, "seq_len_list.npy"),
      "x_cols": self.get_rnn_input_filename(window_size, sequence_length, "x_cols.npy")
    }

    if all(self.check_file(filename) for filename in filename_dict.values()):
      x = self.load_npy(filename_dict["x"])
      y = self.load_npy(filename_dict["y"])
      data_key_df = self.load_df(filename_dict["data_key_df"])
      seq_len_list = self.load_npy(filename_dict["seq_len_list"])
      x_cols = self.load_npy(filename_dict["x_cols"])
      return x, y, data_key_df, seq_len_list, x_cols

    x, y, data_key_df, seq_len_list, x_cols = self._convert_merged_to_rnn_input(
      merged_df,
      window_size,
      sequence_length
    )

    self.save_npy(x, filename_dict["x"])
    self.save_npy(y, filename_dict["y"])
    self.save_df(data_key_df, filename_dict["data_key_df"])
    self.save_npy(seq_len_list, filename_dict["seq_len_list"])
    self.save_npy(x_cols, filename_dict["x_cols"])
    return x, y, data_key_df, seq_len_list, x_cols

  def get_survival_input_filename(self, time_to_use, time_to_observe, data_type, postfix):
    feature_list_str = "_".join(map(str, self.feature_list + [
      time_to_use,
      time_to_observe,
      data_type
    ]))
    return f"{feature_list_str}_{postfix}"

  def _convert_merged_to_survival_input(
    self,
    merged_df: pd.DataFrame,
    time_to_use: int,
    time_to_observe: int,
    data_type: str
  ):
    # 방문별 입원시간 추출
    max_hr_df = merged_df.groupby("hadm_id").hr.max().reset_index()

    # 사용 기준 시간 후 데이터 날림
    merged_df = merged_df[merged_df.hr <= time_to_use]

    # Septic Shock 발생 시간 확인
    min_time_df = merged_df.groupby("hadm_id").endtime.min().reset_index()
    min_time_df = pd.merge(
      min_time_df,
      merged_df.groupby("hadm_id").charttime.min().reset_index(),
      how="left",
      on="hadm_id"
    )

    min_time_df.loc[
      pd.notnull(min_time_df.charttime),
      "timediff"
    ] = pd.to_datetime(min_time_df.charttime) - pd.to_datetime(min_time_df.endtime)
    min_time_df.loc[
      pd.isnull(min_time_df.charttime),
      "timediff"
    ] = None
    min_time_df.timediff = min_time_df.timediff / np.timedelta64(1, "h")

    # 기준 시간 이전에 Septic Shock이 발생한 환자는 버림
    print(f"Septic shock occured before {time_to_use} hours count", (min_time_df.timediff <= time_to_use).sum())
    min_time_df = min_time_df[~(min_time_df.timediff < time_to_use)]

    min_time_df = pd.merge(
      min_time_df,
      max_hr_df,
      how="left",
      on="hadm_id"
    )

    # 기준 시간 이전에 퇴원한 환자는 버림
    print(f"Discharge before {time_to_use} hours count", (min_time_df.hr < time_to_use).sum())
    min_time_df = min_time_df[~(min_time_df.hr < time_to_use)]

    # Septic Shock 발생
    min_time_df.loc[pd.notnull(min_time_df.charttime), "target"] = 1
    min_time_df.loc[pd.notnull(min_time_df.charttime), "target_hours"] = min_time_df.timediff
    # Septic Shock 미발생
    min_time_df.loc[pd.isnull(min_time_df.charttime), "target"] = 0
    min_time_df.loc[pd.isnull(min_time_df.charttime), "target_hours"] = min_time_df.hr

    target_df = min_time_df[["hadm_id", "target", "target_hours"]].reset_index(drop=True)

    # time_to_observe 값이 있으면, clipping
    if time_to_observe is not None and time_to_observe > 0:
      target_df.loc[
        (target_df.target == 1) & (target_df.target_hours >= time_to_observe),
        "target"
      ] = 0
      target_df.target_hours = target_df.target_hours.clip(0, time_to_observe)

    merged_df = pd.merge(
      target_df,
      merged_df.drop("target", axis=1),
      on=["hadm_id"],
      how="left"
    ).fillna(0.0)

    cols_to_drop_from_merged = [
      "target",
      "target_hours",
      "icustay_id",
      "hr",
      "endtime",
      "charttime"
    ]
    if data_type == "last":
      merged_df = merged_df[merged_df.hr == time_to_use]
      x = pd.merge(
        target_df[["hadm_id"]],
        merged_df.drop(
          cols_to_drop_from_merged,
          axis=1
        ),
        on="hadm_id",
        how="left"
      ).drop("hadm_id", axis=1)
      x_cols = x.columns.tolist()
    elif data_type == "stat":
      group_df = merged_df.drop(
        cols_to_drop_from_merged,
        axis=1
      ).groupby("hadm_id").agg([
        "mean",
        "std",
        "min",
        "max"
      ])

      x = pd.merge(
        target_df[["hadm_id"]],
        group_df,
        on="hadm_id",
        how="left"
      ).drop("hadm_id", axis=1)
      x_cols = x.columns.tolist()
    elif data_type == "rnn":
      x_cols = merged_df.columns.drop(["hadm_id"] + cols_to_drop_from_merged)
      x = np.array([group_df.values for _, group_df in merged_df.drop(
        cols_to_drop_from_merged,
        axis=1
      ).groupby(
        "hadm_id"
      )])

    if isinstance(x, pd.DataFrame):
      x = x.values
    y = target_df.drop("hadm_id", axis=1).values
    data_key_df = target_df[["hadm_id"]]
    seq_len_list = np.ones(len(data_key_df))
    return x, y, data_key_df, seq_len_list, x_cols

  def get_survival_inputs(
    self,
    time_to_use: int = 24,   # 입원 후, 기준으로 사용할 시간
    time_to_observe: Optional[int] = 100, # 관찰 기간. 입력 시간 이후의 Septic Shock 발생은 추적하지 않음
    data_type: str = "last"     # last or stat, 마지막 값 사용하거나, 통계값 사용
  ):
    args = [time_to_use, time_to_observe, data_type]
    filename_dict = {
      "x": self.get_survival_input_filename(*args, postfix="x.npy"),
      "y": self.get_survival_input_filename(*args, postfix="y.npy"),
      "data_key_df": self.get_survival_input_filename(*args, postfix="data_key_df.pkl"),
      "seq_len_list": self.get_survival_input_filename(*args, postfix="seq_len_list.npy"),
      "x_cols": self.get_survival_input_filename(*args, postfix="x_cols.npy")
    }

    print("Checking File", self.get_file_path(filename_dict["x"]))

    if all(self.check_file(filename) for filename in filename_dict.values()):
      x = self.load_npy(filename_dict["x"])
      y = self.load_npy(filename_dict["y"])
      data_key_df = self.load_df(filename_dict["data_key_df"])
      seq_len_list = self.load_npy(filename_dict["seq_len_list"])
      x_cols = self.load_npy(filename_dict["x_cols"])
      return x, y, data_key_df, seq_len_list, x_cols

    merged_df = self.get_merged_data()
    x, y, data_key_df, seq_len_list, x_cols = self._convert_merged_to_survival_input(
      merged_df,
      time_to_use,
      time_to_observe,
      data_type
    )

    self.save_npy(x, filename_dict["x"])
    self.save_npy(y, filename_dict["y"])
    self.save_df(data_key_df, filename_dict["data_key_df"])
    self.save_npy(seq_len_list, filename_dict["seq_len_list"])
    self.save_npy(x_cols, filename_dict["x_cols"])
    return x, y, data_key_df, seq_len_list, x_cols

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

  def _split_by_random(
    self,
    x,
    y,
    data_key_df,
    seq_len_list,
    test_ratio: float = 0.2,
    random_state: int = 1234
  ):
    hadm_list = data_key_df.hadm_id.unique()
    train_patients, test_patients = train_test_split(hadm_list, test_size=test_ratio, random_state=random_state)

    train_idx = data_key_df.hadm_id.isin(train_patients)
    test_idx = data_key_df.hadm_id.isin(test_patients)

    train_x, train_y = x[train_idx], y[train_idx]
    train_data_key_df = data_key_df[train_idx]
    train_seq_key_list = seq_len_list[train_idx]

    test_x, test_y = x[test_idx], y[test_idx]
    test_data_key_df = data_key_df[test_idx]
    test_seq_key_list = seq_len_list[test_idx]

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
      # OODD has no meaning for random spliting
      "oodd_label": np.random.randint(0, 2, len(test_y))
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
