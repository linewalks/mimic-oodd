import os
import pandas as pd

from collections import Iterable
from sqlalchemy import create_engine


class MIMIC3:
  def __init__(
    self,
    db_uri: str,
    mimic_schema: str = "mimiciii",
    sepsis_schema: str  ="mimiciii_sepsis",
    derived_schema: str = "mimiciii_derived",
    trewscore_feature_schema: str = "trewscore_feat",
    data_save_path: str = "./files"
  ):
    self.engine = create_engine(db_uri)
    self.conn = self.engine.connect()

    self.mimic_schema = mimic_schema
    self.sepsis_schema = sepsis_schema
    self.derived_schema = derived_schema
    self.trewscore_feature_schema = trewscore_feature_schema

    self.data_save_path = data_save_path
    os.makedirs(self.data_save_path, exist_ok=True)

  def save_df(self, df: pd.DataFrame, filename: str) -> None:

    file_path = os.path.join(self.data_save_path, filename)
    df.to_pickle(file_path)

  def load_df(self, filename: str) -> pd.DataFrame:
    file_path = os.path.join(self.data_save_path, filename)
    return pd.read_pickle(file_path)

  def check_df(self, filename: str) -> bool:
    file_path = os.path.join(self.data_save_path, filename)
    return os.path.exists(file_path)

  def _get_merged_filename(self, feature_list: list) -> str:
    return "_".join(feature_list) + ".pkl"

  def get_merged_data(
    self,
    feature_list: list = [
      "bun",
      "bun_cr_ratio",
      "gcs",
      "pao2",
      "ph_art",
      "resprate",
      "sysbp",
      "urineoutput_6hr"
    ]
  ) -> pd.DataFrame:
    print("Loading Merged Data")
    filename = self._get_merged_filename(feature_list)
    if self.check_df(filename):
      return self.load_df(filename)

    key_cols = ["icustay_id", "hr", "endtime"]
    target_df = self.get_target()
    key_df = target_df[key_cols]

    feature_df_list = []
    for feature in feature_list:
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
        {self.sepsis_schema}.septic_shock_occurrence AS septic_shock
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
