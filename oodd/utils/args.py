import argparse


def get_common_args():
  argparser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawTextHelpFormatter
  )
  argparser.add_argument(
    "--feature_list",
    dest="feature_list",
    type=str,
    default=None,
    help="""Comma separated string of features to use.
Available features are
  bun
  bun_cr_ratio
  fio2
  gcs
  heartrate
  pao2
  ph_art
  platelet
  resprate
  shock_idx
  sofa_score|hepatic
  sofa_score|renal
  sofa_score|neurologic
  sysbp
  urineoutput_6hr
  wbc,
  prescriptions|drug|10000
    {table_name}|{column_name}|{min_count}

    min_count: drugs used less than min_count times in the whole data will be ignored.
    """
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
    "--random_state",
    dest="random_state",
    type=int,
    default=1234,
    help="Random state to use."
  )
  return argparser


def parse_common_args(argparser):
  args = argparser.parse_args()
  if args.feature_list:
    args.feature_list = args.feature_list.split(",")
  return args
