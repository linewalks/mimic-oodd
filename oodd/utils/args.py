import argparse


def get_common_args():
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
    "--random_state",
    dest="random_state",
    type=int,
    default=1234,
    help="Random state to use."
  )
  return argparser

