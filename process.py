import os
import re
from typing import Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import int64, float64
from scipy.stats import ranksums

pd.set_option('display.precision', 2)

# Due to oversight on the part of the author, 19 instances were run using a separate "compute" partition of the
# DelftBlue supercomputer. The instances were re-run on the correct partition, and only those results were considered
# for the respective instances.
BAD_RUNS = {
    'J30_47_1.dzn',
    'J30_7_1.dzn',
    'J30_7_10.dzn',
    'J30_7_2.dzn',
    'J30_7_3.dzn',
    'J30_7_4.dzn',
    'J30_7_5.dzn',
    'J30_7_6.dzn',
    'J30_7_7.dzn',
    'J30_7_8.dzn',
    'J30_7_9.dzn',
    'J30_8_1.dzn',
    'J30_8_10.dzn',
    'J30_8_2.dzn',
    'J30_8_3.dzn',
    'J30_8_4.dzn',
    'J30_8_5.dzn',
    'J30_8_6.dzn',
    'J30_8_7.dzn',
}


def parse_to_df(result_folder):
    groups = re.findall("runs_([0-9]+)-([0-9]+)", " ".join(x for x in os.listdir(result_folder)))

    # instance, outcome, strategy, timeout, time, makespan, resource_util, decisions
    parsed_results = []

    # hacky way of changing dimension of list
    for (s, e) in groups:
        working_dir = f"{result_folder}/runs_{s}-{e}/"

        for path in [working_dir + f"run_{str(i).zfill(len(s))}/run.log" for i in range(int(s), int(e) + 1)]:
            with open(path, 'r') as file:
                logger_metrics = []
                custom_metrics = []

                (instance, strategy, timeout) = re.search(".*(J\d{2,3}.*)\" with strategy (.*) with timeout (.*)",
                                                          file.readline()).groups()
                # if part of the faulty runs, do not consider the first occurrence
                if instance in BAD_RUNS:
                    BAD_RUNS.remove(instance)
                    continue

                instance = instance.removesuffix('.dzn')

                while True:
                    line = file.readline()

                    if '%%%' in line:
                        mak = re.search('%%%mzn-stat: objective=(\d*)', line).groups()[0]
                        dc = re.search('%%%mzn-stat: numberOfDecisions=(\d*)', file.readline()).groups()[0]
                        for _ in range(5):
                            file.readline()
                        t = re.search('%%%mzn-stat: timeSpentInSolverInMilliseconds=(\d*)', file.readline()).groups()[0]
                        for _ in range(2):
                            file.readline()

                        logger_metrics.append([instance, strategy, timeout, t, mak, dc])

                    # see if solution was found
                    else:
                        if "Unknown" in line:
                            outcome = "UNS"
                            parsed_results.append([instance, strategy, timeout, -1, -1, -1, "UNS", -1, -1])
                        elif "satisfiable" in line:
                            outcome = "SAT"
                        else:
                            outcome = "OPT"
                        break

                for line in file.readlines():
                    (ru, pre) = re.search('RU\((.*)\), PRE\((.*)\)', line).groups()
                    custom_metrics.append([outcome, ru, pre])

                for l, c in zip(logger_metrics, custom_metrics):
                    parsed_results.append(l + c)

    df = pd.DataFrame(
        parsed_results,
        columns=["ins", "strt", "to", "t", "mk", "dc", "out", "ru", "pre"],
    )

    df = df.astype({
        "ins": str,
        "out": str,
        "strt": str,
        "to": float64,
        "t": float64,
        "mk": int64,
        "ru": float64,
        "dc": int64,
        "pre": int64,
    })

    df = df.replace(-1, pd.NaT)

    best = pd.read_csv('csvs/optimals.csv').set_index('ins')
    df = df.set_index('ins')

    df = df.join(best)
    df['dev'] = (df['opt'] - df['mk']) / df['opt'] * 100

    return df


def parse_to_df_old(result_folder):
    (s, e) = re.search("runs_([0-9]+)-([0-9]+)", " ".join(x for x in os.listdir(result_folder))).groups()

    working_dir = f"{result_folder}/runs_{s}-{e}/"

    # instance, outcome, strategy, timeout, time, makespan, resource_util, decisions
    parsed_results = []

    for path in [working_dir + f"run_{str(i).zfill(len(s))}/run.log" for i in range(int(s), int(e) + 1)]:
        with open(path, 'r') as file:
            (instance, strategy, timeout) = re.search(".*instances/j.{0,3}/(.*)\" with strategy (.*) with timeout (.*)",
                                                      file.readline()).groups()
            instance = instance.removesuffix('.dzn')

            result = file.readline()

            # see if solution was found
            if "Unknown" in result:
                outcome = "UNS"
                parsed_results.append([instance, outcome, strategy, timeout, -1, -1, -1, -1, 0])
            elif "satisfiable" in result:
                outcome = "SAT"
            else:
                outcome = "OPT"

            for line in file.readlines():
                (time, mak, ru, dc, pre) = re.search('T\((.*)\), M\((.*)\), RU\((.*)\), DC\((.*)\), PRE\((.*)\)',
                                                     line).groups()
                parsed_results.append(
                    [instance, outcome, strategy, float(timeout) / 1000, float(time) / 1000, mak, ru, dc, pre])

    df = pd.DataFrame(
        parsed_results,
        columns=["ins", "out", "strt", "to", "t", "mk", "ru", "dc", "pre"],
    )

    df = df.astype({
        "ins": str,
        "out": str,
        "strt": str,
        "to": float64,
        "t": float64,
        "mk": int64,
        "ru": float64,
        "dc": int64,
        "pre": int64,
    })

    df = df.replace(-1, pd.NaT)

    best = pd.read_csv('csvs/optimals.csv').set_index('ins')
    df = df.set_index('ins')

    df = df.join(best)
    df['dev'] = abs(df['mk'] - df['opt']) / df['opt'] * 100

    return df


def aggregate(df: pd.DataFrame):
    return df[df.groupby(['ins', 'strt'])['mk'].transform('min') == df['mk']].astype({
        'ru': float64,
        'dev': float64,
    })


def separate_sat_opt(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return df[df['out'] == 'OPT'], df[df['out'] != 'OPT']


if __name__ == '__main__':
    reruns = parse_to_df('db_runs/db_reruns') # !! contains the faulty runs; MUST be processed first !!
    vslw = parse_to_df('db_runs/db_vslw')
    defaults = parse_to_df('db_runs/db_default')
    grdlw = parse_to_df('db_runs/db_grdlw')
    leftovers = parse_to_df('db_runs/db_leftover')

    df = pd.concat([vslw, defaults, grdlw, leftovers, reruns])

    # sort in order
    df['strt'] = pd.Categorical(df['strt'], categories=['default', 'grdlw', 'vslw'], ordered=True)
    df = df.sort_values(['strt', 'ins', 'mk'], ascending=[True, True, False])

    df.to_csv('csvs/raw.csv')
    agg = aggregate(df)

    agg.to_csv('csvs/aggregates.csv')

    opt, sat = separate_sat_opt(agg)

    opt.to_csv('csvs/opt.csv')
    sat.to_csv('csvs/sat.csv')
