import os
import re

import pandas as pd


def extract_from_files():
    # LEGACY METHOD; optimals / makespans should be extracted from the `opt/lb` files
    res = []

    for path in os.listdir('online'):
        with open(f'online/{path}', 'r') as file:
            # discard first 14 lines
            for _ in range(14):
                file.readline()
            opt = re.search('(?:[0-9]+ *){3}([0-9]*)', file.readline()).groups()[0]
            res.append([f'J30_{path.removeprefix("j30").removesuffix(".sm")}', opt])

    df = pd.DataFrame(res, columns=['ins', 'mk']).astype({'ins': 'string', 'mk': 'Int32'})
    df = df.set_index('ins')
    df.to_csv('j30_opt.csv')


def extract_bounds(folder) -> pd.DataFrame:
    res = []

    for path in os.listdir(folder):
        with open(f'{folder}/{path}', 'r') as file:
            # discard first 26 lines
            for _ in range(26):
                file.readline()
            for line in file.readlines():
                search = re.search('^ *(\d{1,3}) *(\d{1,3}) *(\d{1,3}) *(\d{1,3})', line)

                if search is None:
                    break
                (x, y, ub, lb) = search.groups()

                # file_name | lb | ub | opt
                file_name = f'{path}_{x}_{y}'
                res.append([file_name, lb, ub, lb if lb == ub else pd.NaT])
    df = pd.DataFrame(res, columns=['ins', 'lb', 'ub', 'opt']).set_index('ins')
    return df


if __name__ == '__main__':
    df = extract_bounds('optimals')

    df.to_csv('optimals.csv')
