import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import utils as ut

pd.set_option('display.precision', 2)

sns.set_theme(style='ticks')


def to_latex(df: pd.DataFrame, filename: str, *, caption='', label='', cols=''):
    print(df)

    typed_df = df.astype({
        'dev': 'float',
        'ru': 'float',
        'dc': 'int',
        't': 'float',
        'pre': 'int',
        'count': 'int'
    })

    latex = typed_df.to_latex(
        caption=caption,
        label=label,
        column_format=cols,
        float_format='{:.2f}'.format
    )

    latex = (latex.replace('table', 'tabe')
             .replace('\\begin{tabular}', '\\begin{tabi}[.45]')
             .replace('tabular', 'tabi')
             .replace('cline', 'cmidrule')
             .replace('[t]', '[c]')
             .replace('default', 'VS/SG')
             .replace('grdlw', 'GRD/dEST')
             .replace('vslw', 'VS/dEST')
             .replace('all', 'All')
            )

    latex = re.sub(r'(.*J120(?:.*\n){4})((?:.*\n){12})', r'\2\1', latex, re.M)
    latex = re.sub(r' *\\cmidrule\{.*}\n(.*\\bottomrule)', r'\1', latex, re.M)

    indented_latex = ""
    indent_level = 4
    for line in latex.split('\n'):
        if '\\end' in line:
            indent_level -= 4

        indented_latex += indent_level * ' ' + line + '\n'

        if '\\begin' in line:
            indent_level += 4

    with open(f'tex/{filename}.tex', 'w') as f:
        f.write(indented_latex)


def get_metric_means():
    # discrepancies in %dev may be due to differences in optimal instances found
    dfs = [('opt', ut.read_csv('csvs/opt.csv')), ('sat', ut.read_csv('csvs/sat.csv'))]

    for name, df in dfs:
        df['count'] = (df['out'] == name.upper()).astype(int)
        df['imp'] = (df['dev'] > 0).astype(int)

        # uncomment to keep only instances for which every strategy found the optimal
        # df = df.keep_matching()

        # uncomment to use either lower or upper bounds if optimal non-existent
        df.include_bounds('lb')

        by_groups = df.add_groups().as_s()

        per_dataset = by_groups[['group', 'strt', 'dev', 'ru', 'imp', 'dc', 't', 'pre', 'count']].groupby(['group', 'strt'])
        to_latex(per_dataset.agg({
            'dev': 'mean', 'ru': 'mean', 'imp': 'sum', 'dc': 'mean', 't': 'mean', 'pre': 'mean', 'count': 'sum'
        }), f'{name}_means', cols='cc llllll')


def hist_plot_preamble(df: ut.MyDataframe, metric):
    df = df.rename_strt()

    sns.histplot(
        df,
        linewidth=.5,
        x=metric,
        hue='Method',
        bins=30,
        log_scale=True,
        multiple='stack',
    )


def plot_opt_progress(metric):
    df = ut.read_csv(f'csvs/opt.csv').as_s()
    print(df)
    hist_plot_preamble(df, metric)
    plt.xlabel('Seconds spent in solver (log10)')
    plt.ylabel('Optimal solutions found')
    plt.savefig('../../../../../../codrin/RP_documents/figures/opt_runtime_t.png', dpi=600)
    plt.show()


def plot_sat_progress(metric):
    # keep only the first satisfiable assignments found
    df = ut.read_csv('csvs/raw.csv')[['strt', 'ins', metric]]
    if metric == 's':
        df.as_s()
    sorted = df.sort_values(['strt', 'ins', metric])
    no_dup = sorted.drop_duplicates(['strt', 'ins'])
    no_na = no_dup.dropna()

    hist_plot_preamble(no_na, metric)

    plt.xlabel('Seconds spent in solver (log10)')
    plt.ylabel('First satisfiable assignment found')
    plt.savefig('../../../../../../codrin/RP_documents/figures/sat_runtime_t.png', dpi=600)
    plt.show()


def plot_opt_separately():
    df = ut.read_csv(f'csvs/opt.csv').as_s().add_groups().rename_strt()

    gb = df.groupby('group')
    for group in [gb.get_group(x) for x in gb.groups]:
        sns.histplot(group, linewidth=.5, x='dc', hue='Method', bins=30, log_scale=True, multiple='stack')
        plt.show()


def plot_sat_separately():
    df = ut.read_csv('csvs/raw.csv')[['strt', 'ins', 'dc']].add_groups()
    sorted = df.sort_values(['strt', 'ins', 'dc'])
    no_dup = sorted.drop_duplicates(['strt', 'ins'])
    no_na = no_dup.dropna()

    gb = no_na.rename_strt().groupby('group')
    for group in [gb.get_group(x) for x in gb.groups]:
        sns.histplot(group, linewidth=.5, x='dc', hue='Method', bins=30, log_scale=True, multiple='stack')
        plt.show()


def nodes_per_second():
    df = ut.read_csv('csvs/raw.csv')[['strt', 't', 'dc']].as_s().dropna()
    df['rate'] = df['dc'] / df['t']
    group = df.groupby('strt')
    print(group['rate'].mean())


if __name__ == '__main__':
    get_metric_means()
