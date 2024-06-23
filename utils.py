# Some utility functions for manipulating dataframes containing results
import pandas as pd


class MyDataframe(pd.DataFrame):
    """
    All methods implement chaining to make working with them more palatable
    """
    @property
    def _constructor(self):
        return MyDataframe

    def update_dev(self) -> 'MyDataframe':
        # Update the dev metric, used if e.g. bounds are included
        self['dev'] = (self['opt'] - self['mk']) / self['opt'] * 100
        return self

    def add_groups(self) -> 'MyDataframe':
        # Add a new column for the datasets themselves
        prefix_dataset_map = {'1': 'J120', '3': 'J30', '6': 'J60', '9': 'J90'}
        self['group'] = self['ins'].map(lambda x: prefix_dataset_map[x[1]])
        # also include a group with all datasets
        all_group = self.copy()
        all_group['group'] = 'all'
        return pd.concat([self, all_group])
    
    def include_bounds(self, bound) -> 'MyDataframe':
        # Take the mean of the bounds and make it the new optimal
        self['opt'] = self[bound]
        self.update_dev()
        return self
    
    def keep_matching(self) -> 'MyDataframe':
        # Keep only cases where all heuristics solved the same instances
        self = self[self.groupby(['ins', 'out'])[['ins', 'out']].transform('size') > 2]
        return self

    def as_s(self) -> 'MyDataframe':
        # Display time as seconds instead of ms
        self['t'] = self['t'].apply(lambda x: x / 1000)
        return self

    def rename_strt(self) -> 'MyDataframe':
        # Rename the 'strt' column and the methods; used for figures
        rename_map = {'default': 'VS/SG', 'grdlw': 'GRD/dEST', 'vslw': 'VS/dEST'}
        self['Method'] = self['strt'].apply(lambda x: rename_map[x])
        return self


def read_csv(filename: str) -> MyDataframe:
    return MyDataframe(pd.read_csv(filename))

