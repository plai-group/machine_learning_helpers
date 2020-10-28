import pandas as pd
from parallel import pmap, pmap_df
import pandas_flavor as pf
import numpy as np

@pf.register_dataframe_method
def highlight_best(df,
                   col,
                   criterion=np.max,
                   style='background: lightgreen'
                  ):
    # other useful styles: 'font-weight: bold'
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html
    best = df.apply(criterion)[col]
    return df.style.apply(lambda x: [style if (x[col] == best) else '' for i in x], axis=1)


@pf.register_dataframe_method
def remove_boring(df):
    df = df.dropna(1, how='all')
    return df[[i for i in df if len(set(df[i])) > 1]]

@pf.register_dataframe_method
def ppipe(df, f, **kwargs):
    return pmap_df(f, df, **kwargs)

@pf.register_dataframe_method
def pgroupby(df, groups, f,  **kwargs):
    '''# mirror groupby order (group then agg)
    replace:
        results = df.groupby(['col1','col2']).apply(f)
    with:
        results = df.pgroupby(['col1','col2'], f)
    '''
    # split into names and groups
    names, df_split = zip(*[(n,g) for n,g in df.groupby(groups)])
    # pmap groups
    out = pmap(f, df_split, **kwargs)
    # reassemble and return
    groups = [groups] if isinstance(groups, str) else groups
    return pd.concat([pd.concat({k: v}, names=groups) for k, v in zip(names, out)])

