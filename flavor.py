from parallel import pmap_df
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
def str_get_numbers(df, column_name: str):
    """Wrapper around df.str.replace"""

    df[column_name] = df[column_name].str.extract('(\d+)', expand=False)
    return df

@pf.register_dataframe_method
def str_drop_after(df, pat, column_name: str):
    """Wrapper around df.str.replace"""

    df[column_name] = df[column_name].str.split(pat='[', expand=True)
    return df

# collapse_levels(sep='_')
# @pf.register_dataframe_method
# def flatten_cols(df):
#     df.columns = ['_'.join(col).strip() for col in df.columns.values]
#     return df


# to be converted to flavor when I find myself needing them

# def process_dictionary_column(df, column_name):
#     if column_name in df.columns:
#         return (df
#                 .join(df[column_name].apply(pd.Series))
#                 .drop(column_name, 1))
#     else:
#         return df


# def process_tuple_column(df, column_name, output_column_names):
#     if column_name in df.columns:
#         return df.drop(column_name, 1).assign(**pd.DataFrame(df[column_name].values.tolist(), index=df.index))
#     else:
#         return df


# def process_list_column(df, column_name, output_column_names):
#     if column_name in df.columns:
#         new = pd.DataFrame(df[column_name].values.tolist(), index=df.index, columns=output_column_names)
#         old = df.drop(column_name, 1)
#         return old.merge(new, left_index=True, right_index=True)
#     else:
#         return df


# def show_uniques(df):
#     for col in df:
#         print(f'{col}: ', df[col].unique())


# def highlight_best(df, col):
#     best = df[col].max()
#     return df.style.apply(lambda x: ['background: lightgreen' if (x[col] == best) else '' for i in x], axis=1)


# def filter_uninteresting(df):
#     df = df.dropna(1, how='all')
#     return df[[i for i in df if len(set(df[i])) > 1]]

