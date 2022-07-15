import contextlib
import joblib
import numpy as np
import pandas as pd
import janitor
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    # from https://stackoverflow.com/questions/24983493/tracking-progress-of-joblib-parallel-execution/49950707
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def pmap(f, arr, n_jobs=-1, **kwargs):
    arr = list(arr) # convert generators to list so tqdm works
    with tqdm_joblib(tqdm(total=len(arr))) as progress_bar:
        return Parallel(n_jobs=n_jobs, **kwargs)(delayed(f)(i) for i in arr)

def pmap_df(f, df, n_chunks = 100, groups=None, axis=0, **kwargs):
    # https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
    if groups:
        group_kfold = GroupKFold(n_splits=n_chunks)
        df_split = [df.iloc[test_index]  for _, test_index in group_kfold.split(df, groups=df[groups])]
    else:
        df_split = np.array_split(df, n_chunks)
    df = pd.concat(pmap(f, df_split, **kwargs), axis=axis)
    return df
