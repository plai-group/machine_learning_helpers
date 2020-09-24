from __future__ import division, print_function
import errno
from types import SimpleNamespace
import json
import os
import pickle
import random
import socket
import sys
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from multiprocessing import Pool
from psutil import cpu_count

import contextlib
import joblib
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_nb
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from joblib import Parallel, delayed
from sklearn import metrics
from torch._six import inf



PRUNE_COLUMNS = [
    '__doc__',
    'checkpoint',
    'meta',
    'resources',
    'checkpoint_frequency',
    'cuda',
    'heartbeat',
    'verbose',
    'command',
    'data_dir',
    'experiment',
    'artifact_dir',
    'artifacts',
]

persist_dir = Path('./.persistdir')


def nested_dict():
    return defaultdict(nested_dict)


"""
Loggers and Meters
"""


# from the excellent https://github.com/pytorch/vision/blob/master/references/detection/utils.py
class Meter(object):
    """Track a series of values and provide access to a number of metric
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

        self.M2   = 0
        self.mean = 0
        self.fmt = fmt

    def reset(self):
        self.total = 0.0
        self.count = 0
        self.M2    = 0

    def update(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.M2 += delta * delta2

    @property
    def var(self):
        return self.M2 / self.count if self.count > 2 else 0

    @property
    def sample_var(self):
        return self.M2 / (self.count - 1) if self.count > 2 else 0

    @property
    def median(self):
        return np.median(self.deque)

    @property
    def smoothed_avg(self):
        return np.mean(self.deque)

    @property
    def avg(self):
        return self.total / self.count

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.smoothed_avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter=" ", header='', print_freq=1, wandb=None):
        self.meters = defaultdict(Meter)
        self.delimiter = delimiter
        self.print_freq = print_freq
        self.header = header
        self.wandb = wandb

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int)), f'{k} is of type {type(v)}'
            self.meters[k].update(v)
        if self.wandb is not None:
            self.wandb.log(kwargs)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def step(self, iterable):
        i = 0
        start_time = time.time()
        end = time.time()
        iter_time = Meter(fmt='{avg:.4f}')
        data_time = Meter(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                self.header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                self.header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % self.print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            self.header, total_time_str, total_time / len(iterable)))


class ConvergenceMeter(object):
    """This is a modification of pytorch's ReduceLROnPlateau object
        (https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#ReduceLROnPlateau)
        which acts as a convergence meter. Everything
        is the same as ReduceLROnPlateau, except it doesn't
        require an optimizer and doesn't modify the learning rate.
        When meter.converged(loss) is called it returns a boolean that
        says if the loss has converged.

    Args:
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity metered has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity metered has stopped increasing. Default: 'min'.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. For example, if
            `patience = 2`, then we will ignore the first 2 epochs
            with no improvement, and will only decrease the LR after the
            3rd epoch if the loss still hasn't improved then.
            Default: 10.
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.

    Example:
        >>> meter = Meter('min')
        >>> for epoch in range(10):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     if meter.converged(val_loss):
        >>>         break
    """

    def __init__(self, mode='min', patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, eps=1e-8):

        self.has_converged = False
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def update(self, metrics, epoch=None):
        self.step(metrics, epoch=None)

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self.has_converged = True

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode


class BestMeter(object):
    """ This is like ConvergenceMeter except it stores the
        best result in a set of results. To be used in a
        grid search

    Args:
        mode (str): One of `min`, `max`. In `min` mode, best will
            be updated when the quantity metered is lower than the current best;
            in `max` mode best will be updated when the quantity metered is higher
            than the current best. Default: 'max'.

    """

    def __init__(self, name='value', mode='max', object_name='epoch', verbose=True):

        self.has_converged = False
        self.verbose = verbose
        self.mode = mode
        self.name = name
        self.obj_name = object_name
        self.best = None
        self.best_obj = None
        self.mode_worse = None  # the worse value for the chosen mode
        self._init_is_better(mode=mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse

    def step(self, metrics, **kwargs):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)

        if self.is_better(current, self.best):
            self.best = current
            self.best_obj = kwargs
            if self.verbose:
                print("*********New best**********")
                print(f"{self.name}: ", current)
                print(f"{self.best_obj}")
                print("***************************")
            return True

        return False

    def is_better(self, a, best):
        if self.mode == 'min':
            return a < best
        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best

    def _init_is_better(self, mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode


def save_model(args):
    torch.save(args.model.state_dict(),
               os.path.join(args.wandb.run.dir, "model.h5"))



def collate_fn(batch):
    return tuple(zip(*batch))


def one_vs_all_cv(mylist):
    folds = []
    for i in range(len(mylist)):
        train = mylist.copy()
        test = [train.pop(i)]
        folds += [(train, test)]
    return folds


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python


def flatten(container):
    for i in container:
        if isinstance(i, (list, tuple)):
            yield from flatten(i)
        else:
            yield i

# https://codereview.stackexchange.com/questions/185785/scale-numpy-array-to-certain-range


def scale(x, out_range=(-1, 1)):
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2


def hits_and_misses(y_hat, y_testing):
    tp = sum(y_hat + y_testing > 1)
    tn = sum(y_hat + y_testing == 0)
    fp = sum(y_hat - y_testing > 0)
    fn = sum(y_testing - y_hat > 0)
    return tp, tn, fp, fn


def get_auc(roc):
    prec = roc['prec'].fillna(1)
    recall = roc['recall']
    return metrics.auc(recall, prec)


def classification_metrics(tp, tn, fp, fn):
    precision   = tp / (tp + fp)
    recall      = tp / (tp + fn)
    f1          = 2.0 * (precision * recall / (precision + recall))
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return {
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "prec": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity)
    }


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_data_loader(dataset, batch_size, args, shuffle=True):
    """Args:
        np_array: shape [num_data, data_dim]
        batch_size: int
        device: torch.device object

    Returns: torch.utils.data.DataLoader object
    """

    if args.device == torch.device('cpu'):
        kwargs = {'num_workers': 4, 'pin_memory': True}
    else:
        kwargs = {}

    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)


def split_train_test_by_percentage(dataset, train_percentage=0.8):
    """ split pytorch Dataset object by percentage """
    train_length = int(len(dataset) * train_percentage)
    return torch.utils.data.random_split(dataset, (train_length, len(dataset) - train_length))

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

def pmap(f, arr, n_jobs=-1, notebook=False, **kwargs):
    _tqdm = tqdm_nb if notebook else tqdm
    with tqdm_joblib(_tqdm(total=len(arr))) as progress_bar:
        return Parallel(n_jobs=n_jobs)(delayed(f)(i) for i in arr)

# https://towardsdatascience.com/make-your-own-super-pandas-using-multiproc-1c04f41944a1
def pmap_df(f, df, n_cores=cpu_count(logical=False), n_chunks = 100, is_notebook=True):
    if is_notebook:
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    df_split = np.array_split(df, n_chunks)
    with Pool(n_cores) as p:
        results = list(tqdm(p.imap(f, df_split), total=len(df_split)))
    df = pd.concat(results)
    return df


def put(value, filename):
    persist_dir.mkdir(exist_ok=True)
    filename = persist_dir / filename
    print("Saving to ", filename)
    joblib.dump(value, filename)


def get(filename):
    filename = persist_dir / filename
    assert filename.exists(), "{} doesn't exist".format(filename)
    print("Loading from ", filename)
    return joblib.load(filename)


def smooth(arr, window):
    return pd.Series(arr).rolling(window, min_periods=1).mean().values


def detect_cuda(args):
    if args.cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.cuda = True
    else:
        args.device = torch.device('cpu')
        args.cuda = False
    return args


def logaddexp(a, b):
    """Returns log(exp(a) + exp(b))."""

    return torch.logsumexp(torch.cat([a.unsqueeze(0), b.unsqueeze(0)]), dim=0)


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a tensor.
    """

    log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
    # log_numerator = values
    return values - log_denominator


def make_sparse(sparse_mx, args):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)

    indices = tensor(np.vstack((sparse_mx.row, sparse_mx.col)), args, torch.long)
    values = tensor(sparse_mx.data, args)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a tensor.

    Args:
        values: tensor [dim_1, ..., dim_N]
        dim: n

    Returns:
        result: tensor [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =
                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    return torch.exp(lognormexp(values, dim=dim))


def seed_all(seed, tf=False):
    """Seed all devices deterministically off of seed and somewhat
    independently."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_grads(model):
    return torch.cat([torch.flatten(p.grad.clone()) for p in model.parameters()]).cpu()


def log_ess(log_weight):
    """Log of Effective sample size.
    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, S] (or [S])
    Returns: log of effective sample size [batch_size] (or [1])
    """
    dim = 1 if log_weight.ndimension() == 2 else 0

    return 2 * torch.logsumexp(log_weight, dim=dim) - \
        torch.logsumexp(2 * log_weight, dim=dim)


def ess(log_weight):
    """Effective sample size.
    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, S] (or [S])
    Returns: effective sample size [batch_size] (or [1])
    """

    return torch.exp(log_ess(log_weight))


def get_unique_dir(comment=None):
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    host = socket.gethostname()
    name = f"{current_time}_{host}"
    if comment: name = f"{name}_{comment}"
    return name


def spread(X, N, axis=0):
    """
    Takes a 1-d vector and spreads it out over
    N rows s.t spread(X, N).sum(0) = X
    """
    return (1 / N) * duplicate(X, N, axis)


def duplicate(X, N, axis=0):
    """
    Takes a 1-d vector and duplicates it across
    N rows s.t spread(X, N).sum(axis) = N*X
    """
    order = (N, 1) if axis == 0 else (1, N)
    return X.unsqueeze(axis).repeat(*order)


def safe_json_load(path):
    path = Path(path)
    res = {}
    try:
        if path.stat().st_size != 0:
            with open(path) as data_file:
                res = json.load(data_file)
    except Exception as e:
        print(f"{path} raised exception:")
        print("------------------------------")
        print(e)
        print("------------------------------")

    return res


def get_experiments_from_fs(path):
    path = Path(path)
    assert (path / '_sources/').exists(), f"Bad path: {path}"
    exps = {}
    dfs = []

    for job in path.glob("*"):
        if job.parts[-1] in ['_resources', '_sources']:
            continue
        job_id = job.parts[-1]

        run = safe_json_load(job / 'run.json')
        config = safe_json_load(job / 'config.json')
        metrics = safe_json_load(job / 'metrics.json')

        exps[job_id] = {**config, **run}

        if metrics:
            for metric, v in metrics.items():
                df = pd.DataFrame(v)
                df.index = pd.MultiIndex.from_product([[job_id], [metric], df.index], names=['_id', 'metric', 'index'])
                dfs += [df]

    exps = pd.DataFrame(exps).T
    exps.index.name = '_id'
    if dfs:
        df = pd.concat(dfs).drop('timestamps', axis=1)
    else:
        df = None
    return exps, df


def get_experiments_from_dir(path, observer_name="file_storage_observer", prune=PRUNE_COLUMNS):
    path = Path(path)
    assert path.exists(), f'Bad path: {path}'
    exps = {}
    dfs = {}
    for p in path.rglob(observer_name):
        _id = str(p).replace(f"/{observer_name}", "")
        exp, df = get_experiments_from_fs(p)
        exps[_id] = exp
        if df is None:
            print(f"{p} returned empty df")
        else:
            dfs[_id] = df

    if exps and dfs:
        exps = pd.concat(exps.values(), keys=exps.keys()).droplevel(1)
        dfs = pd.concat(dfs.values(), keys=dfs.keys()).droplevel(1)

        exps.index.name = '_id'
        dfs.index.names = ['_id', 'metric', 'index']
    else:
        raise ValueError(f"results empty! path:{path}")

    if prune:
        exps = exps.remove_columns([c for c in prune if c in exps.columns])
    return exps, dfs


def post_process(exp, df, CUTOFF_EPOCH=2000):
    print(f"{exp[exp.status == 'COMPLETED'].shape[0]} jobs completed")
    print(f"{exp[exp.status == 'RUNNING'].shape[0]} jobs timed out")
    print(f"{exp[exp.status == 'FAILED'].shape[0]} jobs failed")

    # Remove jobs that failed
    exp = exp[exp.status != 'FAILED']

    df = df[df.steps <= CUTOFF_EPOCH]

    # get values at last epoch
    results_at_cutoff = df[df.steps == CUTOFF_EPOCH].reset_index().pivot(index='_id', columns='metric', values='values')

    # join
    exp = exp.join(results_at_cutoff, how='outer')
    return exp, df


def process_dictionary_column(df, column_name):
    if column_name in df.columns:
        return (df
                .join(df[column_name].apply(pd.Series))
                .drop(column_name, 1))
    else:
        return df


def process_tuple_column(df, column_name, output_column_names):
    if column_name in df.columns:
        return df.drop(column_name, 1).assign(**pd.DataFrame(df[column_name].values.tolist(), index=df.index))
    else:
        return df


def process_list_column(df, column_name, output_column_names):
    if column_name in df.columns:
        new = pd.DataFrame(df[column_name].values.tolist(), index=df.index, columns=output_column_names)
        old = df.drop(column_name, 1)
        return old.merge(new, left_index=True, right_index=True)
    else:
        return df


def show_uniques(df):
    for col in df:
        print(f'{col}: ', df[col].unique())


def highlight_best(df, col):
    best = df[col].max()
    return df.style.apply(lambda x: ['background: lightgreen' if (x[col] == best) else '' for i in x], axis=1)


def filter_uninteresting(df):
    df = df.dropna(1, how='all')
    return df[[i for i in df if len(set(df[i])) > 1]]


"""
Safe initalizers
"""


def tensor(data, args=None, dtype=torch.float, device=torch.device('cpu')):
    if args is not None:
        device = args.device
    if torch.is_tensor(data):
        return data.to(dtype=dtype, device=device)
    elif isinstance(data, list) and torch.is_tensor(data[0]):
        return torch.stack(data)
    else:
        return torch.tensor(np.array(data), device=device, dtype=dtype)


def parameter(*args, **kwargs):
    return torch.nn.Parameter(tensor(*args, **kwargs))


def numpyify(val):
    if isinstance(val, dict):
        return {k: np.array(v) for k, v in val.items()}
    if isinstance(val, (float, int, list, np.ndarray)):
        return np.array(val)
    if isinstance(val, (torch.Tensor)):
        return val.cpu().numpy()
    else:
        raise ValueError("Not handled")


def array(val):
    return numpyify(val)


def slist(val):
    """
    safe list
    """
    if isinstance(val, list):
        return val
    else:
        return [val]


def notnan(val):
    return not pd.DataFrame(val).isnull().values.any()


def get_unique_legend(axes):
    unique = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for label, handle in zip(labels, handles):
            unique[label] = handle
    handles, labels = zip(*unique.items())
    return handles, labels


def get_all_dirs(path):
    return [p for p in Path(path).glob("*") if p.is_dir()]


def get_frequency(y):
    y = np.bincount(y)
    ii = np.nonzero(y)[0]
    return {k: v for k, v in zip(ii, y[ii])}


def get_debug_args():
    args = SimpleNamespace()
    args.model_dir = './models'
    args.data_dir = ''

    # Training settings
    args.epochs = 10
    args.seed = 0
    args.cuda = True
    args.warmup = 5000
    args.lr_max = 0.00005
    args.eval_steps = 4
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args
