from pymongo import MongoClient
from db import test_connection, open_ssh, LOCAL_URI, REMOTE_URI, DATABASE_NAME
import pandas as pd
import numpy as np
from pandas import json_normalize
from joblib import Memory
from incense import ExperimentLoader
from pathlib import Path
import janitor
import gridfs
from static import DEFAULT_FILTER, METRIC_FILTER, METRIC_FILTER_NO_TIMESTAMP, FILTER_ARTIFACTS

cache_location = Path('/Users/vmasrani/.joblib_cache')
memory = Memory(cache_location, verbose=0)
memory.clear(warn=True)

if test_connection(LOCAL_URI):
    URI = LOCAL_URI
elif test_connection(REMOTE_URI):
    URI = REMOTE_URI
else:
    raise RuntimeError("""Error, connection not established. Tried:
{}
{}
Did you start an SSH tunnel?
""".format(LOCAL_URI, REMOTE_URI))

client = MongoClient(URI).vadmas_experiments
fs = gridfs.GridFS(client)

print("MongoClient ('client') loaded into namespace")
print("client =", client)
print("""
    Available functions
    -------------------------------------------------------------
    get_experiments(query, db_filter=DEFAULT_FILTER, exps_only=False, **kwargs)
    get_metrics(exps, timestamps=False)
    get_artifacts(exps)
    delete_exp(exps, confirmed=False)
    delete_exp_by_query(query, **kwargs)
    ")
    -------------------------------------------------------------
    """)

@memory.cache
def get_experiments(query, exps_only=False, drop_defaults=True, **kwargs):
    query_result = list(client.runs.find(query, DEFAULT_FILTER))
    assert query_result, "Results are empty for query: {}".format(query)

    # json
    exps = json_normalize(query_result).set_index("_id")
    # runtime
    if 'stop_time' in exps.columns:
        exps['run_time'] = (exps['stop_time'] - exps['start_time'])
    else:
        exps['run_time'] = (exps['heartbeat'] - exps['start_time'])

    # clean up cols
    config_cols = [c for c in exps.columns if 'config' in c]
    exps = (exps
            .reorder_columns(config_cols)
            .rename_columns({"host.hostname":"hostname"})
            .rename_columns({c: c.replace("config.", "") for c in config_cols}))

    if exps_only:
        return exps

    return exps, get_metrics(query_result, **kwargs)


@memory.cache
def get_metrics(exps, timestamps=False):
    if not isinstance(exps, list):
        exps = [exps]

    query = {"run_id": {"$in": [(e["_id"]) for e in exps]}}

    mfilter = METRIC_FILTER if timestamps else METRIC_FILTER_NO_TIMESTAMP
    metric_db_entries = client.metrics.find(query, mfilter)

    metrics = {}

    for e in metric_db_entries:
        key = (e.pop("run_id"), e.pop("name"))
        metrics[key] = pd.DataFrame(e)

    df = pd.concat(metrics)
    df.index.names = ['_id', 'metric', 'index']

    return df


@memory.cache
def get_artifacts(exps):
    if not isinstance(exps, list):
        exps = [exps]

    query = {"_id": {"$in": exps}}
    returned_artifact_ids = client.runs.find({}, FILTER_ARTIFACTS)

    for e in returned_artifact_ids:
        for artifact in e['artifacts']:
            art_file = fs.get(artifact['file_id'])
            f = open(str(e['_id']) + ':' + artifact['name'], 'wb')
            f.write(art_file.read())
            f.close()


# Use incense for now
def delete_exp(exps):
    if isinstance(exps, (pd.core.frame.DataFrame, pd.core.series.Series)):
        ids = list(exps.index)
    elif isinstance(exps, pd.Index):
        ids = list(exps)
    elif isinstance(exps, (int, np.integer)):
        ids = [exps]
    elif isinstance(exps, list):
        ids = exps
    else:
        raise ValueError("Unknown type")

    loader = ExperimentLoader(mongo_uri=URI, db_name='vadmas_experiments')

    flag = input("Delete {} experiments? (y/n) ".format(len(ids)))
    if flag.lower() in ['yes', 'y']:
        for id in ids:
            exp = loader.find_by_id(id)
            exp.delete(True)
            memory.clear(warn=False)

def delete_exp_by_query(query, **kwargs):
    exps = get_experiments(query, exps_only=True, **kwargs)
    flag = input("Delete {} experiments? (y/n) ".format(len(exps)))

    if flag.lower() in ['yes', 'y']:
        delete_exp(exps)

