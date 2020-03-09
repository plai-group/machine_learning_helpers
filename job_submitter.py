#!/usr/bin/env python
import os
from string import Template
import time
import datetime
from pprint import pprint
from pathlib import Path
import sys
import itertools
from itertools import chain
import static
import socket
import subprocess

# Global Arguments
COMPUTE_CANADA_HOSTS = ['cedar{}.cedar.computecanada.ca'.format(i) for i in range(10)]
UBC_SLURM_HOSTS = ['borg.cs.ubc.ca']

hostname = socket.gethostname()

# find host and scheduler
if hostname in COMPUTE_CANADA_HOSTS:
    HOST      = static.CC
elif hostname in UBC_SLURM_HOSTS:
    HOST      = static.UBC
else:
    raise ValueError("Scheduler not detected")

# Paths
PROJECT_DIR    = ""
EXPERIMENT_DIR = ""
SRC_PATH       = ""
DATA_DIR       = ""
RESULTS_DIR    = ""

SLEEP_TIME = 0.25

REQUIRED_OPTIONS = set(["gpu","hrs","cpu","mem","partition","env"])

########################
# Main submission loop #
########################

def submit(hyper_params, experiment_name, experiment_dir, manual_mode=False, file_storage_observer=False, **kwargs):
    # Validate arguments
    verify_dirs(experiment_dir, experiment_name)

    # Display info
    hypers = process_hyperparameters(hyper_params)

    assert set(kwargs.keys()) == REQUIRED_OPTIONS, f"{REQUIRED_OPTIONS} must be specified"

    print("------Scheduler Options------")
    pprint(kwargs)
    print("-----------(SLURM)-----------")

    print("Saving results in: {}".format(RESULTS_DIR))
    print("------Sweeping over------")
    pprint(hyper_params)
    print("-------({} runs)-------".format(len(hypers)))

    ask = True
    for idx, hyper_string in enumerate(hypers):
        if manual_mode:
            path = Path("./manual")
            path.mkdir(exist_ok=True)
            scheduler_command, python_command = make_commands(hyper_string, experiment_name, idx, file_storage_observer)
            file_name = static.SUBMISSION_FILE_NAME.replace(".sh", f"{idx}.sh")
            scheduler_command = scheduler_command.replace(static.SUBMISSION_FILE_NAME, file_name)
            make_bash_script(python_command, str(path/file_name), **kwargs)
            print(scheduler_command)
            continue
        if ask:
            flag = input("Submit ({}/{}): {}? (y/n/all/exit) ".format(idx + 1, len(hypers), hyper_string))
        if flag in ['yes', 'all', 'y', 'a']:
            scheduler_command, python_command = make_commands(hyper_string, experiment_name, idx, file_storage_observer)
            make_bash_script(python_command, static.SUBMISSION_FILE_NAME, **kwargs)
            output = subprocess.check_output(scheduler_command,  stderr=subprocess.STDOUT, shell=True)
            print("Submitting ({}/{}): {}".format(idx + 1, len(hypers), output.strip().decode()))

        if flag in ['all', 'a']:
            ask = False
            time.sleep(SLEEP_TIME)

        if flag in ['exit', 'e']:
            sys.exit()


########################
# ---- path management -
########################

# Strictly enforce directory structure
def verify_dirs(experiment_dir, experiment_name):
    experiment_dir = Path(experiment_dir)
    project_dir    = Path(experiment_dir).parents[1]
    src_path       = Path(project_dir) / 'main.py'
    data_dir       = Path(project_dir) / 'data'

    assert project_dir.is_dir(), "{} does not exist".format(project_dir)
    assert data_dir.is_dir(), "{} does not exist".format(data_dir)
    assert src_path.is_file(), "{} does not exist".format(src_path)

    now = datetime.datetime.now()

    results_dir = experiment_dir / 'results' / experiment_name / now.strftime("%Y_%m_%d_%H:%M:%S")

    # make global
    global PROJECT_DIR
    global EXPERIMENT_DIR
    global SRC_PATH
    global DATA_DIR
    global RESULTS_DIR
    PROJECT_DIR, EXPERIMENT_DIR, SRC_PATH, DATA_DIR, RESULTS_DIR = project_dir, experiment_dir, src_path, data_dir, results_dir


def job_name_to_hyper_string(failed_job_names):
    if isinstance(failed_job_names, (str)):
        failed_job_names = [failed_job_names]
    def process(name):
        return (name
                 .replace(".res", "")
                 .replace(".err", "")
                 .replace(".", " "))
    return [process(name) for name in failed_job_names]


#################################
# ------- hyperparameters -------
#################################

def process_hyperparameters(hyper_params):
    if isinstance(hyper_params, dict):
        return make_hyper_string_from_dict(hyper_params)
    elif isinstance(hyper_params, list):
        return list(itertools.chain.from_iterable([make_hyper_string_from_dict(d) for d in hyper_params]))
    else:
        raise ValueError("hyper_params must be either a single dictionary or a list of dictionaries")

# returns strings of form: name1=value1 name2=value2 name3=value3...
def make_hyper_string_from_dict(hyper_dict):
    # Check all values are iterable lists
    def type_check(value):
        if isinstance(value, (list, range)):
            return list(value)
        else:
            return [value]

    hyper_dict = {key: type_check(value) for key, value in hyper_dict.items()}

    commands = []
    for args in itertools.product(*hyper_dict.values()):
        command = "".join(["'{}={}' ".format(k, v) for k, v in zip(hyper_dict.keys(), args)])
        commands.append(command[:-1])

    return commands

def make_bash_script(python_command, file_name, **kwargs):
    file = static.SLURM_TEMPLATE

    # if host is UBC remove RRG
    # if host is cc remove partition
    if HOST == static.UBC:
        file = file.replace(static.RRG_TOKEN, "")
        python_init = Template(static.UBC_PYTHON_INIT_TOKEN).safe_substitute(env=kwargs['env'])
        file = Template(file).safe_substitute(partition=kwargs['partition'])
    else:
        file = file.replace(static.PARTITION_TOKEN, "")
        python_init = Template(static.CC_PYTHON_INIT_TOKEN).safe_substitute(pip_install=static.CC_PIP_INSTALLS[kwargs['env']])

    file = Template(file).safe_substitute(init=python_init)

    if not kwargs['gpu']:
        file = file.replace(static.SLURM_GPU_TOKEN, '')
        file = file.replace("tensorflow_gpu", "tensorflow_cpu")

    file = Template(file).safe_substitute(hrs=kwargs['hrs'], mem=kwargs['mem'], cpu=kwargs['cpu'], python_command=python_command)

    with open(file_name, 'w') as rsh:
        rsh.write(file)


def make_commands(hyper_string, experiment_name, job_idx, file_storage_observer):

    job_dir = Path(RESULTS_DIR) / f"job_{job_idx}"
    job_dir.mkdir(exist_ok=False, parents=True)

    model_dir = job_dir / 'models'
    model_dir.mkdir(exist_ok=False, parents=True)

    python_command = f"python {SRC_PATH} with data_dir={DATA_DIR} model_dir={model_dir} {hyper_string} -p --name {experiment_name}"

    if file_storage_observer or (HOST == static.CC):
        python_command = f"{python_command} -F {RESULTS_DIR}/file_storage_observer"

    args_file_name = job_dir / "args.txt"
    res_name = job_dir / 'results.res'
    err_name = job_dir / 'error.err'

    with open(args_file_name, 'w') as rsh:
        rsh.write(hyper_string.replace("' '", "'\n'"))

    scheduler_command = f"sbatch -o {res_name} -e {err_name} -J {experiment_name} --export=ALL {static.SUBMISSION_FILE_NAME}"

    return scheduler_command, python_command

