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
UBC_PBS_HOSTS   = ['headnode']
UBC_SLURM_HOSTS = ['borg.cs.ubc.ca']

hostname = socket.gethostname()

# find host and scheduler
if hostname in COMPUTE_CANADA_HOSTS:
    HOST      = static.CC_TOKEN
    SCHEDULER = static.SLURM_TOKEN
elif hostname in UBC_PBS_HOSTS:
    HOST      = static.UBC_TOKEN
    SCHEDULER = static.PBS_TOKEN
elif hostname in UBC_SLURM_HOSTS:
    HOST      = static.UBC_TOKEN
    SCHEDULER = static.SLURM_TOKEN
else:
    raise ValueError("Scheduler not detected")

# if scheduler is PBS
if SCHEDULER == static.PBS_TOKEN:
    GPU_TOKEN = static.PBS_GPU_TOKEN
else:
    GPU_TOKEN = static.SLURM_GPU_TOKEN

# if host is UBC
if HOST == static.UBC_TOKEN:
    SLURM_ACCOUNT_TOKEN = static.UBC_SLURM_ACCOUNT_TOKEN
else:
    SLURM_ACCOUNT_TOKEN = static.CC_SLURM_ACCOUNT_TOKEN


# Paths
PROJECT_DIR    = ""
EXPERIMENT_DIR = ""
SRC_PATH    = ""
DATA_DIR    = ""
RESULTS_DIR = ""

SLEEP_TIME=0.25

########################
# Main submission loop #
########################

def submit(hyper_params, experiment_name, experiment_dir,  **kwargs):
    # Validate arguments
    verify_dirs(experiment_dir, experiment_name)

    # Init
    gpu   = kwargs.get('gpu',True)
    cpu   = kwargs.get('cpu',1)
    hrs   = kwargs.get('hrs',1)
    mem   = kwargs.get('mem',"12400M")
    queue = kwargs.get('queue','gpu')
    env   = kwargs.get('env','ml3')

    # Display info
    hypers = process_hyperparameters(hyper_params)

    print("------Scheduler Options------")
    print(f"gpu: {gpu}")
    print(f"cpu: {cpu}")
    print(f"hrs: {hrs}")
    print(f"mem: {mem}")
    print(f"queue: {queue}")
    print(f"env: {env}")
    print("-------({})-------".format(SCHEDULER))

    print("Saving results in: {}".format(RESULTS_DIR))
    print("------Sweeping over------")
    pprint(hyper_params)
    print("-------({} runs)-------".format(len(hypers)))

    ask = True

    for idx, hyper_string in enumerate(hypers):
        if ask:
            flag = input("Submit ({}/{}): {}? (y/n/all/exit) ".format(idx + 1, len(hypers), hyper_string))
        if flag in ['yes', 'all', 'y', 'a']:
            scheduler_command, python_command = make_commands(hyper_string, experiment_name, idx)
            make_bash_script(python_command, gpu, cpu, hrs, mem, queue, env)
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

def make_bash_script(python_command, gpu, cpu, hrs, mem, queue, env):

    # if host is UBC
    if HOST == static.UBC_TOKEN:
        PYTHON_INIT_TOKEN = static.UBC_DEFAULT_PYTHON_INIT_TOKEN
    else:
        PYTHON_INIT_TOKEN = static.CC_PYTHON_INIT[env]

    gpu_token = GPU_TOKEN if gpu else ''

    init = Template(PYTHON_INIT_TOKEN).safe_substitute(env=env)

    if SCHEDULER == static.PBS_TOKEN:
        template = static.PBS_TEMPLATE.safe_substitute(hrs=hrs,
                                                       mem=mem,
                                                       cpu=cpu,
                                                       init=init,
                                                       queue=queue,
                                                       python_command=python_command,
                                                       gpu=gpu_token)
    else:
        template = static.SLURM_TEMPLATE.safe_substitute(hrs=hrs,
                                                         mem=mem,
                                                         cpu=cpu,
                                                         init=init,
                                                         queue=queue,
                                                         python_command=python_command,
                                                         gpu=gpu_token,
                                                         account=SLURM_ACCOUNT_TOKEN)

    with open(static.BASH_FILE_NAME_TOKEN, 'w') as rsh:
        rsh.write(template)

    return gpu, hrs, mem, queue, env



def make_commands(hyper_string, experiment_name, job_idx):

    job_dir = Path(RESULTS_DIR) / f"job_{job_idx}"
    job_dir.mkdir(exist_ok=False, parents=True)

    model_dir = job_dir / 'models'
    model_dir.mkdir(exist_ok=False, parents=True)

    python_command = f"python {SRC_PATH} with data_dir={DATA_DIR} model_dir={model_dir} {hyper_string} -p --name {experiment_name}"

    if HOST == static.CC_TOKEN:
        python_command = f"{python_command} -F {RESULTS_DIR}/file_storage_observer"

    args_file_name = job_dir / "args.txt"
    res_name = job_dir / 'results.res'
    err_name = job_dir / 'error.err'

    with open(args_file_name, 'w') as rsh:
        rsh.write(hyper_string.replace("' '", "'\n'"))

    if SCHEDULER == static.SLURM_TOKEN:
        scheduler_command = (f"sbatch -o {res_name} -e {err_name} "
                   f"-J {experiment_name} "
                   f"--export=ALL {static.BASH_FILE_NAME_TOKEN}")
    else:
        scheduler_command = (f"qsub -o {res_name} -e {err_name} "
                   f"-N {experiment_name} "
                   f"-d {PROJECT_DIR} "
                   f" {static.BASH_FILE_NAME_TOKEN}")
    return scheduler_command, python_command
