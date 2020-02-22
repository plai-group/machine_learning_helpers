#!/usr/bin/env python
import os
import subprocess
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
from static import JOB_COMPLETE_TOKEN

# Global Arguments
BASH_FILE_NAME        = "train.sh"
PBS_DEFAULT_QUEUE     = "laura"
PBS_DEFAULT_GPU_QUEUE = "gpu"

COMPUTE_CANADA_HOSTS = ['cedar{}.cedar.computecanada.ca'.format(i) for i in range(10)]
UBC_PBS_HOSTS   = ['headnode']
UBC_SLURM_HOSTS = ['borg.cs.ubc.ca']

hostname = 'headnodes'
# hostname = socket.gethostname()

if hostname in COMPUTE_CANADA_HOSTS:
    HOST        = static.CC_TOKEN
    SCHEDULER   = static.SLURM_TOKEN
    GPU_OPTIONS = static.SLURM_CC_GPU_OPTIONS
    CPU_OPTIONS = static.SLURM_CC_CPU_OPTIONS
elif hostname in UBC_PBS_HOSTS:
    HOST        = static.UBC_TOKEN
    SCHEDULER   = static.PBS_TOKEN
    GPU_OPTIONS = static.PBS_GPU_OPTIONS
    CPU_OPTIONS = static.PBS_CPU_OPTIONS
elif hostname in UBC_SLURM_HOSTS:
    HOST        = static.UBC_TOKEN
    SCHEDULER   = static.SLURM_TOKEN
    GPU_OPTIONS = static.SLURM_UBC_GPU_OPTIONS
    CPU_OPTIONS = static.SLURM_UBC_CPU_OPTIONS
else:
    raise ValueError("Scheduler not detected")

# Paths
PROJECT_DIR    = ""
EXPERIMENT_DIR = ""
SRC_PATH    = ""
DATA_DIR    = ""
RESULTS_DIR = ""
MODEL_DIR   = ""

########################
# Main submission loop #
########################

def submit(hyper_params,
           experiment_name,
           experiment_dir,
           gpu=False,
           hrs=12,
           mem="12400M",
           queue=PBS_DEFAULT_GPU_QUEUE,
           sleep_time=0.1,
           env='default'):
    # Validate Arguments and create bash script
    verify_dirs(experiment_dir, experiment_name)

    # Init
    options = make_bash_script(gpu, hrs, mem, queue, env)

    # Display info
    hypers = process_hyperparameters(hyper_params)

    print("------Scheduler Options------")
    print(options.strip())
    print("-------({})-------".format(SCHEDULER))

    print("Saving results in: {}".format(RESULTS_DIR))
    print("------Sweeping over------")
    pprint(hyper_params)
    print("-------({} runs)-------".format(len(hypers)))

    # Submit
    _submit_jobs(hypers, experiment_name, sleep_time)


# def submit_patch(patch_dir, gpu=False, hrs=12, mem="12400M", queue=PBS_DEFAULT_GPU_QUEUE, sleep_time=0.1):
#     # Validate Arguments and create bash script
#     experiment_name = verify_patch_path(patch_dir)

#     # Init
#     options = make_bash_script(gpu, hrs, mem, queue)

#     print("------Scheduler Options------")
#     print(options.strip())
#     print("-------({})-------".format(SCHEDULER))

#     # Display info
#     print("Saving results in: {}".format(RESULTS_DIR))

#     paths, hypers = get_patch_jobs(patch_dir)
#     print("------Failed jobs------")
#     pprint(paths)
#     print("-------({} runs)-------".format(len(hypers)))

#     # Submit
#     _submit_jobs(hypers, experiment_name, sleep_time)


def _submit_jobs(hypers, experiment_name, sleep_time):
    # Submit
    ask = True
    for idx, hyper_string in enumerate(hypers):
        python_command = make_python_command(hyper_string, experiment_name)
        command = make_scheduler_command(python_command, hyper_string, experiment_name, idx)

        if ask:
            flag = input("Submit ({}/{}): {}? (y/n/all/exit) ".format(idx + 1, len(hypers), python_command))

        if flag in ['yes', 'all', 'y', 'a']:
            output = subprocess.check_output(command,  stderr=subprocess.STDOUT, shell=True)
            print("Submitting ({}/{}): {}".format(idx + 1, len(hypers), output.strip().decode()))

        if flag in ['all', 'a']:
            ask = False
            time.sleep(sleep_time)

        if flag in ['exit', 'e']:
            sys.exit()


########################
# ---- Verifiers ------
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
    models_dir  = results_dir / 'models'

    results_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True, parents=True)

    # make global
    global PROJECT_DIR
    global EXPERIMENT_DIR
    global SRC_PATH
    global DATA_DIR
    global RESULTS_DIR
    global MODEL_DIR
    PROJECT_DIR, EXPERIMENT_DIR, SRC_PATH, DATA_DIR, RESULTS_DIR, MODEL_DIR = project_dir, experiment_dir, src_path, data_dir, results_dir, models_dir


#########################
# ------- Patch  --------
#########################

# Strictly enforce directory structure
def verify_patch_path(patch_dir):
    assert "*.res" not in patch_dir, "patch patch should not contain '*.res'"
    patch_dir       = Path(patch_dir)
    experiment_dir  = patch_dir.parents[2]
    project_dir     = experiment_dir.parents[1]

    src_path       = Path(project_dir) / 'main.py'
    data_dir       = Path(project_dir) / 'data'

    assert project_dir.is_dir(), "{} does not exist".format(project_dir)
    assert data_dir.is_dir(), "{} does not exist".format(data_dir)
    assert src_path.is_file(), "{} does not exist".format(src_path)

    results_dir = patch_dir / 'patch'
    models_dir  = results_dir / 'models'

    results_dir.mkdir(exist_ok=True, parents=True)
    models_dir.mkdir(exist_ok=True, parents=True)

    # make global
    global PROJECT_DIR
    global EXPERIMENT_DIR
    global SRC_PATH
    global DATA_DIR
    global RESULTS_DIR
    global MODEL_DIR
    PROJECT_DIR, EXPERIMENT_DIR, SRC_PATH, DATA_DIR, RESULTS_DIR, MODEL_DIR = project_dir, experiment_dir, src_path, data_dir, results_dir, models_dir

    experiment_name = patch_dir.parts[-2]
    return experiment_name


def job_name_to_hyper_string(failed_job_names):
    if isinstance(failed_job_names, (str)):
        failed_job_names = [failed_job_names]
    def process(name):
        return (name
                 .replace(".res", "")
                 .replace(".err", "")
                 .replace(".", " "))
    return [process(name) for name in failed_job_names]

def get_failed_jobs(path):
    grep_path = str(Path(path) / "*.res")
    print(f"Looking for failed jobs in: {grep_path}")
    grep_results = subprocess.check_output(f'grep -L "{JOB_COMPLETE_TOKEN}" {grep_path}', shell=True)
    grep_results = grep_results.decode("utf-8").split()
    return [path.replace(str(RESULTS_DIR.parent) + "/", "") for path in grep_results]

def get_patch_jobs(path):
    paths = get_failed_jobs(path)
    hyper_string = job_name_to_hyper_string(paths)
    return paths, hyper_string


#########################
# ------- Makers --------
#########################

def make_bash_script(gpu, hrs, mem, queue, env):
    options  = GPU_OPTIONS if gpu else CPU_OPTIONS
    options = options.format(hrs=hrs, mem=mem, queue=queue)
    import ipdb; ipdb.set_trace()
    template = static.TEMPLATES[env][HOST]
    bash_contents = "#!/bin/bash {} {}".format(options, template)

    with open(BASH_FILE_NAME, 'w') as rsh:
        rsh.write(bash_contents)

    return options


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
        command = "".join(["{}={} ".format(k, v) for k, v in zip(hyper_dict.keys(), args)])
        commands.append(command[:-1])

    return commands


def make_python_command(hyper_string, experiment_name):
    return ("python {src_path} with "
            "data_dir={data_dir} "
            "model_dir={model_dir} "
            "{hyper_string} "
            "-p --name {experiment_name}").format(
                src_path=SRC_PATH,
                data_dir=DATA_DIR,
                model_dir=MODEL_DIR,
                hyper_string=hyper_string,
                experiment_name=experiment_name)


def make_scheduler_command(python_command, hyper_string, experiment_name, job_idx):
    job_dir = Path(RESULTS_DIR) / f"job_{job_idx}"
    job_dir.mkdir(exist_ok=False, parents=False)

    args_file_name = job_dir / "args.txt"
    res_name = job_dir / 'results.res'
    err_name = job_dir / 'error.err'

    with open(args_file_name, 'w') as rsh:
        rsh.write(hyper_string)

    if SCHEDULER == static.SLURM_TOKEN:
        command = (f"sbatch -o {res_name} -e {err_name} "
                   f"-J {experiment_name} "
                   f"--export=ALL,PYTHON_COMMAND=\"{python_command}\" {BASH_FILE_NAME}")
    else:
        command = (f"qsub -o {res_name} -e {err_name} "
                   f"-N {experiment_name} "
                   f"-d {PROJECT_DIR} "
                   f"-v PYTHON_COMMAND=\"{python_command}\" {BASH_FILE_NAME}")
    return command

if __name__ == "__main__":
    gpu =  False
    hrs =  1
    mem =  "12400M"
    queue =  'parallel'
    sleep_time = 1.0
    env = 'vodasafe'
    import ipdb; ipdb.set_trace()
    make_bash_script(gpu, hrs, mem, queue, env)
