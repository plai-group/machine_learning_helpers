from string import Template

UBC = 'ubc'
CC  = 'cc'
SLURM_GPU_TOKEN = '#SBATCH --gres=gpu:1'
RRG_TOKEN       = '#SBATCH --account=rrg-kevinlb'
PARTITION_TOKEN = '#SBATCH --partition=$partition'
SUBMISSION_FILE_NAME = 'train.sh'

# 'tags' ($cpu, $mem) are filled dynamically in job_submitter
SLURM_TEMPLATE = f'''#!/bin/bash

{SLURM_GPU_TOKEN}
#SBATCH --mem=$mem
#SBATCH --time=00-$hrs:00
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=$cpu
{PARTITION_TOKEN}
{RRG_TOKEN}

# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `/bin/date`"

$init

echo "Running python command:"
echo "$python_command"
$python_command

echo "Ending run at: `/bin/date`"
echo 'Job complete!'
'''

UBC_PYTHON_INIT_TOKEN = f'source /ubc/cs/research/fwood/vadmas/miniconda3/bin/activate $env'

CC_PYTHON_INIT_TOKEN = f'''
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch
pip install joblib sacred pymongo wandb tensorboard
$pip_install
echo "Virutalenv created"
'''

# hack for cc
CC_PIP_INSTALLS = {}

CC_PIP_INSTALLS['ml3'] = f'''
pip install --no-index torch
pip install GPy
pip install scikit-image
pip install emukit==0.4.6
'''

CC_PIP_INSTALLS['vodasafe'] = f'''
pip install --no-index tensorflow_gpu
pip install scikit_learn
pip install tqdm
pip install imbalanced-learn
'''


#########################
# ---- MONGO FILTERS ----
#########################

# MONGO FILTERS
DEFAULT_FILTER = {"_id": 1,
                  "status": 1,
                  "config": 1,
                  "status": 1,
                  "heartbeat": 1,
                  "experiment.name": 1,
                  "comment": 1,
                  "host.hostname": 1,
                  "captured_out": 1,
                  "result": 1,
                  "stop_time": 1,
                  "start_time": 1}

METRIC_FILTER = {'name': 1,
                 'steps': 1,
                 'timestamps': 1,
                 'values': 1,
                 "run_id": 1,
                 "_id": False}

METRIC_FILTER_NO_TIMESTAMP = {'name': 1,
                              'steps': 1,
                              'values': 1,
                              "run_id": 1,
                              "_id": False}

FILTER_ARTIFACTS = {"artifacts": True}
