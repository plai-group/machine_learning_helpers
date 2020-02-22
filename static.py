from string import Template

# 'tags' are filled dynamically
# 'tokens' are actual values

################################
# Tags to be filled dynamically
# within job_submitter
########################
# pip install --no-index --upgrade pip
# pip install --no-index tensorflow
# pip install --no-index torch
# pip install sacred
# pip install pymongo
# pip install scikit_learn
# pip install tqdm
# pip install imbalanced-learn


QUEUE_TAG = "$queue"
MEM_TAG = "$mem"
HRS_TAG = "$hrs"
GPU_TAG = "$gpu"
VIRTUAL_ENV_TAG = "$env"
ACCOUNT_TAG="$account"
PYTHON_INIT_TAG="$init"
PYTHON_COMMAND_TAG = "${PYTHON_COMMAND}" #filled in by submission script

########################
# tokens to be used whereever
########################
PBS_TOKEN   = 'PBS'
SLURM_TOKEN = 'SLURM'
UBC_TOKEN   = 'UBC'
CC_TOKEN    = 'CC'
BASH_FILE_NAME_TOKEN    = "train.sh"
PBS_DEFAULT_QUEUE_TOKEN = "laura"
PBS_DEFAULT_GPU_QUEUE_TOKEN = "gpu"

CC_SLURM_ACCOUNT_TOKEN = '#SBATCH --account=rrg-kevinlb'
UBC_SLURM_ACCOUNT_TOKEN = '#SBATCH --partition=plai'

PBS_GPU_TOKEN = '#PBS -l gpus=1'
SLURM_GPU_TOKEN = '#SBATCH --gres=gpu:1'

CC_DEFAULT_PYTHON_INIT_TOKEN = f'''
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install -r /home/vadmas/dev/envs/{VIRTUAL_ENV_TAG}.txt

echo "Virutalenv created "
'''

UBC_DEFAULT_PYTHON_INIT_TOKEN = f'source /ubc/cs/research/fwood/vadmas/miniconda3/bin/activate {VIRTUAL_ENV_TAG}'

########################
# templates
########################

SLURM_HEADER = f'''
#SBATCH --mem={MEM_TAG}
#SBATCH --time=00-{HRS_TAG}:00
#SBATCH --output=%x-%j.out
{ACCOUNT_TAG}
{GPU_TAG}'''

PBS_HEADER = f'''
#PBS -q {QUEUE_TAG}
#PBS -V
#PBS -l walltime={HRS_TAG}:00:00
#PBS -l mem={MEM_TAG}
{GPU_TAG}
'''

BODY = f'''
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `/bin/date`"

{PYTHON_INIT_TAG}

echo "Running python command:"
echo {PYTHON_COMMAND_TAG}
eval {PYTHON_COMMAND_TAG}

echo "Ending run at: `/bin/date`"
echo 'Job complete!'
'''

SLURM_TEMPLATE = Template(f'''#!/bin/bash
{SLURM_HEADER}
{BODY}
''')

PBS_TEMPLATE = Template(f'''#!/bin/bash
{PBS_HEADER}
{BODY}
''')

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

DEFAULT_COLUMNS = ["captured_out",
                   "data_dir",
                   "checkpoint",
                   "checkpoint_frequency",
                   "cuda",
                   "__doc__",
                   "model_dir",
                   'test_during_training',
                   'test_frequency',
                   'train_only',
                   'heartbeat']
