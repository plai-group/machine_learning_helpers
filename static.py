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
pip install joblib sacred pymongo wandb tensorboard scikit-image sklearn
$pip_install
echo "Virutalenv created"
'''

# hack for cc
CC_PIP_INSTALLS = {}

CC_PIP_INSTALLS['ml3'] = f'''
pip install --no-index torch
pip install GPy
pip install pyDOE
pip install scikit-image
pip install emukit
pip install tqdm
'''

CC_PIP_INSTALLS['vodasafe'] = f'''
pip install scikit_learn pillow tqdm imbalanced-learn torchvision pycocotools
'''


