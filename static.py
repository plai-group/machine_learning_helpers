from string import Template

UBC = 'ubc'
CC  = 'cc'
SUBMISSION_FILE_NAME = 'train.sh'

# 'tags' ($cpu, $mem) are filled dynamically in job_submitter
SLURM_TEMPLATE = f'''#!/bin/bash


# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `/bin/date`"

export HOME_DIR=$home_dir
export JOB_DIR=$job_dir

$init

echo "Running python command:"
echo "$python_command"

$python_command

echo "Ending run at: `/bin/date`"
echo 'Job complete!'
'''

CC_PYTHON_INIT_TOKEN = f'''
module load python/3.6
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install joblib sacred pymongo wandb tensorboard scikit-image sklearn
$pip_install
echo "Virutalenv created"
'''

# hack for cc
CC_PIP_INSTALLS = {}

CC_PIP_INSTALLS['ml3'] = f'''
pip install GPy pyDOE scikit-image emukit tqdm seaborn python-Levenshtein nltk
'''

CC_PIP_INSTALLS['vodasafe'] = f'''
pip install scikit_learn pillow tqdm imbalanced-learn torchvision pycocotools
'''

