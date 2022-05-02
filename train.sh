#!/bin/bash

#SBATCH --account=rrg-kevinlb
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --time=00-1:00
#SBATCH --mem=12400M
# ---------------------------------------------------------------------
echo "Current working directory: `pwd`"
echo "Starting run at: `/bin/date`"

export HOME_DIR=/Users/vmasrani/dev/phd
export JOB_DIR=/Users/vmasrani/dev/phd/git_repos_to_maintain/global_python_files/results/tvo_dual_loss_debug/2021_05_16_20:12:29/job_0



echo "Running python command:"
echo "singularity exec --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt --nv -B /home -B /project -B /scratch -B /localscratch $container python train.py --seed=1 --batch_size=2 --S=5 --learning_rate=0.0003 --model_type=bicycle --mon_trials=10 --num_birdviews=1 --num_epochs=1000 --num_rnn_layers=2 --test_interval=200 --val_keeponly=100 --z_dim=2 --test --scene_name=DR_DEU_Merging_MT --latent_loss=kl-sample"

singularity exec --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt --nv -B /home -B /project -B /scratch -B /localscratch $container python train.py --seed=1 --batch_size=2 --S=5 --learning_rate=0.0003 --model_type=bicycle --mon_trials=10 --num_birdviews=1 --num_epochs=1000 --num_rnn_layers=2 --test_interval=200 --val_keeponly=100 --z_dim=2 --test --scene_name=DR_DEU_Merging_MT --latent_loss=kl-sample

echo "Ending run at: `/bin/date`"
echo 'Job complete!'
