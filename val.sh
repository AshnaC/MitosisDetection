#!/bin/bash -l
#SBATCH --job-name='train_dpath'
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --mail-type=end,fail
#SBATCH --time=23:15:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

source ~/.bashrc

# Set proxy to access internet from the node
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

module purge
module load python
module load cuda
module load cudnn

# Conda
source activate seminar_dpdlv2

# create a temporary job dir on $WORK
mkdir ${WORK}/$SLURM_JOB_ID
cd ${WORK}/$SLURM_JOB_ID

# copy input file from location where job was submitted, and run
cp -r ${SLURM_SUBMIT_DIR}/. .

echo "Running script"

# Run training script
srun python train_cluster.py -m val  # add training parameters



