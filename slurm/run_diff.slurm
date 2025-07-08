#!/bin/bash
#SBATCH --job-name=run_diff_1
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --partition=gpu_long
#SBATCH --cpus-per-task=10              # Number of CPU cores per task
#SBATCH --time=0:30:00
#SBATCH --mail-type=START,END
#SBATCH --mail-user=shasko@bnl.gov
#SBATCH --gres=gpu:4

# Load virtual env
source ~/ecgenv/bin/activate

python /nsls2/users/shasko/Repos/internship_2025/RNN_files/testing_diffraction.py