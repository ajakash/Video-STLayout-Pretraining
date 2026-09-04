#!/bin/bash -l

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:59:00
#SBATCH --output=cluster_logs/log_%A.out
#SBATCH --error=cluster_logs/error_%A.out

hostname

# Activate your Python environment and load required modules here (cluster-specific).
# source /path/to/your/venv/bin/activate
# module load python/3.10
# module load scipy-stack/2023b
# module load cuda

nvidia-smi

# Evaluate a trained encoder. Example (detected-box canonical):
python test.py --data_dir detection_data/min10boxes/BoxEncoderTraining --model_name Det12hr_min10_H3_L2_D96_LR0001

# Annotated-box canonical:
# python test.py --data_dir data --model_name H4_L3_D256_LR0001
