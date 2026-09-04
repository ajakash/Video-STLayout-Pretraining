#!/bin/bash -l

#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH --output=cluster_logs/log_%A.out
#SBATCH --error=cluster_logs/error_%A.out

hostname

# Activate your Python environment and load required modules here (cluster-specific).
# source /path/to/your/venv/bin/activate
# module load python/3.10
# module load scipy-stack/2023b
# module load cuda

nvidia-smi

# Detected-box variant: pass hidden_dim, num_heads, num_layers, learning_rate, output_dir as $1..$5
python main.py --input_dim 116 \
    --hidden_dim $1 \
    --num_heads $2 \
    --num_layers $3 \
    --learning_rate $4 \
    --batch_size 32 \
    --data_dir detection_data/min10boxes/BoxEncoderTraining \
    --output_dir $5
