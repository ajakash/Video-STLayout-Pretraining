#!/bin/bash -l

#SBATCH --account=def-mori
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=11:59:00
#SBATCH --output=cluster_logs/log_%A.out
#SBATCH --error=cluster_logs/error_%A.out

# signal_handler()  >>>###>>> #SBATCH --signal=B:USR1@120
# {
#     # Timeout commands go here
#     echo "Caught USR1 signal"
#     PID=$!
#     # Wait for 2 seconds
#     sleep 2
#     # Kill it
#     kill -INT $PID
# }
# trap 'signal_handler' USR1

hostname

# module load python
# module load cuda
# nvidia-smi

# source ../moma/moma/bin/activate
source /home/aabdujyo/scratch/Video-STLayout-Pretraining/VidMAE/bin/activate
module load python/3.10
module load scipy-stack/2023b
module load cuda
nvidia-smi

# run script from above

# 11/01/25
python main.py --input_dim 116 \
    --hidden_dim $1 \
    --num_heads $2 \
    --num_layers $3 \
    --learning_rate $4 \
    --batch_size 32 \
    --data_dir detection_data/min10boxes/BoxEncoderTraining \
    --output_dir $5

# python main.py --input_dim 116 --learning_rate 0.0001 --num_layers 3 --batch_size 32 --data_dir detection_data/min5boxes/BoxEncoderTraining --output_dir Det_min5_H4_L3_D256_LR0001
# python main.py --input_dim 116 --learning_rate 0.0001 --num_layers 3 --batch_size 32 --data_dir detection_data/min10boxes/BoxEncoderTraining --output_dir Det_min10_H4_L3_D256_LR0001
# python main.py --input_dim 116 --hidden_dim 128 --learning_rate 0.0001 --num_layers 3 --batch_size 32 --data_dir detection_data/min10boxes/BoxEncoderTraining --output_dir Det_min10_H4_L3_D128_LR0001
# python main.py --input_dim 116 --learning_rate 0.0001 --num_layers 4 --batch_size 32 --data_dir detection_data/min10boxes/BoxEncoderTraining --output_dir Det_min10_H4_L4_D256_LR0001
# python main.py --input_dim 116 --hidden_dim 128 --learning_rate 0.00001 --num_layers 2 --batch_size 32 --data_dir detection_data/min10boxes/BoxEncoderTraining --output_dir Det_min10_H4_L2_D128_LR00001
# python main.py --input_dim 116 --learning_rate 0.00001 --num_layers 2 --batch_size 32 --data_dir detection_data/min10boxes/BoxEncoderTraining --output_dir Det_min10_H4_L2_D256_LR00001
# python main.py --input_dim 116 --learning_rate 0.00001 --num_layers 3 --batch_size 32 --data_dir detection_data/min10boxes/BoxEncoderTraining --output_dir Det_min10_H4_L3_D256_LR00001

# 18/10 
# python main.py --output_dir H4_L2_D256_LRsch001
# python main.py --num_layers 3 --batch_size 32 --output_dir H4_L3_D256_LRsch001
# python main.py --num_layers 4 --batch_size 32 --output_dir H4_L4_D256_LRsch001
# python main.py --hidden_dim 128 --output_dir H4_L2_D128_LRsch001
# python main.py --hidden_dim 128 --num_layers 3 --batch_size 32 --output_dir H4_L3_D128_LRsch001
# python main.py --hidden_dim 128 --num_layers 4 --batch_size 32 --output_dir H4_L4_D128_LRsch001
# python main.py --num_heads 2 --num_layers 3 --batch_size 32 --output_dir H2_L3_D256_LRsch001
# python main.py --num_heads 2 --hidden_dim 128 --num_layers 3 --batch_size 32 --output_dir H2_L3_D128_LRsch001

# python main.py --learning_rate 0.00001 --num_layers 3 --batch_size 32 --output_dir H4_L3_D256_LR00001
# python main.py --learning_rate 0.00001 --hidden_dim 128 --num_layers 3 --batch_size 32 --output_dir H4_L3_D128_LR00001
# python main.py --learning_rate 0.0001 --hidden_dim 128 --num_layers 3 --batch_size 32 --output_dir H4_L3_D128_LR0001

# python main.py --learning_rate 0.00001 --num_layers 4 --batch_size 32 --output_dir H4_L4_D256_LR00001
# python main.py --learning_rate 0.00001 --hidden_dim 128 --num_layers 4 --batch_size 32 --output_dir H4_L4_D128_LR00001
# python main.py --learning_rate 0.0001 --hidden_dim 128 --num_layers 4 --batch_size 32 --output_dir H4_L4_D128_LR0001

# try scheduler to start with 0.0001???

# 17/10 morning
# python main.py --hidden_dim 192 --output_dir H4_L2_D192_LR001
# python main.py --hidden_dim 192 --num_heads 2 --output_dir H2_L2_D192_LR001
# python main.py --hidden_dim 128 --output_dir H4_L2_D128_LR001
# python main.py --hidden_dim 128 --num_heads 2 --output_dir H2_L2_D128_LR001
# python main.py --hidden_dim 128 --learning_rate 0.0001 --output_dir H4_L2_D128_LR0001
# python main.py --hidden_dim 128 --learning_rate 0.0001 --output_dir H4_L2_D128_LR0001v2
# python main.py --hidden_dim 128 --learning_rate 0.00001 --output_dir H4_L2_D128_LR00001
# python main.py --learning_rate 0.00001 --output_dir H4_L2_D256_LR00001
# python main.py --learning_rate 0.0001 --num_layers 3 --batch_size 32 --output_dir H4_L3_D256_LR0001
# python main.py --learning_rate 0.0001 --num_layers 3 --batch_size 32 --output_dir H4_L3_D256_LR0001v2
# python main.py --learning_rate 0.0001 --num_layers 4 --batch_size 32 --output_dir H4_L4_D256_LR0001
# python main.py --hidden_dim 252 --learning_rate 0.0001 --num_heads 6 --batch_size 32 --output_dir H6_L2_D252_LR0001

# 09/10
# python main.py --output_dir baseline_H4_L2_D256_LR001
# python main.py --learning_rate 0.0001 --output_dir H4_L2_D256_LR0001
# python main.py --num_heads 2 --output_dir H2_L2_D256_LR001

