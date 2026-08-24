#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=16G
#SBATCH --time=2-23:59:00
#SBATCH --output=experiments/cluster_logs/log_%A.out
#SBATCH --error=experiments/cluster_logs/error_%A.out
#SBATCH --account=rrg-mori

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
nvidia-smi

# activate conda env
source ~/.bashrc
conda activate imp
module load gcc/8.4.0
module load cuda/10.2
export PATH="/home/aabdujyo/softwares/miniconda3/envs/imp/bin:/home/aabdujyo/softwares/miniconda3/bin:$PATH"

# run script from above
#    ***********   CHANGE TIME AS NEEDED   ***********
# python main.py --useWeightedLoss --loss_weight_frequency 10 --loss_weight 10 --output_dir BB3_evenLoss_freq10w10
# python main.py --useWeightedLoss --loss_weight_frequency 10 --loss_weight 5 --output_dir BB3_evenLoss_freq10w5
# python main.py --useWeightedLoss --loss_weight_frequency 10 --loss_weight 2 --output_dir BB3_evenLoss_freq10w2
# python main.py --useWeightedLoss --loss_weight_frequency 5 --loss_weight 10 --output_dir BB3_evenLoss_freq5w10
# python main.py --useWeightedLoss --loss_weight_frequency 5 --loss_weight 5 --output_dir BB3_evenLoss_freq5w5
# python main.py --useWeightedLoss --loss_weight_frequency 5 --loss_weight 2 --output_dir BB3_evenLoss_freq5w2

# python main.py --dataset billiard --useWeightedLoss --loss_weight_frequency 20 --loss_weight 2 --output_dir IB4_evenLoss_freq20w2
# python main.py --dataset billiard --useWeightedLoss --loss_weight_frequency 20 --loss_weight 5 --output_dir IB4_evenLoss_freq20w5
python main.py --dataset billiard --useWeightedLoss --loss_weight_frequency 10 --loss_weight 5 --output_dir IB4_evenLoss_freq10w5
# python main.py --dataset billiard --useWeightedLoss --loss_weight_frequency 20 --loss_weight 10 --output_dir IB4_evenLoss_freq20w10
