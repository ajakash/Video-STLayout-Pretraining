# Set the path to save checkpoints
OUTPUT_DIR='/home/aabdujyo/scratch/VideoBoxPretraining/checkpoints/MOMA_sact_default'
# path to Kinetics set (train.csv/val.csv/test.csv)
LOG_DIR='/home/aabdujyo/scratch/VideoBoxPretraining/log/MOMA_sact_default'
# path to pretrain model
MODEL_PATH='/home/aabdujyo/scratch/VideoBoxPretraining/VideoMAE_pretrained_ckpts/checkpoint_ViT-S_K400_ep1600.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)

echo 'Hello, starting to run the timm script!'

python -c "import timm"