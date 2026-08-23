# Set the path to save checkpoints
OUTPUT_DIR='/home/aabdujyo/scratch/Video-STLayout-Pretraining/checkpoints/sact_defSpl_S'
# path to Kinetics set (train.csv/val.csv/test.csv)
LOG_DIR='/home/aabdujyo/scratch/Video-STLayout-Pretraining/log/sact_defSpl_S'
# path to pretrain model
MODEL_PATH='/home/aabdujyo/scratch/Video-STLayout-Pretraining/VideoMAE_pretrained_ckpts/checkpoint_ViT-S_K400_ep1600.pth'

# We add repeated_aug (--num_sample = 2) on Kinetics-400 here, 
# which could better performance while need more time for fine-tuning

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)

source /home/aabdujyo/scratch/Video-STLayout-Pretraining/VidMAE/bin/activate
module load python/3.10
module load scipy-stack/2023b
module load cuda

echo 'Starting to run the script!'

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
#     --master_port 12320 --nnodes=4  --node_rank=$1 --master_addr=$2 \
python run_class_finetuning.py \
    --num_workers 0 \
    --model vit_small_patch16_224 \
    --epochs 150 \
    --batch_size 4 \
    --num_sample 1 \
    --save_ckpt_freq 10 \
    --data_set MOMA_sact \
    --nb_classes 91 \
    --finetune ${MODEL_PATH} \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 \
    --short_side_size 224 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 5 \
    --test_num_crop 3 #\
    # --eval #\
    # --dist_eval \
    # --enable_deepspeed 
    # --data_path ${DATA_PATH} \
    # change num_wrokers for distributed training
