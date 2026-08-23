# Set the path to save checkpoints
OUTPUT_DIR='/home/aabdujyo/scratch/Video-STLayout-Pretraining/checkpoints/'$1
# path to Kinetics set (train.csv/val.csv/test.csv)
LOG_DIR='/home/aabdujyo/scratch/Video-STLayout-Pretraining/log/'$1
# path to pretrain model
# VID_ENCODER_PATH='/home/aabdujyo/scratch/Video-STLayout-Pretraining/VideoMAE_pretrained_ckpts/checkpoint_ViT-B_SS_ep2400.pth'
VID_ENCODER_PATH='/home/aabdujyo/scratch/Video-STLayout-Pretraining/checkpoints/stlayoutPTdet_sact_B_3daysV2/checkpoint-'$3

# BOX_ENCODER_PATH='/home/aabdujyo/scratch/activity_moma/saved_models/H4_L3_D256_LR0001v2/bbox2activity_best.pt'
BOX_ENCODER_PATH='/home/aabdujyo/scratch/activity_moma/saved_models/Det12hr_min10_H3_L2_D96_LR0001/bbox2activity_best.pt'

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
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 12321 \
    run_stlayout_pretraining.py \
    --num_workers 10 \
    --model vit_base_patch16_224 \
    --batch_size $2 \
    --epochs 300 \
    --num_sample 1 \
    --data_set MOMA_sact_frames_detected_boxes \
    --nb_classes 91 \
    --vid_encoder_init_ckpt ${VID_ENCODER_PATH} \
    --box_encoder_init_ckpt ${BOX_ENCODER_PATH} \
    --log_dir ${LOG_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 10 \
    --num_frames 16 \
    --sampling_rate 4 \
    --opt adamw \
    --lr 5e-4 \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 5 \
    --test_num_crop 3 \
    --distributed \
    --eval \
    --dist_eval 
    # --enable_deepspeed 
    # --data_path ${DATA_PATH} \
    # change num_wrokers for distributed training
