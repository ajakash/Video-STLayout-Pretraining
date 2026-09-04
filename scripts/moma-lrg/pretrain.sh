# Set the path to save checkpoints
OUTPUT_DIR='checkpoints/'$1
# path to Kinetics set (train.csv/val.csv/test.csv)
LOG_DIR='log/'$1
# path to pretrain model
VID_ENCODER_PATH='VideoMAE_pretrained_ckpts/checkpoint_ViT-B_SS_ep2400.pth'
STLAYOUT_ENCODER_PATH='stlayout_encoder/saved_models/H4_L3_D256_LR0001/bbox2activity_best.pt'

# source /path/to/your/venv/bin/activate
# module load python/3.10
# module load scipy-stack/2023b
# module load cuda

echo 'Starting to run the script!'

# OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
#     --master_port 12320 --nnodes=4  --node_rank=$1 --master_addr=$2 \
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 \
    --master_port 12321 \
    run_stlayout_pretraining.py \
    --num_workers 10 \
    --model vit_base_patch16_224 \
    --batch_size $2 \
    --epochs 150 \
    --num_sample 1 \
    --data_set MOMA_sact_frames_boxes \
    --nb_classes 91 \
    --vid_encoder_init_ckpt ${VID_ENCODER_PATH} \
    --stlayout_encoder_init_ckpt ${STLAYOUT_ENCODER_PATH} \
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
    --dist_eval
    # --eval \
    # --enable_deepspeed
    # --data_path ${DATA_PATH} \
    # change num_wrokers for distributed training
