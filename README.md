# Video-STLayout-Pretraining

Extends [VideoMAE](https://github.com/MCG-NJU/VideoMAE) with a spatio-temporal layout pretraining objective, targeting [MOMA-LRG](https://moma.stanford.edu/) sub-activity detection. The original VideoMAE README is preserved as [VideoMAE_README.md](VideoMAE_README.md).

## Pipeline

Three stages. Stage 2 has two variants — one using ground-truth MOMA-LRG boxes, one using detector-generated boxes.

```
Stage 1  STLayout encoder training  stlayout_encoder/                            → bbox2activity_best.pt
Stage 2  STLayout pretraining       scripts/moma-lrg{,-det}/pretrain[_slurm].sh
Stage 3  Fine-tuning (MOMA sact)    scripts/moma-lrg{,-det}/finetune[_slurm].sh
```

## Setup

Python 3.6+. Core deps: `torch`, `torchvision` (PyTorch ≥ 1.8 recommended — see [INSTALL.md](INSTALL.md) for CUDA-paired versions), `timm==0.4.12`, `deepspeed==0.5.8`, `tensorboardX`, `decord`, `einops`. Data prep per [DATASET.md](DATASET.md).

Stage 1 ([stlayout_encoder/](stlayout_encoder/README.md)) shares this env; only extra dep is `wandb`.

Additional prerequisites:
- **VideoMAE ViT-B checkpoint** — `checkpoint_ViT-B_SS_ep2400.pth` from the [VideoMAE model zoo](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md).
- **STLayout encoder checkpoint** — `bbox2activity_best.pt` produced by Stage 1 in [stlayout_encoder/](stlayout_encoder/README.md). Annotated and detected pipelines use *different* STLayout encoders (trained on different box sources).

Before running, edit the paths at the top of each script (`OUTPUT_DIR`, `LOG_DIR`, `VID_ENCODER_PATH`, `STLAYOUT_ENCODER_PATH`) and the Slurm `--account` if applicable.

## Stage 2 — Pretraining

```bash
# Annotated-box pipeline
sbatch scripts/moma-lrg/pretrain_slurm.sh      <run_name> <batch_per_gpu> <epochs>

# Detected-box pipeline (epochs hardcoded to 300)
sbatch scripts/moma-lrg-det/pretrain_slurm.sh  <run_name> <batch_per_gpu>
```

Non-Slurm equivalents (`pretrain.sh`) live in the same folders and wrap the same `torch.distributed.launch` command. Some hyperparameters (epochs, and for the detected pipeline the video-encoder init checkpoint) are hardcoded in the non-Slurm variants — edit them directly if you need different values.
Output checkpoint: `checkpoints/<run_name>/checkpoint-best.pth`.

## Stage 3 — Fine-tuning

```bash
sbatch scripts/moma-lrg/finetune_slurm.sh <run_name> <checkpoint_path> <batch_per_gpu>
```

**The `<run_name>` selects the fine-tuning method** (parsed by `utils.py`). The same command supports full fine-tuning and three parameter-efficient variants:

| Run name substring          | Method                               | Trainable params      |
|-----------------------------|--------------------------------------|-----------------------|
| *(none)*                    | Full fine-tuning                     | All                   |
| `BExp-<i1>-<i2>-...`        | Block expansion at block indices     | New (added) blocks    |
| `ADAP-<scaling>-<dim>`      | AdaptFormer-style adapters           | Adapter layers        |
| `LoRA-<scaling>-<rank>`     | LoRA on attention `qkv` and `proj`   | LoRA matrices         |

`scaling` is a float, or `LRN` for a learnable scalar. Example: run name `run1_LoRA-32-32` → LoRA with scaling=32, rank=32.

The detected-box pipeline uses `scripts/moma-lrg-det/finetune[_slurm].sh` — identical logic, kept in that folder for symmetry.

## Example: 2×4 comparison

Baseline (no STLayout PT) vs STLayout PT × {full FT, BExp, ADAP, LoRA}, annotated-box pipeline:

```bash
# Pretraining
sbatch scripts/moma-lrg/pretrain_slurm.sh stlayoutPT_B 4 2000

# Baselines: fine-tune directly from VideoMAE checkpoint
CKPT=VideoMAE_pretrained_ckpts/checkpoint_ViT-B_SS_ep2400.pth
sbatch scripts/moma-lrg/finetune_slurm.sh baseline_FTfull        $CKPT 4
sbatch scripts/moma-lrg/finetune_slurm.sh baseline_BExp-11-10-9  $CKPT 4
sbatch scripts/moma-lrg/finetune_slurm.sh baseline_ADAP-32-32    $CKPT 4
sbatch scripts/moma-lrg/finetune_slurm.sh baseline_LoRA-32-32    $CKPT 4

# With STLayout PT
CKPT=checkpoints/stlayoutPT_B/checkpoint-best.pth
sbatch scripts/moma-lrg/finetune_slurm.sh stlayoutPT_FTfull        $CKPT 4
sbatch scripts/moma-lrg/finetune_slurm.sh stlayoutPT_BExp-11-10-9  $CKPT 4
sbatch scripts/moma-lrg/finetune_slurm.sh stlayoutPT_ADAP-32-32    $CKPT 4
sbatch scripts/moma-lrg/finetune_slurm.sh stlayoutPT_LoRA-32-32    $CKPT 4
```

For the detected-box pipeline, substitute `moma-lrg-det/` in the Stage 2 command.

## License & Attribution

Built on [VideoMAE](https://github.com/MCG-NJU/VideoMAE) (Tong et al., NeurIPS 2022). Released under CC-BY-NC 4.0 (inherited from VideoMAE — see [LICENSE](LICENSE)); non-commercial research use only.
