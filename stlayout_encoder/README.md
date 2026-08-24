# STLayout Encoder

Stage 1 of the parent [Video-STLayout-Pretraining](../README.md) pipeline. Trains a small transformer that maps a bounding-box sequence to a MOMA-LRG sub-activity class. The resulting checkpoint (`bbox2activity_best.pt`) is consumed by Stage 2 as the STLayout target encoder.

## Contents

- `main.py` — training entrypoint
- `models/bbox2activity_transformer.py` — the encoder model (`BBox2Activity`)
- `dataset_moma.py`, `dataset_to_dict.py`, `dataset_to_tensor.py` — MOMA-LRG data loading / prep
- `test.py` — evaluation
- `momaapi/` — bundled [Stanford MOMA API](https://github.com/d1ngn1gefe1/moma) (loads MOMA-LRG annotations & taxonomy). See its own `docs/` for details.
- `third_party/` — data-prep helpers for the detected-box variant (DeepSORT tracking, Detectron2, SlowFast).
- `job_scripts/run.sh` — Slurm submission script (Compute-Canada-specific paths; edit for your cluster).

## Setup

Uses the parent repo's Python environment. Extra dependency: `wandb`.

```bash
pip install wandb
```

Optional (only for `momaapi` visualizers): `pygraphviz` (requires `graphviz-dev` at the system level).

## Data

Trains on `<data_dir>/train.pt` and `<data_dir>/val.pt`. Each `.pt` is a dict with keys `bbox_sets`, `bbox_masks`, `sact_label_indices`. Generate these with `dataset_to_tensor.py` from the raw MOMA-LRG annotations.

## Training

The parent repo's two Stage-2 pipelines each consume a *different* encoder trained here:

**Annotated boxes** (`H4_L3_D256_LR0001v2` — 4 heads, 3 layers, hidden 256, lr 1e-4):
```bash
python main.py \
    --data_dir data \
    --num_heads 4 --num_layers 3 --hidden_dim 256 \
    --learning_rate 0.0001 --batch_size 32 \
    --output_dir H4_L3_D256_LR0001v2
```

**Detected boxes** (`Det12hr_min10_H3_L2_D96_LR0001` — 3 heads, 2 layers, hidden 96):
```bash
python main.py \
    --input_dim 116 \
    --data_dir detection_data/min10boxes/BoxEncoderTraining \
    --num_heads 3 --num_layers 2 --hidden_dim 96 \
    --learning_rate 0.0001 --batch_size 32 \
    --output_dir Det12hr_min10_H3_L2_D96_LR0001
```

`input_dim` differs because the detected-box feature vector packs different metadata than the annotated one.

## Output

Best-validation checkpoint is written to:
```
saved_models/<output_dir>/bbox2activity_best.pt
```

The parent repo's Stage-2 scripts reference this file via `STLAYOUT_ENCODER_PATH`.

## Attribution

The `momaapi/` package and the `docs/`, `figures/`, `download/` folders are from the [Stanford MOMA-LRG API](https://github.com/d1ngn1gefe1/moma) (Luo et al.). See the upstream repo for license.
