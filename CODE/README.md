# EgoRecovery: Acquiring Failure Recovery Ability Through Human Recovery Demonstration

## 1. Environment

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

The project is pinned to Python 3.11 + PyTorch (bf16 AMP) + PyTorch
Lightning + Hydra. Training is multi-GPU (DDP, NCCL).

---

## 2. Training

Main method (FiLM-gated intent co-training, the configuration reported in
the paper):

```bash
python egomimic/trainHydra.py --config-name=train_hpt_aloha_intent_film -m \
    aloha_data_path=/dev/shm/<robot_zarr_dir> \
    hand_data_path=/dev/shm/<human_hand_zarr_dir> \
    trainer.max_steps=20000 \
    trainer.val_check_interval=2500 \
    callbacks.model_checkpoint.every_n_train_steps=2500 \
    name=E_film description=film_modulation_20k
```

Token-concat baseline (Ablation A1 in the paper):

```bash
python egomimic/trainHydra.py --config-name=train_hpt_aloha_intent_rw -m \
    aloha_data_path=/dev/shm/<robot_zarr_dir> \
    hand_data_path=/dev/shm/<human_hand_zarr_dir> \
    name=E_rw_baseline description=token_concat_baseline_for_A1
```

Quick smoke test (8 GPU, ~5 min, validates forward / loss / DDP):

```bash
python egomimic/trainHydra.py --config-name=train_hpt_aloha_intent_film \
    trainer=debug logger=debug norm_stats.sample_frac=0.05 \
    aloha_data_path=<robot_zarr_dir> \
    hand_data_path=<human_hand_zarr_dir> \
    description=film_smoke_test
```

Drop the `-m` flag for interactive single-node runs; keep it to dispatch via
the Hydra submitit launcher on SLURM.
