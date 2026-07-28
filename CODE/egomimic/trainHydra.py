import copy
import os
import signal
from typing import Any, Dict, List, Optional, Tuple

# NCCL flight recorder — on watchdog timeout, dump per-rank stacks so we can see
# which rank hung. Must be set before torch is imported / process group inits.
os.environ.setdefault("TORCH_NCCL_TRACE_BUFFER_SIZE", "2000")
os.environ.setdefault("TORCH_NCCL_DUMP_ON_TIMEOUT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TORCH_NCCL_DEBUG_INFO_TEMP_FILE", "/tmp/nccl_trace_rank_")

# Alibaba DSW images ship with NCCL_IB_HCA=erdma baked into the shell profile,
# which makes NCCL try to use eRDMA even for single-node inter-GPU traffic.
# On L20 boxes this deadlocks periodically (ranks stuck on BROADCAST NumelIn=1
# for hours) — see the 2026-04-22 second hang in ddp_nccl_timeout_troubleshooting.md.
# Force-disable IB here so single-node runs use PCIe P2P + shared memory instead.
# Override (not setdefault) because the DSW profile re-sets these on every shell.
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ.pop("NCCL_IB_HCA", None)
os.environ.pop("NCCL_IB_GID_INDEX", None)
os.environ.pop("NCCL_IB_QPS_PER_CONNECTION", None)
# NCCL_DEBUG=INFO on DSW floods stderr and output.log with MB/s of noise.
# WARN keeps the NCCL hang-time dumps without the fire-hose.
if os.environ.get("NCCL_DEBUG", "").upper() == "INFO":
    os.environ["NCCL_DEBUG"] = "WARN"

# Belt-and-suspenders: keep any direct wandb.init() calls out of NAS. The
# PL WandbLogger already uses `save_dir=${paths.output_dir}` which trainHydra
# reroutes to /dev/shm when staging is enabled, but wandb also respects
# WANDB_DIR for ad-hoc init() calls and its own cache.
os.environ.setdefault("WANDB_DIR", "/dev/shm/wandb_staging")

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig, OmegaConf, open_dict
from tabulate import tabulate

from egomimic.eval.eval import Eval
from egomimic.pl_utils.pl_model import ModelWrapper
from egomimic.rldb.zarr.utils import DataSchematic, set_global_seed
from egomimic.rldb.zarr.zarr_dataset_multi import MultiDataset
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.instantiators import instantiate_callbacks, instantiate_loggers
from egomimic.utils.local_staging import (
    NasMirrorDaemon,
    heal_logs_symlink,
    staging_dir_for,
)
from egomimic.utils.logging_utils import log_hyperparameters
from egomimic.utils.pylogger import RankedLogger
from egomimic.utils.utils import extras, task_wrapper

OmegaConf.register_new_resolver("eval", eval)
log = RankedLogger(__name__, rank_zero_only=True)

# Convert ./logs from "symlink to NAS" to "real tmpfs dir + marker file".
# Must run BEFORE @hydra.main creates any output dir, otherwise hydra will
# have already materialised its run dir through the symlink on NAS.
# Recorded at module import so DDP subprocesses (which re-import this module)
# also pick up the healed NAS target via the marker file.
_LOCAL_LOGS_DIR = os.path.abspath(os.path.join(os.getcwd(), "logs"))
_NAS_LOGS_ROOT = heal_logs_symlink(_LOCAL_LOGS_DIR)


def _build_model_config_tree(cfg: DictConfig) -> DictConfig:
    model_cfg = copy.deepcopy(cfg.model)
    if (
        "robomimic_model" in model_cfg
        and isinstance(model_cfg.robomimic_model, DictConfig)
        and "data_schematic" in model_cfg.robomimic_model
    ):
        model_cfg.robomimic_model.data_schematic = None
    return OmegaConf.create({"model": model_cfg})


def _log_dataset_frame_counts(train_datasets: dict, valid_datasets: dict) -> None:
    rows = []
    for name, ds in train_datasets.items():
        rows.append(("train", name, len(ds)))
    if train_datasets:
        rows.append(
            ("TOTAL", "(train)", sum(len(ds) for ds in train_datasets.values()))
        )
    for name, ds in valid_datasets.items():
        rows.append(("valid", name, len(ds)))
    if valid_datasets:
        rows.append(
            ("TOTAL", "(valid)", sum(len(ds) for ds in valid_datasets.values()))
        )
    table = tabulate(
        rows,
        headers=["Split", "Dataset", "Frames"],
        tablefmt="rounded_outline",
        intfmt=",",
    )
    log.info("Dataset frame counts:\n" + table)


def _propagate_data_schematic_to_datasets(data_schematic, datasets):
    """
    Set the shared data schematic on all top-level datasets.
    """
    split_datasets = datasets
    for dataset_name, dataset in split_datasets.items():
        if not isinstance(dataset, MultiDataset):
            raise ValueError(
                f"{dataset_name} is not a MultiDataset. All top level datasets in data config should be MultiDataset"
            )
        dataset.set_data_schematic(data_schematic)


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

        set_global_seed(cfg.seed)
    else:
        raise ValueError("Seed must be provided in cfg for reproducibility!")

    load_env()
    # log.info(f"Instantiating data schematic <{cfg.data_schematic._target_}>")

    data_schematic: DataSchematic = hydra.utils.instantiate(cfg.data_schematic)

    # Modify dataset configs to include `data_schematic` dynamically at runtime
    train_datasets = {}
    for dataset_name in cfg.data.train_datasets:
        train_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.train_datasets[dataset_name]
        )

    valid_datasets = {}
    for dataset_name in cfg.data.valid_datasets:
        valid_datasets[dataset_name] = hydra.utils.instantiate(
            cfg.data.valid_datasets[dataset_name]
        )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    assert "MultiDataModuleWrapper" in cfg.data._target_, (
        "cfg.data._target_ must be 'MultiDataModuleWrapper'"
    )
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.data, train_datasets=train_datasets, valid_datasets=valid_datasets
    )

    for dataset_name, dataset in datamodule.train_datasets.items():
        log.info(f"Inferring shapes for dataset <{dataset_name}>")
        data_schematic.infer_shapes_from_batch(dataset[0])
        instantiate_copy = copy.deepcopy(cfg.data.train_datasets[dataset_name])
        keymap_cfg = instantiate_copy.resolver.key_map
        km = OmegaConf.to_container(keymap_cfg, resolve=False)  # plain dict

        # this remove annotation and image keys from the keymap
        km["norm_mode"] = True

        instantiate_copy.resolver.key_map = km
        norm_dataset = hydra.utils.instantiate(instantiate_copy)
        # infer_norm_from_dataset: load from precomputed JSON/dir if set, else compute (no disk write).
        data_schematic.infer_norm_from_dataset(
            norm_dataset,
            dataset_name,
            sample_frac=OmegaConf.select(cfg, "norm_stats.sample_frac", default=1.0),
            num_workers=OmegaConf.select(cfg, "norm_stats.num_workers", default=4),
            precomputed_norm_path=OmegaConf.select(
                cfg, "norm_stats.precomputed_norm_path", default=None
            ),
        )
        # Cache norm stats if save_cache_dir is set
        save_cache_dir = OmegaConf.select(
            cfg, "norm_stats.save_cache_dir", default=None
        )
        if save_cache_dir:
            data_schematic.cache_stats(save_cache_dir=save_cache_dir)

    if cfg.reject_outliers:
        # Propagate the shared data schematic to top-level MultiDatasets for bounds checks.
        _propagate_data_schematic_to_datasets(
            data_schematic,
            train_datasets,
        )
    viz_func_dict = {}
    for embodiment_name, embodiment_viz_func in cfg.get("visualization", {}).items():
        viz_func_dict[embodiment_name] = hydra.utils.instantiate(embodiment_viz_func)

    # NOTE: We also pass the data_schematic_dict into the robomimic model's instatiation now that we've initialzied the shapes and norm stats.  In theory, upon loading the PL checkpoint, it will remember this, but let's see.
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = ModelWrapper(
        config_tree=_build_model_config_tree(cfg),
        data_schematic_state=data_schematic.to_state(),
        viz_func=viz_func_dict,
    )

    _log_dataset_frame_counts(train_datasets, valid_datasets)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    # Resolve mode: support both new `mode` key and legacy `train`/`eval` booleans
    if cfg.get("mode") is not None:
        mode = cfg.mode
    elif cfg.get("train", False):
        mode = "train"
    elif cfg.get("eval", False):
        mode = "eval"
    else:
        raise ValueError("Config must specify either `mode` or `train`/`eval` booleans")

    # In eval mode, apply trainer overrides from the eval object and disable logger
    if mode == "eval":
        eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
        log.info(
            "Eval mode: applying trainer overrides from eval config, disabling logger"
        )
        with open_dict(cfg):
            for k, v in eval_obj.override_dict.items():
                cfg.trainer[k] = v
            cfg.trainer.devices = 1
            cfg.trainer.num_nodes = 1
            cfg.trainer.num_sanity_val_steps = 0
            cfg.logger = None

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    plugins = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(
            SLURMEnvironment(requeue_signal=[signal.SIGUSR1, signal.SIGUSR2])
        )
        print("SLURM REQUEUE ENABLED")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if (
        os.environ.get("SLURM_JOB_ID")
        and os.environ.get("SLURM_RESTART_COUNT", "0") != "0"
    ):
        # Must read from nas_output_dir (not trainer.default_root_dir): with
        # staging, default_root_dir is /dev/shm which doesn't have a
        # checkpoints/ subdir — AsyncCopyModelCheckpoint mirrors straight to
        # nas_output_dir/checkpoints.
        resume_root = OmegaConf.select(
            cfg, "paths.nas_output_dir", default=trainer.default_root_dir
        )
        last_ckpt_path = os.path.join(resume_root, "checkpoints", "last.ckpt")
        log.info("Detected SLURM requeue — resuming from 'last.ckpt'")
        cfg.ckpt_path = last_ckpt_path

    os.makedirs(os.path.join(trainer.default_root_dir, "videos"), exist_ok=True)

    if mode == "train":
        if cfg.get("evaluator") is not None:
            eval_obj: Eval = hydra.utils.instantiate(cfg.evaluator)
            eval_obj.trainer = trainer
            eval_obj.model = model.model
            model.evaluator = eval_obj
        log.info("Starting training!")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path"),
            weights_only=False,
        )
    elif mode == "eval":
        eval_obj.trainer = trainer
        eval_obj.model = model.model
        model.evaluator = eval_obj
        # Load checkpoint weights manually so we can reset the epoch counter
        ckpt_path = cfg.get("ckpt_path")
        if ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            log.info(f"Loaded weights from {ckpt_path}")
        log.info("Starting evaluation!")
        trainer.validate(model=model, datamodule=datamodule)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    train_metrics = trainer.callback_metrics

    # if cfg.get("test"):
    #     log.info("Starting testing!")
    #     ckpt_path = trainer.checkpoint_callback.best_model_path
    #     if ckpt_path == "":
    #         log.warning("Best ckpt not found! Using current weights for testing...")
    #         ckpt_path = None
    #     trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    #     log.info(f"Best ckpt path: {ckpt_path}")

    # test_metrics = trainer.callback_metrics

    # merge train and test metrics
    test_metrics = {}  # my stub
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


def _setup_staging(cfg: DictConfig) -> List[NasMirrorDaemon]:
    """Reroute `paths.output_dir` to tmpfs and start NAS mirror daemons.

    Two sources need mirroring to NAS:
      1. ``paths.output_dir`` — staging dir on tmpfs that wandb / vis / video
         callbacks write to. Mirrored from ``/dev/shm/run_staging/<hash>``.
      2. ``hydra:runtime.output_dir`` — Hydra's own run dir, where it writes
         ``trainHydra.log``, ``train_ddp_process_{N}.log``, and ``.hydra/``.
         Previously on NAS via the ``./logs`` symlink; now on tmpfs because
         module-level ``heal_logs_symlink`` replaced the symlink with a real
         dir. The NAS destination comes from the marker file written during
         heal, mapped by substituting the local logs prefix.

    Runs on every rank so all ranks agree on the staged output path, but only
    rank 0 launches the rsync daemons (ranks 1..N-1 don't write to output_dir
    in ways that need mirroring — their checkpoint writes go through
    AsyncCopyModelCheckpoint, and video/vis/logger writes are rank-0-only).

    Eval mode skips staging: eval typically wants videos/metrics to land on
    NAS immediately for inspection, and the 1-GPU run doesn't have a DDP
    collective hot-path to protect.
    """
    if not OmegaConf.select(cfg, "paths.staging_enabled", default=False):
        return []
    if OmegaConf.select(cfg, "mode", default=None) == "eval" or cfg.get("eval", False):
        return []

    # Step 1: recompute paths.nas_output_dir from the healed symlink target.
    # Before the heal, `./logs` -> NAS symlink made hydra:runtime.output_dir
    # physically land on NAS; now it's on tmpfs, so cfg.paths.nas_output_dir
    # (which equals ${hydra:runtime.output_dir}) is also tmpfs — we have to
    # substitute the local-logs prefix with the NAS logs root so mirroring
    # has a real NAS destination.
    hydra_out_dir = cfg.paths.nas_output_dir  # tmpfs path after heal
    if _NAS_LOGS_ROOT and hydra_out_dir.startswith(_LOCAL_LOGS_DIR):
        nas_dir = _NAS_LOGS_ROOT.rstrip("/") + hydra_out_dir[len(_LOCAL_LOGS_DIR):]
        with open_dict(cfg):
            cfg.paths.nas_output_dir = nas_dir
    else:
        nas_dir = hydra_out_dir  # no heal happened; mirror is no-op (local == nas)

    # Step 2: keep the existing staging-dir indirection. wandb, vis, video
    # callbacks write here; daemon rsyncs to NAS.
    stage_dir = staging_dir_for(nas_dir)
    with open_dict(cfg):
        cfg.paths.output_dir = stage_dir

    if os.environ.get("RANK", "0") != "0":
        return []

    interval_s = OmegaConf.select(cfg, "paths.staging_mirror_interval_s", default=30)
    daemons: List[NasMirrorDaemon] = []

    # Source A: staging dir -> NAS. Only meaningful if NAS target differs
    # (i.e. heal worked or nas_output_dir was configured elsewhere).
    if stage_dir != nas_dir:
        daemons.append(
            NasMirrorDaemon(local=stage_dir, nas=nas_dir, interval_s=interval_s)
        )

    # Source B: hydra runtime dir -> NAS. Covers trainHydra.log, .hydra/,
    # submitit ddp process logs. Skip if heal didn't happen (no NAS target).
    if _NAS_LOGS_ROOT and hydra_out_dir != nas_dir:
        daemons.append(
            NasMirrorDaemon(local=hydra_out_dir, nas=nas_dir, interval_s=interval_s)
        )

    for d in daemons:
        d.start()
    return daemons


@hydra.main(
    version_base="1.3",
    config_path="./hydra_configs",
    config_name="train_zarr_cartesian.yaml",
)
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # Must run before `extras(cfg)` — `extras` writes config_tree.log /
    # tags.log to `${paths.output_dir}`, which we want on tmpfs.
    mirror_daemons = _setup_staging(cfg)

    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    print(OmegaConf.to_yaml(cfg))

    # cfg = OmegaConf.resolve(cfg)

    # train the model
    try:
        metric_dict, _ = train(cfg)
    finally:
        # Training is done (success, failure, or KeyboardInterrupt). Run one
        # final /dev/shm -> NAS sync for every source (staging dir + hydra
        # runtime dir) so artefacts generated after the last periodic tick
        # (final last.ckpt, last wandb step, end-of-run vis, trainHydra.log
        # tail, .hydra/overrides.yaml) land on NAS before the process exits.
        # Each sync is bounded by final_timeout_s; if NAS is hung we log a
        # manual `rsync` command to recover.
        if mirror_daemons:
            log.info(
                "Training finished; running final /dev/shm -> NAS sync for %d sources.",
                len(mirror_daemons),
            )
            for d in mirror_daemons:
                d.stop(final_sync=True)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(
    #     metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    # )

    # # return optimized metric
    # return metric_value


if __name__ == "__main__":
    main()
