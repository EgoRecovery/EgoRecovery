import os

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

HYDRA_CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "hydra_configs"
)


def load_config(config_name: str, overrides: list[str] | None = None) -> DictConfig:
    """
    Load a Hydra config by name, resolving the full defaults chain
    from the egomimic/hydra_configs directory.

    For config group paths (e.g. "data/cotrain_pi_lang"), the returned
    DictConfig contains the group contents directly (not nested under
    the group key).

    Args:
        config_name: Name of the config file (without .yaml extension).
            Can also be a config group path like "data/cotrain_pi_lang".
        overrides: Optional list of Hydra overrides (e.g. ["+trainer=debug"]).

    Returns:
        Fully composed DictConfig.
    """
    gh = GlobalHydra.instance()
    was_initialized = gh.is_initialized()
    if was_initialized:
        gh.clear()
    try:
        with initialize_config_dir(config_dir=HYDRA_CONFIG_DIR, version_base="1.3"):
            cfg = compose(config_name=config_name, overrides=overrides or [])
        # Config groups (e.g. "data/foo") get nested under the group key.
        # Unwrap if the config has exactly one key matching the group name.
        if "/" in config_name:
            group_key = config_name.split("/")[0]
            if list(cfg.keys()) == [group_key]:
                cfg = cfg[group_key]
        return cfg
    finally:
        if was_initialized:
            gh.clear()
