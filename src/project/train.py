import logging
from datetime import datetime
from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from project.data.module import DataModule
from project.utils import format_cv_results, save_results, seed_everything

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "configs"


def train(cfg: DictConfig) -> None:
    """
    Train models using the configuration provided by Hydra.

    Args:
        cfg: Configuration composed by Hydra
    """
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)

    logger.info(f"Loaded configuration: \n{OmegaConf.to_yaml(cfg)}")

    # Set random seed for reproducibility
    if cfg.get("seed"):
        logger.info(f"Setting random seed to {cfg.seed}")
        seed_everything(cfg.seed)

    # Instantiate data module
    logger.info(f"Instantiating data module <{cfg.data._target_}>")
    data_module: DataModule = hydra.utils.instantiate(cfg.data)

    _, X, y = data_module.get_split(train=True)

    # Create model pipeline
    logger.info(f"Creating model pipeline <{cfg.model._target_}>")
    pipeline = hydra.utils.instantiate(cfg.model)

    # Perform cross-validation
    if cfg.get("cross_validate"):
        logger.info("Performing k-fold cross-validation")
        cv_results = hydra.utils.instantiate(
            cfg.cross_validate,
            estimator=pipeline,
            X=X,
            y=y,
        )

    # Format and display cross-validation results
    formatted_results = format_cv_results(cv_results)
    results_df = pd.DataFrame(formatted_results, index=[cfg.model.model_type])
    logger.info(f"Cross-validation results:\n{results_df}")

    # Save results
    save_results(results_df, output_dir / "cv_results.csv")

    # Train final model on full dataset
    logger.info("Training final model on full dataset")
    pipeline = hydra.utils.instantiate(cfg.model)
    pipeline.fit(X, y)

    # Save the trained model
    if cfg.get("checkpoint_dir"):
        logger.info("Saving trained model")

        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H-%M-%S")
        filename = f"model-{date_str}_{time_str}.joblib"

        model_path = Path(cfg.checkpoint_dir) / filename
        with open(model_path, "wb") as f:
            joblib.dump(pipeline, f)
        logger.info(f"Model saved to {model_path}")


@hydra.main(
    version_base="1.3",
    config_path=CONFIG_DIR.as_posix(),
    config_name="train.yaml",
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training.

    Args:
        cfg: Configuration composed by Hydra
    """
    train(cfg)


if __name__ == "__main__":
    main()
