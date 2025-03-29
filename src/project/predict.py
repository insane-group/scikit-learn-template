import logging
from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from project.data.module import DataModule
from project.utils import seed_everything

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "configs"


def predict(cfg: DictConfig) -> None:
    """
    Make predictions on unlabeled data using a trained model.

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

    ids, X = data_module.get_test_data()

    # Load the saved model
    if not Path(cfg.checkpoint).is_file():
        logger.error(f"Model file not found at {cfg.checkpoint}")
        logger.info("Please run the training script first to generate a model.")
        return

    logger.info(f"Loading model from {cfg.checkpoint}")
    with open(cfg.checkpoint, "rb") as f:
        pipeline = joblib.load(f)

    # Make predictions on the predict set
    logger.info("Making predictions on the predict set")
    y_pred = pipeline.predict(X)

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({"id": ids, cfg.data.target_variable: y_pred})

    # Sort predictions by target variable (descending) as in the original notebook
    predictions_df = predictions_df.sort_values(
        cfg.data.target_variable, ascending=False
    )

    # Save predictions if specified
    if cfg.get("predictions_file"):
        predictions_file = output_dir / cfg.get("predictions_file")
        logger.info(f"Saving predictions to {predictions_file}")
        predictions_df.to_csv(predictions_file, index=False)

    # Display top predictions
    logger.info(f"Top 10 predictions:\n{predictions_df.head(10)}")


@hydra.main(
    version_base="1.3",
    config_path=CONFIG_DIR.as_posix(),
    config_name="predict.yaml",
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for predicting.

    Args:
        cfg: Configuration composed by Hydra
    """
    predict(cfg)


if __name__ == "__main__":
    main()
