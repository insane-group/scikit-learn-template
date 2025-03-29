import logging
from pathlib import Path

import hydra
import joblib
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from project.data.module import DataModule
from project.utils import evaluate_model, seed_everything

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_DIR = BASE_DIR / "configs"


def test(cfg: DictConfig) -> None:
    """
    Make predictions on the test set using a trained model.

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

    ids_test, X_test, y_test = data_module.get_split(train=False)

    # Load the saved model
    if not Path(cfg.checkpoint).is_file():
        logger.error(f"Model file not found at {cfg.checkpoint}")
        logger.info("Please run the training script first to generate a model.")
        return

    logger.info(f"Loading model from {cfg.checkpoint}")
    with open(cfg.checkpoint, "rb") as f:
        pipeline = joblib.load(f)

    # Make predictions on the test set
    logger.info("Making predictions on the test set")
    y_pred = pipeline.predict(X_test)

    # Create predictions DataFrame
    predictions_df = pd.DataFrame({"id": ids_test, cfg.data.target_variable: y_pred})

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

    # Instantiate metrics
    logger.info("Instantiating metrics")

    # Evaluate model on validation set
    metrics = evaluate_model(cfg.get("metrics"), y_test, y_pred)
    logger.info("Test set metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.4f}")


@hydra.main(
    version_base="1.3",
    config_path=CONFIG_DIR.as_posix(),
    config_name="test.yaml",
)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for testing.

    Args:
        cfg: Configuration composed by Hydra
    """
    test(cfg)


if __name__ == "__main__":
    main()
