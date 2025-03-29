import logging
import os
import random
from typing import Dict, Union
from pathlib import Path
import numpy as np
from omegaconf import DictConfig
import pandas as pd
import hydra

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def evaluate_model(
    cfg: DictConfig,
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """
    Evaluate model predictions using multiple metrics.

    Args:
        cfg (DictConfig): A dictionary with metric names as keys and their parameters as values
        y_true (Union[np.ndarray, pd.Series]): True target values, either as a numpy array or pandas Series
        y_pred (Union[np.ndarray, pd.Series]): Predicted target values, either as a numpy array or pandas Series

    Returns:
        Dict[str, float]: A dictionary with metric names as keys and their computed values as values
    """
    metrics = {}
    for name, params in cfg.items():
        metrics[name] = hydra.utils.instantiate(params, y_true, y_pred)

    return metrics


def format_cv_results(cv_results: Dict[str, np.ndarray]) -> Dict[str, str]:
    """
    Format cross-validation results for display.

    Args:
        cv_results (Dict[str, np.ndarray]): Dictionary of cross-validation results

    Returns:
        Dict[str, str]: Dictionary of formatted metric strings
    """
    # Extract and store mean ± std in a dictionary
    # Iterate over the keys in the cv_results dictionary
    # For each key starting with 'test_', extract the mean and std
    # and store them in a new dictionary
    results = {
        metric.replace(
            "test_", ""
        ): f"{np.mean(cv_results[key]):.3f} ± {np.std(cv_results[key]):.3f}"
        for key, metric in zip(cv_results.keys(), cv_results.keys())
        if key.startswith("test_")
    }

    return results


def save_results(results: pd.DataFrame, filename: Path) -> None:
    """
    Save results to a CSV file.

    Args:
        results (pd.DataFrame): DataFrame of results
        filename (Path): Output filename

    Returns:
        None
    """
    results.to_csv(filename, index=True)
    logger.info(f"Results saved to {filename}")
