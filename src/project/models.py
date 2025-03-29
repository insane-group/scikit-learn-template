from typing import Dict, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def create_pipeline(
    model_type: str = "linear",
    scale_features: bool = True,
    seed: int = 1234,
    **model_params: Dict[str, Any],
) -> Pipeline:
    """
    Create a scikit-learn pipeline with optional feature scaling and a specified model.

    Args:
        model_type (str): A string specifying the type of model to include in the pipeline.
            Supported options are "linear", "lasso", "random_forest", and "svm".
        scale_features (bool): A boolean indicating whether to include feature scaling in the pipeline.
        seed (int): An integer for setting the random state for models that require it.
        **model_params (Dict[str, Any]): Additional parameters to pass to the model constructor.

    Returns:
        Pipeline: A scikit-learn Pipeline object with the specified configuration.
    """
    steps = []

    if scale_features:
        steps.append(("scaler", StandardScaler()))

    if model_type == "linear":
        steps.append(("model", LinearRegression(**model_params)))
    elif model_type == "lasso":
        steps.append(("model", Lasso(random_state=seed, **model_params)))
    elif model_type == "random_forest":
        steps.append(
            ("model", RandomForestRegressor(random_state=seed, **model_params))
        )
    elif model_type == "svm":
        steps.append(("model", SVR(**model_params)))
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    return Pipeline(steps=steps)
