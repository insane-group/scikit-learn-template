import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from models import create_pipeline


@pytest.fixture
def sample_data():
    """Fixture that provides sample data for functional testing."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([1, 2, 3])
    return X, y


def test_linear_model_without_scaling():
    """Test creating a pipeline with a linear model without scaling."""
    pipeline = create_pipeline(model_type="linear", scale_features=False)

    # Check pipeline structure
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0][0] == "model"
    assert isinstance(pipeline.steps[0][1], LinearRegression)


def test_linear_model_with_scaling():
    """Test creating a pipeline with a linear model with scaling."""
    pipeline = create_pipeline(model_type="linear", scale_features=True)

    # Check pipeline structure
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "scaler"
    assert isinstance(pipeline.steps[0][1], StandardScaler)
    assert pipeline.steps[1][0] == "model"
    assert isinstance(pipeline.steps[1][1], LinearRegression)


def test_lasso_model():
    """Test creating a pipeline with a lasso model."""
    pipeline = create_pipeline(model_type="lasso", scale_features=True, alpha=0.1)

    # Check pipeline structure
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[1][0] == "model"
    assert isinstance(pipeline.steps[1][1], Lasso)

    # Check model parameters
    lasso_model = pipeline.steps[1][1]
    assert lasso_model.alpha == 0.1
    assert lasso_model.random_state == 1234


def test_random_forest_model():
    """Test creating a pipeline with a random forest model."""
    pipeline = create_pipeline(
        model_type="random_forest", scale_features=True, n_estimators=100, max_depth=10
    )

    # Check pipeline structure
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[1][0] == "model"
    assert isinstance(pipeline.steps[1][1], RandomForestRegressor)

    # Check model parameters
    rf_model = pipeline.steps[1][1]
    assert rf_model.n_estimators == 100
    assert rf_model.max_depth == 10
    assert rf_model.random_state == 1234


def test_svm_model():
    """Test creating a pipeline with an SVR model."""
    pipeline = create_pipeline(
        model_type="svm", scale_features=True, C=1.0, kernel="rbf"
    )

    # Check pipeline structure
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert pipeline.steps[1][0] == "model"
    assert isinstance(pipeline.steps[1][1], SVR)

    # Check model parameters
    svm_model = pipeline.steps[1][1]
    assert svm_model.C == 1.0
    assert svm_model.kernel == "rbf"


def test_custom_seed():
    """Test that the seed is correctly passed to models."""
    custom_seed = 42
    pipeline = create_pipeline(model_type="lasso", seed=custom_seed)

    # Check that the random state is set correctly
    lasso_model = pipeline.steps[1][1]
    assert lasso_model.random_state == custom_seed


def test_invalid_model_type():
    """Test that an invalid model type raises a ValueError with correct message."""
    with pytest.raises(ValueError, match="Invalid model type: invalid_model"):
        create_pipeline(model_type="invalid_model")


def test_functional_pipeline(sample_data):
    """Test that the pipeline actually works with some data."""
    X, y = sample_data

    pipeline = create_pipeline(model_type="linear")
    pipeline.fit(X, y)

    # Check that prediction works
    predictions = pipeline.predict(X)
    assert len(predictions) == len(y)


@pytest.mark.parametrize(
    "model_type,model_class",
    [
        ("linear", LinearRegression),
        ("lasso", Lasso),
        ("random_forest", RandomForestRegressor),
        ("svm", SVR),
    ],
)
def test_all_model_types(model_type, model_class):
    """Test all supported model types using parametrization."""
    pipeline = create_pipeline(model_type=model_type)

    # Get the model from the pipeline
    model = pipeline.steps[-1][1]

    # Check model type
    assert isinstance(model, model_class)


@pytest.mark.parametrize("scale_features", [True, False])
def test_scaling_option(scale_features):
    """Test that scaling is correctly applied based on the scale_features parameter."""
    pipeline = create_pipeline(model_type="linear", scale_features=scale_features)

    # Check if scaling is included in the pipeline
    if scale_features:
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "scaler"
        assert isinstance(pipeline.steps[0][1], StandardScaler)
    else:
        assert len(pipeline.steps) == 1
        assert pipeline.steps[0][0] == "model"
