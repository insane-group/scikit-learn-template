import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataModule:
    """Data module for loading and processing datasets."""

    def __init__(
        self,
        train_dataset_url: str,
        test_dataset_url: str,
        target_variable: str,
        include_features: list[str] | None = None,
        exclude_features: list[str] | None = None,
        drop_na: bool = True,
        test_size: float = 0.2,
    ):
        """
        Initialize the data module.

        Args:
            train_dataset_url (str): URL to the training dataset.
            test_dataset_url (str): URL to the test dataset.
            target_variable (str): Name of the target variable column.
            include_features (list[str] | None, optional): List of features to include
                (if specified, overrides exclude_features). Defaults to None.
            exclude_features (list[str] | None, optional): List of features to exclude
                (if specified, will be removed from the dataset). Defaults to None.
            drop_na (bool, optional): Whether to drop NA values. Defaults to True.
            test_size (float, optional): Size of the test set. Defaults to 0.2.
        """
        self.train_dataset_url = train_dataset_url
        self.test_dataset_url = test_dataset_url
        self.target_variable = target_variable
        self.drop_na = drop_na
        self.test_size = test_size
        self.include_features = include_features
        self.exclude_features = exclude_features

        logger.info("Loading training data from %s", self.train_dataset_url)
        self.df_train = pd.read_csv(self.train_dataset_url)

        logger.info("Loading test data from %s", self.test_dataset_url)
        self.df_test = pd.read_csv(self.test_dataset_url)

        if self.drop_na:
            logger.info("Dropping rows with missing values")
            self.df_train.dropna(inplace=True)

        # Define feature sets
        all_features = [c for c in self.df_train.columns if c != self.target_variable]
        logger.info("Found %d features in the dataset", len(all_features))

        if self.include_features is not None:
            self.features_selected = [
                f for f in all_features if f in self.include_features
            ]

        if self.exclude_features is not None:
            self.features_selected = [
                f for f in all_features if f not in self.exclude_features
            ]

        if "id" not in self.features_selected:
            logger.warning(
                "The 'id' column is not included in the selected features. "
                "It will be added automatically."
            )
            self.features_selected = ["id", *self.features_selected]

        logger.info(
            "Selected %d features from the dataset: %s",
            len(self.features_selected),
            ", ".join(self.features_selected),
        )

    def get_split(
        self, train: bool = True
    ) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
        """
        Get the training or testing data.

        Args:
            train (bool): If True, return training data; otherwise, return testing data.
                Defaults to True.

        Returns:
            tuple[pd.Series, pd.DataFrame, pd.Series]: id (pd.Series), features (pd.DataFrame), and target variable (pd.Series).
        """
        logger.info("Splitting data into train and test sets")

        X = self.df_train[self.features_selected]
        y = self.df_train[self.target_variable]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size
        )

        if train:
            return (
                X_train["id"],
                X_train.drop(columns=["id"]),
                y_train,
            )
        else:
            return (
                X_test["id"],
                X_test.drop(columns=["id"]),
                y_test,
            )

    def get_train_data(self) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
        """
        Get the training data.

        Returns:
            tuple[pd.Series, pd.DataFrame, pd.Series]:
                id (pd.Series): Unique identifier for each sample.
                features (pd.DataFrame): Features for each sample.
                target (pd.Series): Target variable for each sample.
        """
        logger.info("Loading training data")

        X: pd.DataFrame = self.df_train[self.features_selected]
        y: pd.Series = self.df_train[self.target_variable]

        return (
            X["id"],
            X.drop(columns=["id"]),
            y,
        )

    def get_test_data(self) -> tuple[pd.Series, pd.DataFrame]:
        """
        Get the test data.

        Returns:
            tuple[pd.Series, pd.DataFrame]:
                id (pd.Series): Unique identifier for each sample.
                features (pd.DataFrame): Features for each sample.
        """
        logger.info("Loading test data")

        X = self.df_test[self.features_selected]

        return (
            X["id"],
            X.drop(columns=["id"]),
        )
