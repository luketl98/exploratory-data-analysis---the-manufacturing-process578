import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.impute import KNNImputer


class DataTransform:
    """
    A utility class for transforming data within a DataFrame.

    This class provides methods to convert columns in the DataFrame into different
    data types, such as categorical or boolean.

    Attributes:
        dataframe (pd.DataFrame): The DataFrame containing the data to be transformed.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialise the DataTransform class with a DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame to be transformed.
        """
        self.dataframe = dataframe

    def convert_to_categorical(self, column_name: str) -> None:
        """
        Convert the specified column to a categorical data type.

        Parameters:
            column_name (str): The name of the column to convert to categorical.

        Returns:
            None

        Modifies:
            Converts the specified column to a categorical data type,
            allowing for memory efficiency and potential optimisation
            for certain types of analyses.

        Example:
            transformer.convert_to_categorical('product_type')
        """
        # Converting the specified column to categorical type
        self.dataframe[column_name] = self.dataframe[column_name].astype("category")

    def convert_to_boolean(self, column_name: str) -> None:
        """
        Convert the specified column to a boolean data type.

        Parameters:
            column_name (str): The name of the column to convert to boolean.

        Returns:
            None

        Modifies:
            Converts the specified column to a boolean data type, useful when working
            with binary indicators.

        Example:
            transformer.convert_to_boolean('is_active')
        """
        self.dataframe[column_name] = self.dataframe[column_name].astype(bool)


class DataFrameTransform:
    """
    Class to perform EDA transformations on the DataFrame.

    This class provides methods for imputing missing values, applying
    transformations, and managing DataFrame columns to facilitate
    exploratory data analysis.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialise the DataFrameTransform with a DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame to transform.
        """
        self.dataframe = dataframe

    def knn_impute(
        self, target_column: str, correlated_columns: list, n_neighbors: int = 5
    ) -> None:
        """
        Apply KNN imputation to the target column using specified correlated
        columns.

        Parameters:
            target_column (str): The name of the column to impute.
            correlated_columns (list): List of correlated columns to use for
                                       KNN imputation.
            n_neighbors (int, optional): The number of neighbors to consider
                                         for imputation. Default is 5.
        """
        # Select the target column along with the correlated columns
        columns_to_use = correlated_columns + [target_column]
        knn_data = self.dataframe[columns_to_use]

        # Perform KNN imputation on these columns
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = knn_imputer.fit_transform(knn_data)

        # Replace target column with imputed values & round to 1 decimal point
        self.dataframe[target_column] = np.round(imputed_data[:, -1], 1)

        print(f"\nKNN imputation applied to {target_column} using {correlated_columns}")

    def impute_missing_values(
        self, strategies: dict = None, knn_columns: dict = None
    ) -> None:
        """
        Impute missing numeric values in the DataFrame based on the provided
        strategy.

        Parameters:
            strategies (dict, optional): A dictionary where the key is the
            column name, and the value is the imputation strategy ('mean',
            'median', 'knn').
            knn_columns (dict, optional): Dictionary where the key is the
            column to impute, and the value is the list of correlated columns
            to use for KNN imputation.
        """
        # Select only numeric columns
        numeric_columns = self.dataframe.select_dtypes(include=[np.number])

        # Find numeric columns with null values
        numeric_columns_with_nulls = numeric_columns.loc[
            :, numeric_columns.isnull().any()
        ]

        # Default to median strategy if none is provided
        if strategies is None:
            strategies = {col: "median" for col in numeric_columns_with_nulls}

        # Iterate through the provided strategies
        for column, strategy in strategies.items():
            if column in numeric_columns_with_nulls:
                if strategy == "mean":
                    # Impute using the mean
                    self.dataframe[column].fillna(
                        round(self.dataframe[column].mean(), 1), inplace=True
                    )
                elif strategy == "median":
                    # Impute using the median
                    self.dataframe[column].fillna(
                        round(self.dataframe[column].median(), 1), inplace=True
                    )
                elif strategy == "knn" and knn_columns and column in knn_columns:
                    # Apply KNN imputation for the specified column
                    self.knn_impute(column, knn_columns[column])

    def apply_yeo_johnson(self, column: str) -> None:
        """
        Apply Yeo-Johnson transformation to the specified column and replace
        the original column with the transformed values.

        Parameters:
            column (str): The name of the column to transform.
        """
        # Perform Yeo-Johnson transformation
        transformed_column, _ = stats.yeojohnson(self.dataframe[column])

        # Update the DataFrame with the transformed values
        self.dataframe[column] = transformed_column

        print(f"Yeo-Johnson transformation applied to {column}")

    def preview_transformations(self, column: str) -> None:
        """
        Apply Log, Box-Cox, and Yeo-Johnson transformations to the specified
        column. Print skewness post-transformation and plot the distributions.

        Parameters:
            column (str): The name of the column to transform and preview.
        """
        # Handle log transformation (requires values > 0)
        log_transformed = np.log1p(self.dataframe[column])
        log_skew = log_transformed.skew()
        print(f"\nLog Transform Skewness of {column}: {log_skew}")

        # Handle Box-Cox transformation (requires positive values)
        # Add a small constant to ensure all values are positive
        boxcox_transformed, _ = stats.boxcox(self.dataframe[column] + 1)
        boxcox_skew = stats.skew(boxcox_transformed)
        print(f"Box-Cox Transform Skewness of {column}: {boxcox_skew}")

        # Handle Yeo-Johnson transformation
        # ^(works with both positive and negative)
        yeo_transformed, _ = stats.yeojohnson(self.dataframe[column])
        yeo_skew = stats.skew(yeo_transformed)
        print(f"Yeo-Johnson Transform Skewness of {column}: {yeo_skew}")

        # Plot all transformations
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.hist(log_transformed, bins=30, color="blue")
        plt.title(f"Log Transform (Skew: {log_skew:.2f})")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 2)
        plt.hist(boxcox_transformed, bins=30, color="green")
        plt.title(f"Box-Cox Transform (Skew: {boxcox_skew:.2f})")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 3)
        plt.hist(yeo_transformed, bins=30, color="red")
        plt.title(f"Yeo-Johnson Transform (Skew: {yeo_skew:.2f})")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        plt.suptitle(f"Preview transformations of {column}")
        plt.show()

    def drop_columns(self, columns_to_drop: list) -> pd.DataFrame:
        """
        Drop specified columns from the DataFrame.

        Parameters:
            columns_to_drop (list): A list of column names to drop.

        Returns:
            pd.DataFrame: The modified DataFrame with the columns dropped.
        """
        print(f"\nColumns to be dropped: {columns_to_drop}")

        # Ensure only valid columns are dropped
        valid_columns = [
            col for col in columns_to_drop if col in self.dataframe.columns
        ]

        if not valid_columns:
            print("\nNo valid columns to drop.")
            return self.dataframe

        # Drop the specified columns
        self.dataframe.drop(columns=valid_columns, inplace=True)

        print(f"\nSuccessfully dropped: {valid_columns}\n")
        return self.dataframe
