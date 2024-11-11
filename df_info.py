import pandas as pd
import numpy as np
import scipy.stats as stats

from db_utils import filter_columns
from visualisations import Plotter


class DataFrameInfo:
    """
    A class to extract and display information from a Pandas DataFrame
    for exploratory data analysis (EDA).

    This class provides methods to describe columns, extract statistical
    values, count distinct values, and detect outliers, among other
    functionalities, to aid in the EDA process.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialise the DataFrameInfo with a DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame to analyse.
        """
        self.dataframe = dataframe

    def describe_columns(
        self,
        include_columns: list = None,
        exclude_columns: list = None,
        stats_to_return: list = None,
    ) -> pd.DataFrame:
        """
        Describe selected columns and return specified statistics.

        Parameters:
            include_columns (list, optional): Columns to include in the
            description. If None, all columns are included.
            exclude_columns (list, optional): Columns to exclude from the
            description. Only used if include_columns is None.
            stats_to_return (list, optional): Statistics to return
            (e.g., 'min', 'max'). If None, all statistics are returned.

        Returns:
            pd.DataFrame: Description of selected columns with specified
            statistics.
        """
        if include_columns is None:
            # Include all columns if none are specified, then exclude specified ones
            print("Including all columns")
            include_columns = self.dataframe.columns.difference(exclude_columns or [])

        description = self.dataframe[include_columns].describe(include="all")

        if stats_to_return is not None:
            # Filter the description to include only specified statistics
            description = description.loc[stats_to_return]

        return description

    def extract_stats(self, exclude_columns: list = None) -> pd.DataFrame:
        """
        Extract statistical values: median, standard deviation, and mean
        from all numeric columns in the DataFrame, excluding specified
        columns.

        Parameters:
            exclude_columns (list, optional): List of columns to exclude
            from the statistics.

        Returns:
            pd.DataFrame: A DataFrame containing the mean, median, and
            standard deviation of the numeric columns.
        """
        # Use filter_columns to include only numeric columns, excluding specified ones
        numeric_cols = filter_columns(
            self.dataframe, np.number, exclude_columns=exclude_columns
        )

        # Calculate mean, median, and standard deviation for numeric columns
        statistics = {
            "mean": self.dataframe[numeric_cols].mean(),
            "median": self.dataframe[numeric_cols].median(),
            "std_dev": self.dataframe[numeric_cols].std(),
        }
        return pd.DataFrame(statistics)

    def count_distinct_values(
        self, dataframe: pd.DataFrame, exclude_columns: list = None
    ) -> None:
        """
        Count and print distinct values in categorical columns of the given DataFrame,
        excluding specified columns.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame to analyze.
            exclude_columns (list, optional): A list of columns to exclude
            from the distinct count.

        Returns:
            None
        """
        if exclude_columns is None:
            exclude_columns = []

        # Use filter_columns to get categorical columns, excluding specified ones
        categorical_columns = filter_columns(
            dataframe, "category", exclude_columns=exclude_columns
        )

        # Iterate over categorical columns and print distinct counts
        for column in categorical_columns:
            distinct_count = dataframe[column].nunique()
            print(f"Column '{column}' has {distinct_count} distinct values.")

    def print_shape(self) -> None:
        """
        Print out the shape of the DataFrame.

        This method outputs the number of rows and columns in the DataFrame,
        providing a quick overview of its size.
        """
        print(f"\nDataFrame shape: {self.dataframe.shape}")

    def count_null_values(self) -> pd.DataFrame:
        """
        Generate a count and percentage of NULL values in each column.

        Returns:
            pd.DataFrame: A DataFrame showing the count and percentage of
            NULL values for each column.
        """
        null_count = self.dataframe.isnull().sum()
        null_percentage = (null_count / len(self.dataframe)) * 100

        null_info = pd.DataFrame(
            {"null_count": null_count, "null_percentage": null_percentage}
        )

        return null_info

    def detect_outliers_zscore(
        self, column: str, threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Detect outliers in a column using the Z-score method.

        Parameters:
            column (str): The column to detect outliers in.
            threshold (float, optional): The Z-score threshold to identify
            outliers. Default is 3.0.

        Returns:
            pd.DataFrame: A DataFrame containing the outliers identified
            in the specified column.
        """
        # Calculate Z-scores for the specified column
        z_scores = stats.zscore(self.dataframe[column])
        # Identify outliers based on the Z-score threshold
        outliers = self.dataframe[(z_scores > threshold) | (z_scores < -threshold)]

        return outliers

    def detect_outliers_iqr(self, column: str) -> pd.DataFrame:
        """
        Detect outliers in a column using the IQR method.

        Parameters:
            column (str): The column to detect outliers in.

        Returns:
            pd.DataFrame: A DataFrame containing the outliers identified
            in the specified column.
        """
        # Calculate the first and third quartiles
        Q1 = self.dataframe[column].quantile(0.25)
        Q3 = self.dataframe[column].quantile(0.75)
        # Calculate the interquartile range (IQR)
        IQR = Q3 - Q1
        # Determine the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Identify outliers based on the bounds
        outliers = self.dataframe[
            (self.dataframe[column] < lower_bound)
            | (self.dataframe[column] > upper_bound)
        ]
        return outliers

    def calculate_failure_rate(self) -> None:
        """
        Calculate the total number and percentage of failures.

        This method calculates the failure rate in the manufacturing process and
        returns the calculated values for visualisation.
        """
        # Calculate total number of processes and failures
        total_processes = len(self.dataframe)
        total_failures = self.dataframe["Machine failure"].sum()

        # Calculate failure percentage
        failure_percentage = (total_failures / total_processes) * 100

        # Print failure statistics
        print(f"\nTotal Processes: {total_processes}")
        print(f"Total Failures: {total_failures}")
        print(f"Failure Percentage: {failure_percentage:.2f}%")

        # Prepare data for pie chart
        labels = ["Failures", "Non-failures"]
        sizes = [total_failures, total_processes - total_failures]
        title = "Failure Rate in the Manufacturing Process"

        plotter = Plotter(self.dataframe)

        # Call plot_pie_chart method
        plotter.plot_pie_chart(title, labels, sizes)

    def failures_by_product_quality(self) -> None:
        """
        Analyse and visualise failures based on product quality types.

        This method calculates the number and percentage of failures for each
        product quality type, prints the results to the terminal, and calls the
        plotting method for visualisation.
        """
        # Group by product quality and count the number of failures
        failures_by_quality = self.dataframe.groupby("Type")["Machine failure"].sum()
        total_by_quality = self.dataframe.groupby("Type").size()

        # Calculate the percentage of failures for each product quality type
        failure_percent_by_quality = round(
            (failures_by_quality / total_by_quality) * 100, 2
        )

        # Print the total number of products, failures, and failure percentages
        print("\nTotal by Product Quality:\n", total_by_quality.to_string())
        print("\nFailures by Product Quality:\n", failures_by_quality.to_string())
        print(
            "\nFailure Percentage by Product Quality:\n",
            failure_percent_by_quality.to_string(),
        )

        plotter = Plotter(self.dataframe)

        # Call plot_bar_chart method for visualisation
        plotter.plot_product_quality_chart(
            failures_by_quality,
            failure_percent_by_quality
        )
