import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import yaml
from scipy import stats
from sklearn.impute import KNNImputer
from statsmodels.graphics.gofplots import qqplot
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from tabulate import tabulate


# TODO: 1. Separate classes into different files
# TODO: 2. In the Further Analyis Plotter methods, seperate the calculation
#          logic from the plotting logic.

# Function to load database credentials from a YAML file
def load_db_credentials(file_path: str = "credentials.yaml") -> dict:
    with open(file_path, "r") as file:
        credentials = yaml.safe_load(file)

    return credentials


# --- Utility Methods -----


def filter_columns(
    dataframe: pd.DataFrame, dtype: str | type, exclude_columns: list = None
) -> list:
    """
    Helper method to select DataFrame columns by data type and
    filter out specified columns.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame containing the data.
        dtype (str or type): The data type to filter
                             columns by (e.g., 'number', 'object').
        exclude_columns (list, optional): A list of column names to
                                          exclude from the selection.

    Returns:
        list: A list of column names that match the specified data type,
              excluding those in `exclude_columns`.
    """
    if exclude_columns is None:
        exclude_columns = []
    return [
        col
        for col in dataframe.select_dtypes(include=[dtype]).columns
        if col not in exclude_columns
    ]


def save_data_to_csv(dataframe: pd.DataFrame, filename: str) -> None:
    """
    Saves a DataFrame to a CSV file.

    Parameters:
        dataframe (pd.DataFrame): The DataFrame to save.
        filename (str): The filename to save the CSV file.

    Returns:
        None

    Raises:
        IOError: If an I/O error occurs while saving the file.
        Exception: If an unexpected error occurs.
    """
    try:
        dataframe.to_csv(filename, index=False)
        print(f"\nData successfully saved to '{filename}'.")
    except IOError as e:
        print(f"An error occurred while saving to CSV: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Class to handle database connections and operations
class RDSDatabaseConnector:
    """
    A class to handle database connections and operations for an Amazon RDS instance.

    This class is responsible for connecting to a relational database using SQLAlchemy
    and fetching data using provided SQL queries.

    Attributes:
        engine (sqlalchemy.engine.Engine): The SQLAlchemy engine used to connect to
                                           the RDS database.

    Example Usage:
        credentials = {
            "RDS_USER": "your_username",
            "RDS_PASSWORD": "your_password",
            "RDS_HOST": "your_host",
            "RDS_PORT": "1234",
            "RDS_DATABASE": "your_database"
        }
        connector = RDSDatabaseConnector(credentials)
        df = connector.fetch_data("SELECT * FROM table_name")
        connector.close_connection()
    """

    def __init__(self, credentials: dict[str, str]) -> None:
        """
        Initialise the RDSDatabaseConnector with database credentials.

        Parameters:
            credentials (dict): A dictionary containing the database
                                connection credentials.
                - 'RDS_USER': Username for the RDS database.
                - 'RDS_PASSWORD': Password for the RDS database.
                - 'RDS_HOST': Host address for the RDS database.
                - 'RDS_PORT': Port number for the RDS database.
                - 'RDS_DATABASE': The database name to connect to.

        Returns:
            None
        """

        # Create the SQLAlchemy engine to connect to the database
        self.engine = create_engine(
            f"postgresql://{credentials['RDS_USER']}:"
            f"{credentials['RDS_PASSWORD']}@"
            f"{credentials['RDS_HOST']}:{credentials['RDS_PORT']}/"
            f"{credentials['RDS_DATABASE']}"
        )

    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Fetch data from the database by executing the provided SQL query.

        Parameters:
            query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: A DataFrame containing the query results.

        Raises:
            SQLAlchemyError: If an error occurs while executing the SQL query.
            Exception: For any other unexpected errors.
        """
        try:
            # Establish a connection and execute the SQL query
            with self.engine.connect() as connection:
                result = pd.read_sql(query, connection)
            return result
        except SQLAlchemyError as e:
            # Handle SQLAlchemy-specific errors and return an empty DataFrame
            print(f"An error occurred while executing the query: {e}")
            return pd.DataFrame()
        except Exception as e:
            # Handle unexpected errors and return an empty DataFrame
            print(f"An unexpected error occurred: {e}")
            return pd.DataFrame()

    def close_connection(self) -> None:
        """
        Close the database connection by disposing of the SQLAlchemy engine.

        This method should be called to clean up the connection after all
        operations are completed.

        Returns:
            None
        """
        # Dispose of the engine to release any resources held by the connection
        self.engine.dispose()


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


class Plotter:
    """
     A class for creating various data visualisations.

    This class provides methods to generate plots such as line charts,
    scatter plots, histograms, bar plots, and more, to help analyse
    and interpret data.

    Attributes:
        dataframe (pd.DataFrame): The data to be visualised.
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        """
        Initialise the Plotter with a DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The data to be visualised.
        """
        self.dataframe = dataframe

    # --- Utility Methods ---

    def _create_subplots(
        self, num_plots: int, cols: int = 3, subplot_size: tuple[int, int] = (4, 4)
    ) -> tuple[int, int]:
        """
        Create a grid of subplots with a specified number of plots and columns.

        Parameters:
            num_plots (int): The total number of plots to create.
            cols (int, optional): The number of columns in the subplot grid.
            Default is 3. subplot_size (tuple, optional): The size of each subplot in
            inches. Default is (4, 4).

        Returns:
            tuple: A tuple containing the number of rows and columns used in the
                   subplot grid.

        This method adjusts the subplot grid to fit the specified number of plots and
        hides any extra axes that are not needed.
        """
        # Adjust cols to be the smaller value out of specified cols & num_plots
        cols = min(cols, num_plots)
        # Calculate the number of rows needed based on the number of columns
        rows = (num_plots // cols) + (num_plots % cols > 0)
        # Set the figure size based on the number of rows and columns
        fig, axes = plt.subplots(
            rows, cols, figsize=(subplot_size[0] * cols, subplot_size[1] * rows)
        )

        # Flatten axes if there's more than one, else convert it to a list
        if num_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        # Hide any extra axes
        for ax in axes[num_plots:]:
            ax.set_visible(False)

        # Return rows and cols to maintain compatibility with existing calls
        self.fig = fig
        self.axes = axes
        return rows, cols

    def auto_bin_width(self, data: np.array, num_bins: int = 25) -> float:
        """
        Calculate the optimal bin width for a histogram using a custom method.

        Parameters:
            data (array-like): The data for which to calculate the bin width.
            num_bins (int, optional): The number of bins to use for the calculation.
                                      Default is 25.

        Returns:
            float: The computed bin width, rounded to the nearest significant figure.

        Raises:
            ValueError: If the data contains fewer than two data points or if the
            bin width is zero.

        This method calculates the bin width by dividing the data range by the number
        of bins and rounds it to the nearest significant figure to ensure a meaningful
        histogram.
        """
        data = np.asarray(data)
        n = len(data)

        if n < 2:
            raise ValueError("Data must contain at least two data points.")

        # Calculate the data range to determine the initial bin width
        data_range = np.max(data) - np.min(data)

        # Calculate bin width
        bin_width = data_range / num_bins

        # Round the bin width to the nearest significant figure
        if bin_width == 0:
            raise ValueError("Bin width cannot be 0")

        # Determine the order of magnitude of the value
        order_of_magnitude = 10 ** (np.floor(np.log10(abs(bin_width))))
        # Round to the nearest significant figure
        rounded_bin_width = round(bin_width / order_of_magnitude) * order_of_magnitude

        # If rounding results in zero, return a small but non-zero value
        if rounded_bin_width == 0:
            rounded_bin_width = bin_width

        # Remove trailing .0 if the value is an integer
        if rounded_bin_width.is_integer():
            rounded_bin_width = int(rounded_bin_width)

        return rounded_bin_width

    # --- Plot Methods ---

    def plot_line_chart(
        self,
        ax: 'matplotlib.axes._subplots.AxesSubplot',
        x_values: 'pd.Index | array-like',
        y_values: 'pd.Series | dict',
        title: str = "Line Chart",
        group_columns: 'list | None' = None,
        x_label: 'str | None' = None,
        y_label: 'str | None' = None,
    ) -> None:
        """
        Generic line plot method to visualise data, with optional
        support for grouping.

        Parameters:
            ax (matplotlib.axes._subplots.AxesSubplot): The axes on which
                                                        to plot the line chart
            x_values (pd.Index or array-like): Values to plot on the x-axis.
            y_values (pd.Series or dict): The values to plot on the y-axis.
                                          Can be a single series or a
                                          dictionary for multiple groups.
            title (str, optional): Title for the plot. Default is "Line Chart".
            group_columns (list, optional): List of group labels for
                                            plotting multiple lines.
            x_label (str, optional): Label for the x-axis. Default is None,
                                     which uses the x_values label.
            y_label (str, optional): Label for the y-axis. Default is None,
                                     which uses the y_values label.
        """
        if group_columns is not None:
            # Plotting multiple lines for each group
            for group in group_columns:
                ax.plot(
                    x_values,
                    y_values[group],
                    marker="o",
                    linestyle="-",
                    label=f"{group}",
                )
            ax.legend(title="legend")
        else:
            # Plotting a single line
            ax.plot(x_values, y_values, marker="o", linestyle="-", color="red")
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels(x_values, rotation=45, ha="right")
        ax.set_title(title)
        ax.set_xlabel(x_label if x_label else str(x_values))
        ax.set_ylabel(y_label if y_label else str(y_values))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    def scatter_multiple_plots(self, column_pairs: list[tuple[str, str]]) -> None:
        """
        Create scatter plots for specified column pairs.

        Parameters:
            column_pairs (list of tuples): Each tuple contains two column
                                           names (x, y) to plot against each
                                           other.

        This method generates scatter plots for each pair of columns provided,
        allowing for visual comparison of relationships between variables.
        """
        num_plots = len(column_pairs)
        rows, cols = self._create_subplots(num_plots)

        for i, (x_col, y_col) in enumerate(column_pairs, 1):
            plt.subplot(rows, cols, i)
            plt.scatter(self.dataframe[x_col], self.dataframe[y_col])
            plt.title(f"{x_col} vs {y_col}")
            plt.xlabel(x_col, labelpad=15)
            plt.ylabel(y_col)

        plt.suptitle("Scatter plots for selected column pairs")
        plt.tight_layout()
        plt.show()

    def plot_histograms(self, exclude_columns: list = None) -> None:
        """
        Generate histograms for all numerical columns,
        excluding specified columns.

        Parameters:
            exclude_columns (list): List of columns to exclude
            from the histograms.
        """
        numeric_cols = filter_columns(
            self.dataframe, np.number, exclude_columns=exclude_columns
        )
        axes = self.dataframe[numeric_cols].hist(bins=20, figsize=(15, 10))

        # Iterate over the axes to set titles
        for ax, col in zip(axes.flatten(), numeric_cols):
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")

        plt.suptitle("Histograms for Numeric Columns")
        plt.tight_layout()
        plt.show()

    def plot_bar_plots(self, exclude_columns: list = None) -> None:
        """
        Generate bar plots for all categorical columns in the DataFrame.

        Parameters
        ----------
        exclude_columns : list, optional
            A list of columns to exclude from plotting. If None, all categorical columns
            will be plotted, by default None.

        Returns
        -------
        None
            Displays the bar plots of categorical columns.
        """
        # Filter out specified columns and only keep categorical columns
        categorical_cols = filter_columns(self.dataframe, "category", exclude_columns)

        # Start a new figure for plotting
        plt.figure()

        # Loop through each categorical column to generate a bar plot
        for i, col in enumerate(categorical_cols):
            plt.subplot(1, len(categorical_cols), i + 1)  # Create subplots in a row
            self.dataframe[col].value_counts().plot(kind="bar")
            plt.title(col)
            plt.ylabel("Frequency")  # Set y-axis title to 'Frequency'

        # Add a main title for all subplots and adjust layout to prevent overlap
        plt.suptitle("Bar plots of all categorical columns")
        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self, include_booleans: bool = False) -> None:
        """
        Generate a heatmap of correlations for numerical and optionally boolean columns.

        Parameters
        ----------
        include_booleans : bool, optional
            Whether to include boolean columns in the correlation calculation,
            by default False.

        Returns
        -------
        None
            Displays a heatmap of correlations for the selected DataFrame columns.
        """
        # Select numeric columns and optionally boolean columns from the DataFrame
        if include_booleans:
            cols_to_include = self.dataframe.select_dtypes(include=[np.number, "bool"])
        else:
            cols_to_include = self.dataframe.select_dtypes(include=[np.number])

        # Set figure size for the heatmap
        plt.figure(figsize=(10, 8))  # Adjust the figure size to accommodate all labels

        # Generate the heatmap for the selected columns
        sns.heatmap(
            cols_to_include.corr(),
            annot=True,  # Display correlation coefficients
            cmap="coolwarm",  # Set color map for visualisation
            fmt=".2f",  # Format the numbers to 2 decimal places
            cbar_kws={"shrink": 0.8},  # Control the size of the color bar
        )

        # Rotate x-axis & y-axis labels for readability
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)

        # Add title and adjust layout
        plt.title("Correlation Heatmap")
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def missing_data_matrix(self) -> None:
        """
        Display a missing data matrix for the DataFrame. This function helps
        identify missing data patterns which may be useful for further data imputation
        or data cleaning tasks.

        Returns
        -------
        None
            Displays a visualisation of the missing data in the DataFrame, highlighting
            patterns of missing values.
        """
        msno.matrix(self.dataframe, figsize=(14, 6), sparkline=False)
        plt.title("Missing Data Matrix")
        plt.tight_layout()
        plt.show()

    def plot_boxplots(
        self,
        dataframe: pd.DataFrame = None,
        columns: list = None,
        x_column: str = None,
        exclude_columns: list = None,
    ) -> None:
        """
        Generate boxplots for specified columns, optionally comparing against
        a categorical column (like failure states).

        Parameters:
            dataframe (DataFrame): DataFrame to be used for plotting.
                                   If None, defaults to self.dataframe.
            columns (list): List of columns to plot. If None, all numeric
                            columns will be used.
            x_column (str): Categorical column for x-axis
                            (e.g., 'Machine failure').
                            If None, boxplots will be created without a
                            categorical comparison.
            exclude_columns (list): List of columns to exclude from the plots.
        """
        # Use provided dataframe or default to self.dataframe
        dataframe_to_use = dataframe if dataframe is not None else self.dataframe

        # If columns are not provided, default to all numeric columns
        if columns is None:
            columns = filter_columns(dataframe_to_use, np.number, exclude_columns)

        num_cols = len(columns)

        # Create subplots based on number of columns
        rows, cols = self._create_subplots(num_cols, cols=3, subplot_size=(4, 4))

        # Plot each column
        for i, col in enumerate(columns, 1):
            plt.subplot(rows, cols, i)

            # If x_column is provided, use it for comparison,
            # else plot simple boxplot.
            if x_column:
                sns.boxplot(x=x_column, y=dataframe_to_use[col], data=dataframe_to_use)
                plt.xlabel(x_column)
            else:
                sns.boxplot(x=dataframe_to_use[col])
                plt.xlabel(col, labelpad=15)

            plt.title(f"Boxplot of {col}")

        plt.suptitle("Boxplots for Selected Columns")
        plt.tight_layout()
        plt.show()

    def plot_null_comparison(
        self, null_counts_before: pd.DataFrame, null_counts_after: pd.DataFrame
    ) -> None:
        """
        Plot a comparison of null counts before and after imputation.

        Parameters:
            null_counts_before (pd.DataFrame): A DataFrame containing the null counts
                                               before imputation.
            null_counts_after (pd.DataFrame): A DataFrame containing the null counts
                                              after imputation.

        This method visualises the change in null counts for each column, helping to
        assess the effectiveness of the imputation process.
        """
        # Create a DataFrame to hold null counts before and after imputation
        null_data = pd.DataFrame(
            {
                "Before Imputation": null_counts_before["null_count"],
                "After Imputation": null_counts_after["null_count"],
            }
        )

        # Generate the bar plot to compare null counts
        ax = null_data.plot(kind="bar")

        # Set the title and labels for the plot
        ax.set_title("Null Count Before and After Imputation")
        ax.set_ylabel("Null Count")

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

    def plot_skewness(self, exclude_columns: list = None) -> None:
        """
        Plot histograms for numeric columns with skewness information.

        Parameters:
            exclude_columns (list, optional): List of columns to exclude from the
                                             histograms. Default is None, which
                                             includes all numeric columns.

        This method generates histograms for each numeric column, displaying the
        skewness value in the title to help identify asymmetry in data distribution.
        """
        # Filter numeric columns, excluding specified ones
        numeric_columns = filter_columns(self.dataframe, np.number, exclude_columns)
        num_plots = len(numeric_columns)

        # Create subplots for each numeric column
        rows, cols = self._create_subplots(num_plots)

        for i, column in enumerate(numeric_columns, 1):
            # Calculate skewness for the current column
            skew_value = self.dataframe[column].skew()

            # Plot histogram for the current column
            plt.subplot(rows, cols, i)
            self.dataframe[column].hist(bins=50)
            plt.title(f"{column} (Skew: {skew_value:.2f})")
            plt.ylabel("Frequency")  # Set y-axis title to 'Frequency'
            plt.xlabel(column)  # Set x-axis title to the column name

        # Set a super title for all subplots
        plt.suptitle("Histograms for Numeric Columns with Skewness Value")
        plt.tight_layout()
        plt.show()

    def visualise_transformed_column(
        self, column: str, original: pd.Series, transformed: pd.Series
    ):
        """
        Visualise the original vs transformed column and show skew values.
        """
        # Calculate skew values
        original_skew = original.skew()
        transformed_skew = pd.Series(transformed).skew()

        # Create the plots
        plt.figure(figsize=(12, 6))

        # Plot the original data
        plt.subplot(1, 2, 1)
        plt.hist(original, bins=30, color="blue", alpha=0.7)
        plt.title(f"Original {column} (Skew: {original_skew:.2f})")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        # Plot the transformed data
        plt.subplot(1, 2, 2)
        plt.hist(transformed, bins=30, color="green", alpha=0.7)
        plt.title(f"Transformed {column} (Skew: {transformed_skew:.2f})")
        plt.xlabel(column)
        plt.ylabel("Frequency")

        # Add a main title and show the plots
        plt.suptitle(f"Original vs Transformed {column}")
        plt.show()

    def plot_qq(self, exclude_columns: list = None) -> None:
        """
        Generate Q-Q plots for numeric columns to assess normality.

        Parameters:
            exclude_columns (list, optional): List of columns to exclude from the
                                              Q-Q plots. Default is None, which
                                              includes all numeric columns.

        This method creates Q-Q plots for each numeric column in the DataFrame,
        which are useful for visually assessing how closely the data follows a
        normal distribution.
        """
        # Filter numeric columns, excluding specified ones
        numeric_columns = filter_columns(self.dataframe, np.number, exclude_columns)
        num_plots = len(numeric_columns)

        # Create subplots for each numeric column
        rows, cols = self._create_subplots(num_plots)

        for i, col in enumerate(numeric_columns, 1):
            # Plot Q-Q plot for the current column
            plt.subplot(rows, cols, i)
            qqplot(self.dataframe[col], line="q", ax=plt.gca())
            plt.title(f"Q-Q plot of {col}")

        # Set a super title for all subplots
        plt.suptitle("Q-Q Plots for Numeric Columns")
        plt.tight_layout()
        plt.show()

    # ------------ Further Analysis ------------ #

    def plot_tool_wear_distribution(self, bins: int = 30) -> None:
        """
        Plot a histogram to visualise the distribution of tool wear values.

        Parameters:
            bins (int, optional): Number of bins to use in the histogram. Default is 30.

        This method creates a histogram to show the distribution of tool wear
        times, helping to identify patterns or anomalies in tool usage.
        """
        # Set the figure size for the plot
        plt.figure(figsize=(10, 6))

        # Plot histogram of tool wear values with specified number of bins
        self.dataframe["Tool wear [min]"].plot(
            kind="hist", bins=bins, edgecolor="black"
        )

        # Set plot titles and labels
        plt.title("Tool Wear Distribution")
        plt.xlabel("Tool wear [min]")
        plt.ylabel("Number of Tools")

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

    def calculate_failure_rate(self) -> None:
        """
        Calculate and visualise the total number and percentage of failures.

        This method calculates the failure rate in the manufacturing process and
        visualises it using a pie chart, providing insights into the proportion
        of failed processes.
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

        # Visualise failure rate with a pie chart
        labels = ["Failures", "Non-failures"]
        sizes = [total_failures, total_processes - total_failures]
        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",  # Display percentage on the pie chart
            startangle=90,  # Start the pie chart at 90 degrees
            colors=["red", "green"],  # Use red for failures and green for non-failures
        )
        plt.title("Failure Rate in the Manufacturing Process")
        plt.tight_layout()
        plt.show()

    def failures_by_product_quality(self) -> None:
        """
        Analyse and visualise failures based on product quality types.

        This method calculates the number and percentage of failures for each
        product quality type, prints the results to the terminal, and visualises
        them using a bar chart.
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

        # Visualize failures by product quality using a bar chart
        plt.figure(figsize=(8, 6))
        bars = plt.bar(
            failures_by_quality.index,
            failures_by_quality.values,
            color=["blue", "orange", "green"],
            edgecolor="black",
        )

        # Annotate bars with the percentage of total failures
        for bar, percent in zip(bars, failure_percent_by_quality):
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{percent:.1f}%",
                ha="center",
                va="bottom",
            )

        # Set plot titles and labels
        plt.title("Failure Rate by Product Quality (Type)")
        plt.xlabel("Product Quality (Type)")
        plt.ylabel("Number of Failures")

        # Add a legend explaining the percentage annotation
        plt.text(
            0.95,
            0.95,
            "% = failure rate in each quality type",
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
            bbox=dict(facecolor="white", alpha=0.6),
        )

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()

    def leading_causes_of_failure(self) -> None:
        """
        Determine, print, and visualise the leading causes of failure.

        This method calculates the total number of failures for each failure
        type, prints the results, and visualises them using a bar chart to
        highlight the most common causes of failure.
        """
        failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

        # Sum the failures for each failure type
        cause_of_failure_counts = self.dataframe[failure_columns].sum()
        total_failures = cause_of_failure_counts.sum()

        # Print the total number of failures for each cause
        print("\nLeading Causes of Failure:\n", cause_of_failure_counts.to_string())

        # Calculate the percentage of failures for each cause
        failure_percent_by_cause = (cause_of_failure_counts / total_failures) * 100

        # Visualise the causes of failure using a bar chart
        plt.figure(figsize=(8, 6))
        bars = plt.bar(
            cause_of_failure_counts.index,
            cause_of_failure_counts.values,
            color="red",
            edgecolor="black",
        )

        # Annotate bars with the percentage of total failures
        for bar, percent in zip(bars, failure_percent_by_cause):
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval,
                f"{percent:.1f}%",
                ha="center",
                va="bottom",
            )

        # Set plot titles and labels
        plt.title("Leading Causes of Failure")
        plt.xlabel("Failure Type")
        plt.ylabel("Number of Failures")
        plt.tight_layout()
        plt.show()

    def failure_causes_by_product_quality(self) -> None:
        """
        Visualise leading causes of failure grouped by product quality with
        percentages via Browser.

        This method groups failures by product quality and failure type,
        calculates the percentage of each failure type within each quality
        group, and visualises the results using an interactive bar chart.
        """

        failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]
        failure_data = self.dataframe[self.dataframe["Machine failure"] == 1]

        # Group and prepare data for visualisation
        failures_grouped = (
            failure_data.groupby("Type")[failure_columns].sum().reset_index()
        )
        failures_melted = failures_grouped.melt(
            id_vars="Type", var_name="Failure Type", value_name="Count"
        )

        # Calculate percentages for each failure type within each quality group
        totals = failures_melted.groupby("Type")["Count"].transform("sum")
        failures_melted["Percentage"] = (failures_melted["Count"] / totals) * 100

        # Create an interactive plot via browser
        fig = px.bar(
            failures_melted,
            x="Type",
            y="Count",
            color="Failure Type",
            title="Failures by Product Quality and Failure Type",
            labels={"Type": "Product Quality (Type)", "Count": "Number of Failures"},
            hover_data={"Percentage": ":.1f%"},
        )
        fig.update_layout(
            barmode="stack",
            xaxis_title="Product Quality (Type)",
            yaxis_title="Number of Failures",
        )
        fig.show()

    def failure_rate_analysis(
        self,
        dataframe: pd.DataFrame,
        selected_column: list,
        target_column: list,
    ) -> None:
        """
        Create subplots of line charts using selected columns and target columns.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            selected_column (list): List of machine setting columns.
            target_column (list): List of failure type columns.
        """
        num_plots = len(selected_column)
        self._create_subplots(num_plots=num_plots, cols=3, subplot_size=(6, 4))
        axes = self.axes

        for idx, setting in enumerate(selected_column):
            # Calculate bin width using the auto_bin_width method
            bin_width = self.auto_bin_width(dataframe[setting])

            # Creates bins with calculated width starting from the min. value of data
            min_value = dataframe[setting].min()
            dataframe["Selected Bin"] = pd.cut(
                dataframe[setting],
                bins=(
                    np.arange(
                        min_value, dataframe[setting].max() + bin_width, bin_width
                    ).astype(int)
                    if isinstance(bin_width, int)
                    else np.arange(
                        min_value, dataframe[setting].max() + bin_width, bin_width
                    )
                ),
                right=False,  # Ensures the bins are left-closed, right-open
            )

            # Group by 'Selected Bin' & 'target_column' & calculates
            # failure rate for each type.
            grouped = dataframe.groupby(["Selected Bin"])[target_column].agg(
                ["sum", "size"]
            )
            failure_rates = {}
            for failure in target_column:
                failure_rates[failure] = (
                    grouped[(failure, "sum")] / grouped[(failure, "size")]
                ) * 100

            # Plotting the failure rate for each failure type
            self.plot_line_chart(
                ax=axes[idx],
                x_values=failure_rates[target_column[0]].index.astype(str),
                y_values=failure_rates,
                title=f"Failure Rate Analysis for {setting}",
                group_columns=target_column,
                x_label=f"{setting} Bins",
                y_label="Failure Rate (%)",
            )

            # Remove 'Selected Bin' column after analysis
            dataframe.drop(columns=["Selected Bin"], inplace=True)

        plt.suptitle("Failure Rate Analysis")
        plt.tight_layout()
        plt.show()

    def plot_violin_plots(
        self, dataframe: pd.DataFrame, columns: list, x_column: str
    ) -> None:
        """
        Create violin plots for specified columns, grouped by a categorical column.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame containing the data.
            columns (list): List of column names to plot as violin plots.
            x_column (str): The categorical column for grouping (e.g., 'Type').

        This method generates violin plots for each specified column, grouped by
        the given categorical column, to visualise the distribution and density
        of the data.
        """
        # Determine the number of columns to plot
        num_cols = len(columns)

        # Create subplots layout using the _create_subplots method
        rows, cols = self._create_subplots(num_cols)

        # Generate a violin plot for each specified column
        for i, column in enumerate(columns, 1):
            plt.subplot(rows, cols, i)
            sns.violinplot(data=dataframe, x=x_column, y=column)
            plt.title(f"Violin Plot of {column} by {x_column}")

        # Set a super title for all subplots
        plt.suptitle(f"Violin Plots for Selected Columns by {x_column}")
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()


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


class MachineSettingCalculator:
    """
    Class to handle machine setting calculations and analysis.

    This class provides methods to calculate session statistics based on
    machine settings, initialise and update settings, and display results
    in a tabular format.
    """

    def __init__(self, pre_transform_data: pd.DataFrame) -> None:
        """
        Initialise the MachineSettingCalculator with pre-transformed data.

        Parameters:
            pre_transform_data (pd.DataFrame): The DataFrame containing
            pre-transformed data for analysis.
        """
        self.pre_transform_data = pre_transform_data

    def calculate_setting_limit(
        self, data: pd.DataFrame, column: str, min_value: float = None,
        max_value: float = None
    ) -> dict:
        """
        Calculate the number and percentage of sessions above and below given
        min and max thresholds for a specified machine setting column.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the data.
            column (str): The column to analyse.
            min_value (float, optional): The minimum threshold value.
            max_value (float, optional): The maximum threshold value.

        Returns:
            dict: A dictionary containing the calculated statistics, including
            sessions missed and failed sessions for both min and max thresholds.
        """
        total_sessions = len(data)
        total_failed_sessions = data[data["Machine failure"]].shape[0]

        # Calculate sessions below the minimum threshold, if specified
        sessions_missed_min = (
            data[data[column] < min_value].shape[0] if min_value is not None else 0
        )

        # Calculate sessions above the maximum threshold, if specified
        sessions_missed_max = (
            data[data[column] > max_value].shape[0] if max_value is not None else 0
        )

        # Calculate percentages for sessions missed
        percentage_missed_min = (
            (sessions_missed_min / total_sessions) * 100 if min_value is not None else 0
        )
        percentage_missed_max = (
            (sessions_missed_max / total_sessions) * 100 if max_value is not None else 0
        )

        # Calculate failed sessions below the minimum threshold
        failed_sessions_min = (
            data[(data[column] < min_value) & data["Machine failure"]].shape[0]
            if min_value is not None
            else 0
        )

        # Calculate failed sessions above the maximum threshold
        failed_sessions_max = (
            data[(data[column] > max_value) & data["Machine failure"]].shape[0]
            if max_value is not None
            else 0
        )

        # Calculate percentages for failed sessions
        percentage_failed_sessions_min = (
            (failed_sessions_min / total_failed_sessions) * 100
            if min_value is not None
            else 0
        )
        percentage_failed_sessions_max = (
            (failed_sessions_max / total_failed_sessions) * 100
            if max_value is not None
            else 0
        )

        return {
            "min": min_value,
            "max": max_value,
            "sessions_missed_min": sessions_missed_min,
            "percentage_missed_min": percentage_missed_min,
            "sessions_missed_max": sessions_missed_max,
            "percentage_missed_max": percentage_missed_max,
            "failed_sessions_min": failed_sessions_min,
            "failed_sessions_max": failed_sessions_max,
            "percentage_failed_sessions_min": percentage_failed_sessions_min,
            "percentage_failed_sessions_max": percentage_failed_sessions_max,
        }

    def initialise_machine_settings(self) -> dict:
        """
        Initialise machine settings with descriptive names.

        Returns:
            dict: A dictionary mapping setting keys to descriptive names.
        """
        return {
            "a": "Torque [Nm]",
            "b": "Rotational speed [rpm]",
            "c": "Tool wear [min]",
            "d": "Process temperature [K]",
            "e": "Air temperature [K]",
        }

    def initialise_table(self, machine_settings: dict) -> dict:
        """
        Initialise a table with min and max values set to None for each setting.

        Parameters:
            machine_settings (dict): A dictionary of machine settings.

        Returns:
            dict: A dictionary with settings as keys and min/max values as None.
        """
        return {
            setting: {"min": None, "max": None} for setting in machine_settings.values()
        }

    def display_table(self, table: dict) -> None:
        """
        Display the table with calculated statistics.

        Parameters:
            table (dict): A dictionary containing machine settings and their
            min/max values.
        """
        table_data = []
        for setting, values in table.items():
            # Calculate statistics for each setting using the current min/max values
            stats = self.calculate_setting_limit(
                self.pre_transform_data, setting, values["min"], values["max"]
            )

            # Append statistics for the minimum threshold
            table_data.append(
                [
                    setting,
                    f"Min: {values['min']}" if values["min"] is not None else "N/A",
                    stats["sessions_missed_min"],
                    f"{stats['percentage_missed_min']:.2f}",
                    stats["failed_sessions_min"],
                    f"{stats['percentage_failed_sessions_min']:.2f}",
                ]
            )

            # Append statistics for the maximum threshold
            table_data.append(
                [
                    "",
                    f"Max: {values['max']}" if values["max"] is not None else "N/A",
                    stats["sessions_missed_max"],
                    f"{stats['percentage_missed_max']:.2f}",
                    stats["failed_sessions_max"],
                    f"{stats['percentage_failed_sessions_max']:.2f}",
                ]
            )

        # Define headers for the table
        headers = [
            "Setting",
            "Min / Max",
            "Sessions\nMissed",
            "% Missed",
            "Failed Sessions\nAvoided",
            "% Failed\nAvoided",
        ]
        # Print the table using tabulate for a formatted grid display
        print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f"))

    def update_setting(self, machine_settings: dict, table: dict) -> None:
        """
        Update the min or max value for a selected machine setting.

        Parameters:
            machine_settings (dict): A dictionary of machine settings.
            table (dict): A dictionary containing machine settings and their
            min/max values.
        """
        print("\nSelect a machine setting to update:\n")
        # Display available machine settings for selection
        for key, setting in machine_settings.items():
            print(f"{key}: {setting}")

        # Prompt user to select a setting to update
        selected_key = input("\nEnter a, b, c, d, or e: ").lower()

        if selected_key in machine_settings:
            selected_setting = machine_settings[selected_key]
            # Ask whether to update the min or max value
            min_or_max = input("\nDo you want to set a 'min' or 'max' value? ").lower()

            if min_or_max in ["min", "max"]:
                while True:
                    try:
                        # Prompt user to enter a numeric value for the selected setting
                        value = float(
                            input(
                                f"\nEnter the {min_or_max} value for "
                                f"{selected_setting}: "
                            )
                        )
                        table[selected_setting][min_or_max] = value
                        break
                    except ValueError:
                        print("\nInvalid input. Please enter a numeric value.")
            else:
                print("\nInvalid selection. Please enter 'min' or 'max'.")
        else:
            print("\nInvalid selection. Please try again.")

    def continue_analysis_prompt(self) -> bool:
        """
        Prompt the user to continue or end the analysis.

        Returns:
            bool: True if the user wants to continue, False otherwise.
        """
        while True:
            # Ask user if they want to update another setting
            continue_analysis = input(
                "\nDo you want to update another setting? (yes/no): "
            ).lower()
            if continue_analysis in ["yes", "no"]:
                return continue_analysis == "yes"
            else:
                print("\nInvalid input. Please enter 'yes' or 'no'.")

    def run_machine_setting_calculator(self) -> None:
        """
        Run the machine setting analysis process.

        This method initialises machine settings, displays the table of
        statistics, and allows the user to update settings and continue
        analysis until they choose to stop.
        """
        # Initialise machine settings and table
        machine_settings = self.initialise_machine_settings()
        table = self.initialise_table(machine_settings)

        while True:
            # Display total number of sessions and failed sessions
            print(f"\nTotal number of sessions: {len(self.pre_transform_data)}")
            total_failed_sessions = self.pre_transform_data[
                self.pre_transform_data["Machine failure"]
            ].shape[0]
            print(f"Total number of failed sessions: {total_failed_sessions}")

            # Display the current table of statistics
            self.display_table(table)
            # Allow user to update settings
            self.update_setting(machine_settings, table)

            # Check if the user wants to continue updating settings
            if not self.continue_analysis_prompt():
                # Final display of total sessions and failed sessions
                print(f"\nTotal number of sessions: {len(self.pre_transform_data)}")
                print(f"Total number of failed sessions: {total_failed_sessions}")
                print("\nFinal Table:")
                self.display_table(table)
                break


class EDAExecutor:
    """
    Class to run all EDA, visualisation, and transformations.

    This class provides methods to fetch, reformat, explore, visualise, and
    transform data, facilitating comprehensive exploratory data analysis.
    """

    def __init__(self, db_connector: object) -> None:
        """
        Initialise the EDAExecutor with a database connector.

        Parameters:
            db_connector: An object responsible for database connections and
            data retrieval.
        """
        self.data = None
        self.pre_transform_data = None
        self.db_connector = db_connector

    def fetch_and_save_data(self, query: str) -> pd.DataFrame:
        """
        Fetch data from the database and save it to a CSV file.

        Parameters:
            query (str): The SQL query to execute for data retrieval.

        Returns:
            pd.DataFrame: The DataFrame containing the fetched data.
        """
        data = self.db_connector.fetch_data(query)
        if not data.empty:
            csv_filename = "failure_data.csv"
            save_data_to_csv(dataframe=data, filename=csv_filename)
            print(
                "\nData successfully retrieved from database",
                f"and saved to '{csv_filename}'.",
            )
        return data

    def reformat_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Reformat data to ensure correct column formats, such as converting
        datatypes of specified columns.

        Parameters:
            data (pd.DataFrame): The DataFrame to reformat.

        Returns:
            pd.DataFrame: The reformatted DataFrame.
        """
        transformer = DataTransform(data)

        # Convert 'Type' column to categorical
        transformer.convert_to_categorical("Type")

        # Convert failure columns to boolean
        for col in ["Machine failure", "TWF", "HDF", "PWF", "OSF", "RNF"]:
            transformer.convert_to_boolean(col)

        return data  # Return the reformatted data

    def explore_stats(self, data: pd.DataFrame) -> None:
        """
        Run basic statistical exploration on the data and print results
        to the terminal.

        Parameters:
            data (pd.DataFrame): The DataFrame to explore.
        """
        print("\nPrinting statistics...")
        df_info = DataFrameInfo(data)

        # List which columns to exclude from certain analyses
        exclude_columns = ["UDI", "Product ID"]

        # Print various statistics
        print("\nColumn Descriptions:\n", df_info.describe_columns(
                exclude_columns=exclude_columns
            )
        )
        print(
            "\nExtracted Statistics:\n",
            df_info.extract_stats(exclude_columns=exclude_columns),
        )

        # Call the method without using print, since it already prints the output
        print("\nDistinct Value Counts:")
        df_info.count_distinct_values(dataframe=data, exclude_columns=exclude_columns)

        # Print the shape of the dataframe
        df_info.print_shape()

        # Print the null value counts
        print("\nNull Value Counts:\n", df_info.count_null_values())

    def visualise_data(self, data: pd.DataFrame) -> None:
        """
        Generate visualisations for the data.

        Parameters:
            data (pd.DataFrame): The DataFrame to visualise.
        """
        print("\nGenerating visualisations...")
        plotter = Plotter(data)

        # Define column pairs for scatter plots
        scatter_plot_column_pairs = [
            ("Air temperature [K]", "Process temperature [K]"),
            ("Rotational speed [rpm]", "Torque [Nm]"),
            ("Tool wear [min]", "Rotational speed [rpm]"),
            ("Tool wear [min]", "Process temperature [K]"),
        ]

        # Generate various plots
        plotter.scatter_multiple_plots(scatter_plot_column_pairs)
        plotter.plot_histograms(exclude_columns="UDI")
        plotter.plot_bar_plots(exclude_columns="Product ID")
        plotter.correlation_heatmap()
        plotter.missing_data_matrix()
        plotter.plot_boxplots(exclude_columns="UDI")
        plotter.plot_skewness(exclude_columns="UDI")
        plotter.plot_qq(exclude_columns="UDI")

    def run_imputation_and_null_visualisation(
        self, data: pd.DataFrame, visualisations_on: bool = True
    ) -> None:
        """
        Handle null imputation and optionally visualise null count comparison.

        Parameters:
            data (pd.DataFrame): The DataFrame to impute.
            knn_columns (dict, optional): Dictionary specifying columns for
            KNN imputation.
            visualisations_on (bool, optional): Flag to enable visualisation.
        """
        df_transform = DataFrameTransform(data)
        df_info = DataFrameInfo(data)
        plotter = Plotter(data)

        # Specify imputation strategy
        imputation_strategy = {
            "Air temperature [K]": "mean",
            "Process temperature [K]": "knn",
            "Tool wear [min]": "median",
        }

        # Define the columns for KNN imputation
        knn_cols = {"Process temperature [K]": ["Air temperature [K]"]}

        # Null count before and after imputation
        initial_null_count = df_info.count_null_values()

        # Pass the strategy and KNN columns to impute_missing_values
        df_transform.impute_missing_values(
            strategies=imputation_strategy, knn_columns=knn_cols
        )

        null_count_after_impute = df_info.count_null_values()
        print("\nPost impute null value counts:\n", null_count_after_impute)

        # Visualise null count comparison (if visualisation flag is True)
        if visualisations_on:
            plotter.plot_null_comparison(initial_null_count, null_count_after_impute)

    def handle_skewness_and_transformations(
        self, data: pd.DataFrame, machine_setting: str, visualisations_on: bool = True
    ) -> None:
        """
        Handle skewness detection and transformation of columns. Allows user to
        select a transformation method and visualise the transformation.

        Parameters:
            data (pd.DataFrame): The DataFrame to transform.
            machine_setting (str): The column name to apply transformations to.
            visualisations_on (bool, optional): Flag to enable visualisation.
        """
        df_transform = DataFrameTransform(data)
        plotter = Plotter(data)

        # Save current dataframe to self.pre_transform_data pre transformation
        if self.pre_transform_data is None:
            self.pre_transform_data = data.copy()

        # Preview and visualise transformation (if visualisation flag is True)
        if visualisations_on:
            df_transform.preview_transformations(machine_setting)

        # Retrieve the specified column from pre-transform data
        original_data = self.pre_transform_data[machine_setting]

        # Prompt user to select a transformation method
        print("\nSelect a transformation method:")
        print("a: Log Transform")
        print("b: Box-Cox Transform")
        print("c: Yeo-Johnson Transform")
        choice = input("Enter your choice (a, b, c): ").strip().lower()

        # Perform the selected transformation
        if choice == 'a':
            transformed_data = np.log1p(
                df_transform.dataframe[machine_setting]
            )
        elif choice == 'b':
            transformed_data, _ = stats.boxcox(
                df_transform.dataframe[machine_setting] + 1
            )
        elif choice == 'c':
            transformed_data, _ = stats.yeojohnson(
                df_transform.dataframe[machine_setting]
            )
        else:
            print("Invalid choice. No transformation applied.")
            return

        # Update the dataframe with the transformed data
        df_transform.dataframe[machine_setting] = transformed_data

        # Visualise transformation (if visualisation flag is True)
        if visualisations_on:
            plotter.visualise_transformed_column(
                column=machine_setting,
                original=original_data,
                transformed=transformed_data,
            )

    def handle_outlier_detection(
        self, data: pd.DataFrame, machine_setting: str, remove_outliers: bool = False
    ) -> pd.DataFrame:
        """
        Detect and optionally remove outliers in the data.

        Parameters:
            data (pd.DataFrame): The DataFrame to analyse for outliers.
            machine_setting (str): The column name to analyse for outliers.
            remove_outliers (bool): Flag to remove detected outliers.

        Returns:
            pd.DataFrame: The DataFrame with outliers removed if specified.
        """
        # Initialise helper classes for data analysis and plotting
        df_info = DataFrameInfo(data)
        plotter = Plotter(data)

        # Detect outliers using the z-score method for the specified column
        zscore_outliers = df_info.detect_outliers_zscore(machine_setting)
        # Print the number of outliers detected
        print(f"\nNumber of z-score outliers detected: {len(zscore_outliers)}")

        # Detect outliers using the IQR method for the specified column
        iqr_outliers = df_info.detect_outliers_iqr(machine_setting)
        # Print the number of outliers detected
        print(f"\nNumber of IQR outliers detected: {len(iqr_outliers)}")

        original_count = len(data)

        if remove_outliers:
            # Ask user for removal preference
            print("\nChoose outlier removal method:")
            print("1: Remove combined outliers from both methods")
            print("2: Remove only outliers that appear in both methods")
            choice = input("Enter your choice (1 or 2): ").strip()

            if choice == '1':
                # Combine outliers from both methods
                combined_outliers = pd.concat(
                    [zscore_outliers, iqr_outliers]).drop_duplicates()
                data = data[~data.index.isin(combined_outliers.index)]
                print(f"\nCombined outliers removed: {original_count - len(data)}")
            elif choice == '2':
                # Find outliers that appear in both methods
                common_outliers = zscore_outliers.index.intersection(iqr_outliers.index)
                data = data[~data.index.isin(common_outliers)]
                print(f"\nCommon outliers removed: {original_count - len(data)}")
            else:
                print("Invalid choice. No outliers removed.")

        # Visualise relationships between different data columns
        plotter.scatter_multiple_plots(
            [
                ("Air temperature [K]", "Process temperature [K]"),
                ("Torque [Nm]", machine_setting),
                ("Tool wear [min]", machine_setting),
            ]
        )

        # Plot boxplots for all columns except 'UDI'
        plotter.plot_boxplots(exclude_columns="UDI")

        return data

    # -- Further Analysis --

    # Task 1: Operating Ranges Analysis
    def analyse_operating_ranges(self, data: pd.DataFrame) -> None:
        """
        Analyse and display operating ranges across product types.

        Parameters:
            data (pd.DataFrame): The DataFrame to analyse for operating ranges.
        """
        plotter = Plotter(data)
        selected_columns = [
            "Air temperature [K]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Torque [Nm]",
            "Tool wear [min]",
        ]

        # Ensure pre_transform_data is available
        if self.pre_transform_data is None:
            raise ValueError("Pre-transformation data is not available.")

        # Display overall operating ranges
        print("\nOverall Operating Ranges:")
        df_info = DataFrameInfo(self.pre_transform_data)
        overall_ranges = df_info.describe_columns(
            include_columns=selected_columns,
            stats_to_return=["min", "max", "25%", "75%", "mean"],
        )
        # Round to 3 decimal places
        overall_ranges = overall_ranges.round(3)
        print(overall_ranges)

        # Loop through each product quality type and display the stats
        for product_type in ["L", "M", "H"]:
            print(f"\nOperating Ranges for Product Type: {product_type}")

            filtered_data = self.pre_transform_data[
                self.pre_transform_data["Type"] == product_type
            ]

            df_info_filtered = DataFrameInfo(filtered_data)

            type_specific_ranges = df_info_filtered.describe_columns(
                include_columns=selected_columns,
                stats_to_return=["min", "max", "25%", "75%", "mean"],
            )
            # Round to 3 decimal places
            type_specific_ranges = type_specific_ranges.round(3)
            print(type_specific_ranges)

        plotter.plot_tool_wear_distribution()

    # Task 2: Determine the failure rate in the process
    def analyse_failures(self, data: pd.DataFrame) -> None:
        """
        Analyse failures by product quality and failure type.

        Parameters:
            data (pd.DataFrame): The DataFrame to analyse for failures.
        """
        plotter = Plotter(data)

        # Calculations and plots for failure analysis
        plotter.calculate_failure_rate()
        plotter.failures_by_product_quality()
        plotter.leading_causes_of_failure()
        plotter.failure_causes_by_product_quality()

    # Task 3: Deeper Understanding of Failures
    def analyse_failure_risk_factors(self, data: pd.DataFrame) -> None:
        """
        Investigate potential risk factors for machine failures.

        This method explores whether certain machine settings (e.g., torque,
        temperatures, rpm) are correlated with the different types of failures.
        The aim is to identify specific conditions that lead to increased
        failure rates.

        Parameters:
            data (pd.DataFrame): The DataFrame to analyse for risk factors.
        """
        plotter = Plotter(data)

        # For failure risk factors with failure comparison
        # Boxplots of failure type vs setting value
        failure_types = ["Machine failure", "HDF", "PWF", "OSF", "TWF"]
        machine_settings = [
            "Torque [Nm]",
            "Process temperature [K]",
            "Rotational speed [rpm]",
            "Air temperature [K]",
            "Tool wear [min]",
        ]

        # Generate correlation heatmap including failure types
        plotter.correlation_heatmap(include_booleans=True)

        plotter.failure_rate_analysis(
            dataframe=self.pre_transform_data,
            selected_column=machine_settings,
            target_column=failure_types,
        )

        plotter.plot_violin_plots(
            dataframe=self.pre_transform_data, columns=machine_settings, x_column="Type"
        )

        # Uncomment to see the boxplots for each failure type
        """
        # Creates windows for each failure type, each with 5 boxplots,
        # one for each machine setting.

        for failure_types in failure_types:
            plotter.plot_boxplots(
                dataframe=self.pre_transform_data,
                columns=machine_settings,
                x_column=failure_types,
            )
        """

        # Boxplots for machine settings by product quality ('Type')
        plotter.plot_boxplots(
            dataframe=self.pre_transform_data, columns=machine_settings, x_column="Type"
        )

        # Ensure pre_transform_data is available
        if self.pre_transform_data is None:
            raise ValueError("Pre-transformation data is not available.")

        # Loop through chosen failure type's and True/False statuses
        for failure_type in ["Machine failure", "HDF", "OSF", "PWF"]:
            for failure_status in [True, False]:
                print(f"\nOperating Ranges for {failure_type} = {failure_status}")

                # Filter the pre-transform data based on the failure status
                filtered_data = self.pre_transform_data[
                    self.pre_transform_data[failure_type] == failure_status
                ]

                # Create a DataFrameInfo object for the filtered data
                df_info_filtered = DataFrameInfo(filtered_data)

                # Call the describe_columns method for the filtered dataset
                failure_specific_ranges = df_info_filtered.describe_columns(
                    include_columns=machine_settings,
                    stats_to_return=["min", "max", "25%", "75%", "mean"],
                )
                # Round to 3 decimal places
                failure_specific_ranges = failure_specific_ranges.round(3)
                print(failure_specific_ranges)

    def further_analysis(self, data: pd.DataFrame) -> None:
        """
        Conduct further analysis by calling task-specific methods.

        Parameters:
            data (pd.DataFrame): The DataFrame to analyse further.
        """
        # Task 1: operating range calculator
        print("\nAnalysing operating ranges...")
        self.analyse_operating_ranges(data)
        print("\nAnalysis of operating ranges complete..")
        input("\nPress Enter to continue...")

        # Task 2: failure analysis
        print("\nAnalysing failures and failure types...")
        self.analyse_failures(data)
        print("\nAnalysis of failures and failure types complete..")
        input("\nPress Enter to continue...")

        # Task 3: failure risk factors
        print("\nAnalysing failure risk factors...")
        self.analyse_failure_risk_factors(data)
        print("\nAnalysis of failure risk factors complete..")
        input("\nPress Enter to continue...")

        # Machine setting calculator
        print("\nCalculate machine settings...")
        calculator = MachineSettingCalculator(self.pre_transform_data)
        calculator.run_machine_setting_calculator()
        print("\nCalculation of machine settings complete..")


if __name__ == "__main__":
    """
    Flag control system:
    Each flag corresponds to a different step in the EDA process.
    Set a flag to True to include that step, or False to skip it.
    """
    run_reformat = True  # Reformat data (e.g., column types, categories)
    run_explore_stats = True  # Explore statistics
    run_visualisation = True  # Generate visualisations for data
    run_null_imputation = True  # Carry out null imputation & visualisation
    run_skewness_transformations = True  # Preview & perform transformation
    run_outlier_detection = True  # Detect and visualise outliers
    run_drop_columns = True  # Drop columns after analysis (if applicable)
    run_save_data = True  # Save transformed data
    run_further_analysis = True  # Carry out more in-depth analysis

    # Load database credentials and connect
    credentials = load_db_credentials("credentials.yaml")
    db_connector = RDSDatabaseConnector(credentials)

    # Create an instance of EDAExecutor
    eda_executor = EDAExecutor(db_connector)

    # Fetch and save data
    data = eda_executor.fetch_and_save_data(query="SELECT * FROM failure_data;")

    # Create an instance of df transform
    df_transform = DataFrameTransform(data)

    if not data.empty:
        if run_reformat:
            # Reformat data as needed (ensure correct formats, types, etc.)
            data = eda_executor.reformat_data(data)
            print("\nrun_reformat complete..")
            input("\nPress Enter to continue...")

        if run_explore_stats:
            # Perform initial exploration of data
            eda_executor.explore_stats(data)
            print("\nrun_explore_stats complete..")
            input("\nPress Enter to continue...")

        if run_visualisation:
            # Perform visualisation of data
            eda_executor.visualise_data(data)
            print("\nrun_visualisation complete..")
            input("\nPress Enter to continue...")

        if run_null_imputation:
            # Perform null imputation/removal and visualise the result
            eda_executor.run_imputation_and_null_visualisation(
                data, visualisations_on=True  # visualisations on/off
            )
            print("\nrun_null_imputation complete..")
            input("\nPress Enter to continue...")

        if run_skewness_transformations:
            # Skewness and transformations
            eda_executor.handle_skewness_and_transformations(
                data,
                machine_setting="Rotational speed [rpm]",
                visualisations_on=True  # visualisations on/off
            )
            print("\nrun_skewness_transformations complete..")
            input("\nPress Enter to continue...")

        if run_outlier_detection:
            # Specify the column name you want to analyse for outliers
            machine_setting = "Rotational speed [rpm]"
            # Outlier detection - Only detects outliers
            eda_executor.handle_outlier_detection(
                data, machine_setting, remove_outliers=True
            )
            print("\nrun_outlier_detection complete..")
            input("\nPress Enter to continue...")

        if run_drop_columns:
            # Drop columns after analysis (if applicable)
            columns_to_drop = ["Air temperature [K]"]
            df_transform.drop_columns(columns_to_drop)
            print("\nrun_drop_columns complete..")
            input("\nPress Enter to continue...")

        if run_save_data:
            # Save the transformed data directly using save_data_to_csv
            save_data_to_csv(df_transform.dataframe, "transformed_failure_data.csv")
            save_data_to_csv(eda_executor.pre_transform_data, "pre_transform_data.csv")
            print("\nrun_save_data complete..")
            input("\nPress Enter to continue...")

        if run_further_analysis:
            # Carry out more in-depth analysis
            eda_executor.further_analysis(data)
            print("\nrun_further_analysis complete..")
            input("\nPress Enter to Finish...")

    else:
        print("\nNo data was fetched from the database.")

    # Close the database connection
    db_connector.close_connection()
