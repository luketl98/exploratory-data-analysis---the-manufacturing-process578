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

"""
TODO:
1. Ensure plots are generated well, the correct size and legible.
2. Add handling for new data:
    - Follow the process through as if starting from scratch, starting
      by exploring stats & visualising the data and doing nothing else.
    - Essentially, you would be adding an on/off switch for each
      method (stage of EDA) in the EDAExecutor class.
    - So, you the file will only run up to the stage of EDA you are
      currently at. To continue, you would have to 'switch on' the next
      method (stage of EDA) in the EDAExecutor class.
    - ^ Or otherwise clarify with the file that you are ready to progress
      to the next stage of EDA (switch on the next method)
3. Add 'Yeo-Johnson transform done' print statement, or allow user to choose
   which transformation method to use, following preview.
4. Add full comments and Doc strings to all methods
"""


# Function to load database credentials from a YAML file
def load_db_credentials(file_path: str = 'credentials.yaml') -> dict:
    with open(file_path, 'r') as file:
        credentials = yaml.safe_load(file)

    return credentials


# --- Utility Methods ---

def filter_columns(dataframe: pd.DataFrame, dtype, exclude_columns=None):
    """ Helper method to select DataFrame columns by dtype
    and select columns to drop. """
    if exclude_columns is None:
        exclude_columns = []
    return [
        col for col in dataframe.select_dtypes(include=[dtype]).columns
        if col not in exclude_columns
    ]


def save_data_to_csv(data_frame: pd.DataFrame, filename: str):
    """
    Saves a DataFrame to a CSV file.

    Parameters:
        data_frame (pd.DataFrame): The DataFrame to save.
        filename (str): The filename to save the CSV file.
    """
    try:
        data_frame.to_csv(filename, index=False)
        print(f"\nData successfully saved to '{filename}'.")
    except IOError as e:
        print(f"An error occurred while saving to CSV: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Class to handle database connections and operations
class RDSDatabaseConnector:
    def __init__(self, credentials):
        self.engine = create_engine(
            f"postgresql://{credentials['RDS_USER']}:"
            f"{credentials['RDS_PASSWORD']}@"
            f"{credentials['RDS_HOST']}:{credentials['RDS_PORT']}/"
            f"{credentials['RDS_DATABASE']}"
        )

    def fetch_data(self, query: str) -> pd.DataFrame:
        """
        Fetches data from the database by executing the given SQL query.

        Parameters:
            query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: A DataFrame containing the query results.
        """
        try:
            with self.engine.connect() as connection:
                result = pd.read_sql(query, connection)
            return result
        # error handling:
        except SQLAlchemyError as e:
            print(f"An error occurred while executing the query: {e}")
            return pd.DataFrame()  # Return empty df if sql error occurs
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return pd.DataFrame()  # Return empty df for other exceptions

    def close_connection(self):
        # Closes the database connection
        self.engine.dispose()


class DataTransform:
    """Class to transform specific data as needed"""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def convert_to_categorical(self, column_name: str):
        """ Convert the specified column to a categorical data type. """
        self.dataframe[column_name] = self.dataframe[
            column_name].astype('category')

    def convert_to_boolean(self, column_name: str):
        """ Convert the specified column to a boolean data type. """
        self.dataframe[column_name] = self.dataframe[column_name].astype(bool)


class Plotter:
    """Class to visualise insights from the data."""
# TODO: Can this method be refactored at all? any redundant methods or code
    def __init__(self, dataframe):
        self.dataframe = dataframe

    # --- Utility Methods ---

    def _create_subplots(self, num_plots, cols=3, subplot_size=(5, 5)):
        """Helper method to handle subplot creation."""
        rows = (num_plots // cols) + (num_plots % cols > 0)
        plt.figure(figsize=(subplot_size[0] * cols, subplot_size[1] * rows))
        return rows, cols

    # --- Plot Methods ---

    def scatter_multiple_plots(self, column_pairs):
        # TODO: e number being printed on some plots
        """Create scatter plots for multiple column pairs."""
        num_plots = len(column_pairs)
        rows, cols = self._create_subplots(num_plots)

        for i, (x_col, y_col) in enumerate(column_pairs, 1):
            plt.subplot(rows, cols, i)
            plt.scatter(self.dataframe[x_col], self.dataframe[y_col])
            plt.title(f'{x_col} vs {y_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)

        plt.suptitle('Scatter plots for selected column pairs')
        plt.show()

    def plot_histograms(self, exclude_columns=None):
        """
        Generate histograms for all numerical columns,
        excluding specified columns.

        Parameters:
            exclude_columns (list): List of columns to exclude
            from the histograms.
        """
        numeric_cols = filter_columns(self.dataframe, np.number,
                                      exclude_columns=exclude_columns)
        self.dataframe[numeric_cols].hist(bins=20, figsize=(15, 10))
        plt.suptitle('Histograms for Numeric Columns')
        plt.tight_layout()
        plt.show()

    def plot_bar_plots(self, exclude_columns=None):
        """Generate bar plots for all categorical columns."""
        categorical_cols = filter_columns(self.dataframe, 'category',
                                          exclude_columns)
        plt.figure()

        for i, col in enumerate(categorical_cols):
            plt.subplot(1, len(categorical_cols), i + 1)
            self.dataframe[col].value_counts().plot(kind='bar')
            plt.title(col)
        plt.suptitle('Bar plots of all categorical columns')
        plt.tight_layout()
        plt.show()

    def correlation_heatmap(self, include_booleans=False):
        """Generate a heatmap of correlations for numerical and optionally boolean columns."""
        if include_booleans:
            cols_to_include = self.dataframe.select_dtypes(include=[np.number, 'bool'])
        else:
            cols_to_include = self.dataframe.select_dtypes(include=[np.number])

        plt.figure(figsize=(10, 8))  # Adjust the figure size
        sns.heatmap(cols_to_include.corr(), annot=True,
                    cmap='coolwarm', fmt='.2f', cbar_kws={'shrink': .8})
        # Rotate x-axis & y-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title('Correlation Heatmap')
        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()

    def missing_data_matrix(self):
        # TODO: self.dataframe or data? (same for all plotter class)
        """Display a missing data matrix."""
        msno.matrix(self.dataframe, figsize=(14, 6), sparkline=False)
        plt.title('Missing Data Matrix')
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, columns=None, x_column=None, exclude_columns=None):
        """
        Generate boxplots for specified columns, optionally comparing against
        a categorical column (like failure states).

        Parameters:
            columns (list): List of columns to plot. If None, all numeric columns
                            will be used.
            x_column (str): Categorical column for x-axis (e.g., 'Machine failure').
                            If None, boxplots will be created without a categorical comparison.
            exclude_columns (list): List of columns to exclude from the plots.
        """
        # If columns are not provided, default to all numeric columns
        if columns is None:
            columns = filter_columns(self.dataframe, np.number, exclude_columns)

        num_cols = len(columns)

        # Create subplots based on number of columns
        rows, cols = self._create_subplots(num_cols, cols=3, subplot_size=(5, 6))

        # Plot each column
        for i, col in enumerate(columns, 1):
            plt.subplot(rows, cols, i)

            # If x_column is provided, use it for comparison, else plot simple boxplot
            if x_column:
                sns.boxplot(x=x_column, y=self.dataframe[col], data=self.dataframe)
                plt.xlabel(x_column)
            else:
                sns.boxplot(x=self.dataframe[col])
                plt.xlabel('')

            plt.title(f'Boxplot of {col}')

        plt.suptitle('Boxplots for Selected Columns')
        plt.tight_layout()
        plt.show()

    def plot_null_comparison(self, null_counts_before, null_counts_after):
        """Plot comparison of null counts before and after imputation."""
        null_data = pd.DataFrame({
            'Before Imputation': null_counts_before['null_count'],
            'After Imputation': null_counts_after['null_count']
        })

        # Generate the bar plot and handle figure creation directly
        ax = null_data.plot(kind='bar')  # Use 'ax' to manipulate the plot

        # Set the title and labels directly on the ax object
        ax.set_title('Null Count Before and After Imputation')
        ax.set_ylabel('Null Count')

        plt.tight_layout()
        plt.show()

    def plot_skewness(self, exclude_columns=None):
        """Plot histograms for numeric columns with skewness information."""
        numeric_columns = filter_columns(self.dataframe, np.number,
                                         exclude_columns)
        num_plots = len(numeric_columns)
        rows, cols = self._create_subplots(num_plots)

        for i, column in enumerate(numeric_columns, 1):
            skew_value = self.dataframe[column].skew()
            plt.subplot(rows, cols, i)
            self.dataframe[column].hist(bins=50)
            plt.title(f'{column} (Skew: {skew_value:.2f})')

        plt.suptitle('Histograms for numeric columns with Skewness value')
        plt.show()

    def visualise_transformed_column(self, column: str,
                                     original: pd.Series,
                                     transformed: pd.Series):
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
        plt.hist(original, bins=30, color='blue', alpha=0.7)
        plt.title(f'Original {column} (Skew: {original_skew:.2f})')

        # Plot the transformed data
        plt.subplot(1, 2, 2)
        plt.hist(transformed, bins=30, color='green', alpha=0.7)
        plt.title(f'Transformed {column} (Skew: {transformed_skew:.2f})')

        # Add a main title and show the plots
        plt.suptitle(f'Original vs Transformed {column}')
        plt.show()

    def plot_qq(self, exclude_columns=None):
        """Generate Q-Q plots for numeric columns."""
        numeric_columns = filter_columns(self.dataframe, np.number,
                                         exclude_columns)
        num_plots = len(numeric_columns)
        rows, cols = self._create_subplots(num_plots)

        for i, col in enumerate(numeric_columns, 1):
            plt.subplot(rows, cols, i)
            qqplot(self.dataframe[col], line='q', ax=plt.gca())
            plt.title(f"Q-Q plot of {col}")

        plt.suptitle('Q-Q plots for numeric columns')
        plt.show()

        # -- Further Analysis --

    def plot_tool_wear_distribution(self, bins=20):
        """Visualise the distribution of tool wear values with improvements."""
        plt.figure(figsize=(10, 6))

        # Create bins to group similar tool wear values together
        self.dataframe['Tool wear [min]'].plot(
            kind='hist', bins=30, edgecolor='black')

        plt.title('Tool Wear Distribution')
        plt.xlabel('Tool wear [min]')
        plt.ylabel('Number of Tools')

        plt.tight_layout()
        plt.show()

    def plot_tool_wear_by_quality(self):
        """Visualise tool wear distribution by product quality."""
        plt.figure(figsize=(10, 6))

        # Define consistent bins for all product types
        bins = np.histogram_bin_edges(self.dataframe['Tool wear [min]'],
                                      bins=30)

        # Create a histogram for each product type
        for product_type, color in zip(['L', 'M', 'H'],
                                       ['blue', 'orange', 'green']):
            subset = self.dataframe[self.dataframe['Type'] == product_type]
            plt.hist(subset['Tool wear [min]'], bins=bins, alpha=0.5,
                     label=f'Type {product_type}', edgecolor='black')

        plt.title('Tool Wear Distribution by Product Quality')
        plt.xlabel('Tool wear [min]')
        plt.ylabel('Number of Tools')
        # Set x-axis ticks every 10 minutes
        plt.xticks(np.arange(0, 260, 10))  # Adjust range and step as needed
        # Set y-axis ticks every 50
        plt.yticks(np.arange(0, 500, 50))
        plt.legend(title='Product Quality Type')
        plt.tight_layout()
        plt.show()

    def calculate_failure_rate(self):
        """
        Calculate and visualise the total number
        and percentage of failures.
        """
        total_processes = len(self.dataframe)
        total_failures = self.dataframe['Machine failure'].sum()

        failure_percentage = (total_failures / total_processes) * 100

        print(f"\nTotal Processes: {total_processes}")
        print(f"Total Failures: {total_failures}")
        print(f"Failure Percentage: {failure_percentage:.2f}%")

        # Visualise failure rate
        labels = ['Failures', 'Non-failures']
        sizes = [total_failures, total_processes - total_failures]
        plt.pie(sizes, labels=labels, autopct='%1.1f%%',
                startangle=90, colors=['red', 'green'])
        plt.title('Failure Rate in the Manufacturing Process')
        plt.tight_layout()
        plt.show()

    def failures_by_product_quality(self):
        """
        Analyse failures based on product quality types,
        print in terminal & visualise them.
        """
        # Group by product quality and count failures
        failures_by_quality = self.dataframe.groupby(
            'Type')['Machine failure'].sum()
        total_by_quality = self.dataframe.groupby('Type').size()

        failure_percent_by_quality = (
            failures_by_quality / total_by_quality) * 100

        print("\nTotal by Product Quality:\n", total_by_quality)
        print("\nFailures by Product Quality:\n", failures_by_quality)
        print("\nFailure Percentage by Product Quality:\n",
              failure_percent_by_quality)

        # Visualise failures by product quality
        plt.figure(figsize=(8, 6))
        bars = plt.bar(failures_by_quality.index, failures_by_quality.values,
                       color=['blue', 'orange', 'green'], edgecolor='black')

        # Annotate bars with the percentage of total failures
        for bar, percent in zip(bars, failure_percent_by_quality):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval,
                     f'{percent:.1f}%', ha='center', va='bottom')

        plt.title('Failure rate by Product Quality (Type)')
        plt.xlabel('Product Quality (Type)')
        plt.ylabel('Number of Failures')

        # Add the text legend in a box with opacity
        plt.text(0.95, 0.95, '% = failure rate in each quality type',
                 ha='right', va='top', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.6))

        plt.tight_layout()
        plt.show()

    def leading_causes_of_failure(self):
        """Determine, print and visualise the leading causes of failure."""
        failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']

        # Sum the failures for each failure type
        cause_of_failure_counts = self.dataframe[failure_columns].sum()
        total_failures = cause_of_failure_counts.sum()

        print("\nLeading Causes of Failure:\n", cause_of_failure_counts)

        # Calculate percentage of failures for each cause
        failure_percent_by_cause = (
            cause_of_failure_counts / total_failures) * 100

        # Visualise the causes of failure
        plt.figure(figsize=(8, 6))
        bars = plt.bar(cause_of_failure_counts.index,
                       cause_of_failure_counts.values,
                       color='red', edgecolor='black')

        # Annotate bars with the percentage of total failures
        for bar, percent in zip(bars, failure_percent_by_cause):
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval,
                     f'{percent:.1f}%', ha='center', va='bottom')

        plt.title('Leading Causes of Failure')
        plt.xlabel('Failure Type')
        plt.ylabel('Number of Failures')
        plt.tight_layout()
        plt.show()

    def failure_causes_by_product_quality(self):
        """
        Visualise leading causes of failure grouped by
        product quality with percentages.
        """

        failure_columns = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
        failure_data = self.dataframe[self.dataframe['Machine failure'] == 1]

        # Group and prepare data
        failures_grouped = failure_data.groupby(
            'Type')[failure_columns].sum().reset_index()
        failures_melted = failures_grouped.melt(
            id_vars='Type', var_name='Failure Type', value_name='Count'
        )
        # Calculate percentages
        totals = failures_melted.groupby('Type')['Count'].transform('sum')
        failures_melted['Percentage'] = (
            failures_melted['Count'] / totals) * 100

        # Create interactive plot via browser
        fig = px.bar(
            failures_melted,
            x='Type',
            y='Count',
            color='Failure Type',
            title='Failures by Product Quality and Failure Type',
            labels={'Type': 'Product Quality (Type)',
                    'Count': 'Number of Failures'},
            hover_data={'Percentage': ':.1f%'},
        )
        fig.update_layout(barmode='stack',
                          xaxis_title='Product Quality (Type)',
                          yaxis_title='Number of Failures')
        fig.show()


class DataFrameInfo:
    """
    A class to extract and display information from a Pandas DataFrame
    for exploratory data analysis (EDA).
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialise with a DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame to analyse.
        """
        self.dataframe = dataframe

    def describe_columns(
        self, include_columns=None, stats_to_return=None
    ) -> pd.DataFrame:
        """
        Describe selected columns and return specified statistics.

        Parameters:
            include_columns (list): Columns to include in the description.
                                    If None, all columns are included.
            stats_to_return (list): Statistics to return (e.g., 'min', 'max').
                                    If None, all statistics are returned.

        Returns:
            pd.DataFrame: Description of selected columns.
        """
        if include_columns is None:
            include_columns = self.dataframe.columns

        description = self.dataframe[include_columns].describe(include='all')

        if stats_to_return is not None:
            description = description.loc[stats_to_return]

        return description

    def extract_stats(self, exclude_columns=None) -> pd.DataFrame:
        """
        Extract statistical values: median, standard deviation, and mean
        from all numeric columns in the DataFrame, excluding specified columns.

        Parameters:
            exclude_columns (list): List of columns to exclude
            from the statistics.

        Returns:
            pd.DataFrame: A DataFrame containing the mean,
            median, and standard deviation.
        """
        # Use the filter_columns utility to get the numeric columns
        numeric_cols = filter_columns(self.dataframe, np.number,
                                      exclude_columns=exclude_columns)

        statistics = {
            'mean': self.dataframe[numeric_cols].mean(),
            'median': self.dataframe[numeric_cols].median(),
            'std_dev': self.dataframe[numeric_cols].std()
        }
        return pd.DataFrame(statistics)

    def count_distinct_values(self, exclude_columns=None) -> pd.DataFrame:
        """
        Count distinct values in categorical columns of the DataFrame,
        excluding specified columns.

        Parameters:
            exclude_columns (list): A list of columns to exclude from
            the distinct count.

        Returns:
            pd.DataFrame: A DataFrame with the count of distinct values
            for each categorical column.
        """
        if exclude_columns is None:
            exclude_columns = []

        # Use the filter_columns utility to get categorical columns
        categorical_columns = filter_columns(self.dataframe, 'category',
                                             exclude_columns=exclude_columns)

        distinct_values = {
            column: self.dataframe[column].nunique()
            for column in categorical_columns
        }

        return pd.DataFrame.from_dict(distinct_values, orient='index',
                                      columns=['distinct_count'])

    def print_shape(self) -> None:
        """
        Print out the shape of the DataFrame.
        """
        print(f"\nDataFrame shape: {self.dataframe.shape}")

    def count_null_values(self) -> pd.DataFrame:
        """
        Generate a count and percentage of NULL values in each column.

        Returns:
            pd.DataFrame: A DataFrame showing the count and
            percentage of NULL values for each column.
        """
        null_count = self.dataframe.isnull().sum()
        null_percentage = (null_count / len(self.dataframe)) * 100

        null_info = pd.DataFrame({
            'null_count': null_count,
            'null_percentage': null_percentage
        })

        return null_info

    def detect_outliers_zscore(self, column: str, threshold: float = 3.0):
        """
        Detect outliers in a column using the Z-score method.

        Parameters:
            column (str): The column to detect outliers in.
            threshold (float): The Z-score threshold to identify outliers.
        """
        z_scores = stats.zscore(self.dataframe[column])
        outliers = self.dataframe[(
            z_scores > threshold) | (z_scores < -threshold)]
        return outliers

    def detect_outliers_iqr(self, column: str):
        """
        Detect outliers in a column using the IQR method.

        Parameters:
            column (str): The column to detect outliers in.
        """
        Q1 = self.dataframe[column].quantile(0.25)
        Q3 = self.dataframe[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = self.dataframe[(self.dataframe[column] < lower_bound) |
                                  (self.dataframe[column] > upper_bound)]
        return outliers


class DataFrameTransform:
    """Class to perform EDA transformations on the DataFrame."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def knn_impute(self, target_column, correlated_columns, n_neighbors=5):
        """
        Apply KNN imputation to the target column using only the specified
        correlated columns.

        Parameters:
            target_column (str): The name of the column to impute.
            correlated_columns (list): List of correlated columns to use for
                                       KNN imputation.
            n_neighbors (int): The number of neighbors to consider
                               for imputation.
        """
        # Select the target column along with the correlated columns
        columns_to_use = correlated_columns + [target_column]

        knn_data = self.dataframe[columns_to_use]

        # Perform KNN imputation on these columns
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = knn_imputer.fit_transform(knn_data)

        # Replace the target column with the imputed values
        # (last column is target)
        self.dataframe[target_column] = imputed_data[:, -1]

        print(f"\nKNN imputation applied to {target_column}"
              f"using {correlated_columns}")

    def impute_missing_values(self, strategies=None, knn_columns=None):
        """
        Impute missing numeric values in the DataFrame based on the provided
        strategy.

        Parameters:
            strategies : dict, optional
                A dictionary where the key is the column name, and the value is
                the imputation strategy ('mean', 'median', 'knn').
            knn_columns : dict, optional
                Dictionary where the key is the column to impute, and the value
                is the list of correlated columns to use for KNN imputation.
        """
        # Select only numeric columns
        numeric_columns = self.dataframe.select_dtypes(include=[np.number])

        # Find numeric columns with null values
        numeric_columns_with_nulls = numeric_columns.loc[
            :, numeric_columns.isnull().any()
        ]

        # Default to median strategy if none is provided
        if strategies is None:
            strategies = {col: 'median' for col in numeric_columns_with_nulls}

        # Iterate through the provided strategies
        for column, strategy in strategies.items():
            if column in numeric_columns_with_nulls:
                if strategy == 'mean':
                    # Impute using the mean
                    self.dataframe[column].fillna(
                        self.dataframe[column].mean(), inplace=True
                    )
                elif strategy == 'median':
                    # Impute using the median
                    self.dataframe[column].fillna(
                        self.dataframe[column].median(), inplace=True
                    )
                elif (strategy == 'knn' and knn_columns
                        and column in knn_columns):
                    # Apply KNN imputation for the specified column
                    self.knn_impute(column, knn_columns[column])

    def apply_yeo_johnson(self, column: str):
        """
        Apply Yeo-Johnson transformation to the specified column and
        replace the original column with the transformed values.

        Parameters:
            column (str): The name of the column to transform.
        """
        # Perform Yeo-Johnson transformation
        transformed_column, _ = stats.yeojohnson(self.dataframe[column])

        # Update the DataFrame with the transformed values
        self.dataframe[column] = transformed_column

        print(f'Yeo-Johnson transformation applied to {column}')

    def preview_transformations(self, column: str):
        """
        Apply Log, Box-Cox, and Yeo-Johnson transformations to the specified
        column. Print skewness post-transformation and plot the distributions.
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
        plt.hist(log_transformed, bins=30, color='blue')
        plt.title(f'Log Transform (Skew: {log_skew:.2f})')

        plt.subplot(1, 3, 2)
        plt.hist(boxcox_transformed, bins=30, color='green')
        plt.title(f'Box-Cox Transform (Skew: {boxcox_skew:.2f})')

        plt.subplot(1, 3, 3)
        plt.hist(yeo_transformed, bins=30, color='red')
        plt.title(f'Yeo-Johnson Transform (Skew: {yeo_skew:.2f})')

        plt.suptitle(f'Preview transformations of {column}')
        plt.show()

    def save_transformed_data(self, filename: str = 'transformed_data.csv'):
        # TODO: Is this method necessary?
        """Save the transformed DataFrame to a new CSV file."""
        save_data_to_csv(self.dataframe, filename)

    def drop_columns(self, columns_to_drop: list):
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


class EDAExecutor:
    """Class to run all EDA, visualisation, and transformations."""

    def __init__(self, db_connector):
        """Initialise with db_connector."""
        # TODO: All other classes initialise 'self.dataframe' not 'self.data'
        self.data = None
        self.pre_transform_data = None
        self.db_connector = db_connector

    def fetch_and_save_data(self, query):
        """Fetch data from database and save it to CSV."""
        data = self.db_connector.fetch_data(query)
        if not data.empty:
            csv_filename = 'failure_data.csv'
            save_data_to_csv(data, csv_filename)
            print("\nData successfully retrieved from database",
                  f"and saved to '{csv_filename}'.")
        return data

    def reformat_data(self, data):
        """
        Reformat data to ensure correct column formats, in this instance:
        - Converting datatypes of specified columns.

         Parameters:
            data (pd.DataFrame): The DataFrame to reformat.

        Returns:
            pd.DataFrame: The reformatted DataFrame.
        """
        transformer = DataTransform(data)

        # Convert 'Type' column to categorical
        transformer.convert_to_categorical('Type')

        # Convert failure columns to boolean
        for col in ['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
            transformer.convert_to_boolean(col)

        print('\nReformatting of data complete..')

        return data  # Return the transformed data

    def explore_stats(self, data):
        """Run basic statistical exploration on the data, print to terminal."""
        print("\nprinting statistics...")
        df_info = DataFrameInfo(data)

        # List which columns to exclude
        extract_stats_exclude = ['UDI']
        count_distinct_exclude = ['Product ID']

        # Print statistics
        print("\nColumn Descriptions:\n", df_info.describe_columns())
        print("\nExtracted Statistics:\n",
              df_info.extract_stats(exclude_columns=extract_stats_exclude))

        print("\nDistinct Value Counts:\n", df_info.count_distinct_values(
            exclude_columns=count_distinct_exclude))
        df_info.print_shape()
        print("\nNull Value Counts:\n", df_info.count_null_values())

        print('\nexploration of stats complete..')

    def visualise_data(self, data):
        """Generate visualisations for data."""
        print("\nGenerating visualisations...")
        plotter = Plotter(data)

        # Data selections
        scatter_plot_column_pairs = [
            ('Air temperature [K]', 'Process temperature [K]'),
            ('Rotational speed [rpm]', 'Torque [Nm]'),
            ('Tool wear [min]', 'Rotational speed [rpm]'),
            ('Tool wear [min]', 'Process temperature [K]'),
        ]

        # Call plots
        plotter.scatter_multiple_plots(scatter_plot_column_pairs)
        plotter.plot_histograms(exclude_columns='UDI')
        plotter.plot_bar_plots(exclude_columns='Product ID')
        plotter.correlation_heatmap()
        plotter.missing_data_matrix()
        plotter.plot_boxplots(exclude_columns='UDI')
        plotter.plot_skewness(exclude_columns='UDI')
        plotter.plot_qq(exclude_columns='UDI')
        print('\nVisualisation complete..')

    def run_imputation_and_null_visualisation(self, data, knn_columns=None,
                                              visualisations_on=True):
        """Handle null imputation and optionally
        visualise null count comparison."""
        # TODO: keep knn_columns param?
        df_transform = DataFrameTransform(data)
        df_info = DataFrameInfo(data)
        plotter = Plotter(data)

        # Specify imputation strategy
        imputation_strategy = {
            'Air temperature [K]': 'mean',
            'Process temperature [K]': 'knn',
            'Tool wear [min]': 'median'
        }

        # Define the columns for KNN imputation
        knn_cols = {
            'Process temperature [K]': ['Air temperature [K]']
        }

        # Null count before and after imputation
        initial_null_count = df_info.count_null_values()

        # Pass the strategy and KNN columns to impute_missing_values
        df_transform.impute_missing_values(strategies=imputation_strategy,
                                           knn_columns=knn_cols)

        null_count_after_impute = df_info.count_null_values()
        print("\nPost impute null value counts:\n", null_count_after_impute)

        # Visualise null count comparison (if visualisation flag is True)
        if visualisations_on:
            plotter.plot_null_comparison(initial_null_count,
                                         null_count_after_impute)

        print('\nRun null imputation complete..')

    def handle_skewness_and_transformations(self, data,
                                            visualisations_on=True):
        """Handle skewness detection and transformation of columns."""
        df_info = DataFrameInfo(data)
        df_transform = DataFrameTransform(data)
        plotter = Plotter(data)

        # Save the current dataframe to self.pre_transform_data before transformation
        if self.pre_transform_data is None:
            self.pre_transform_data = data.copy()  # Save the whole dataframe pre-transform

        # Preview and visualise transformation (if visualisation flag is True)
        if visualisations_on:
            df_transform.preview_transformations('Rotational speed [rpm]')

        # Retrieve 'Rotational speed [rpm]' column from pre-transform data
        original_data = self.pre_transform_data['Rotational speed [rpm]']

        # Perform Yeo-Johnson transformation on 'Rotational speed [rpm]'
        yeo_transformed_data, _ = stats.yeojohnson(
            df_transform.dataframe['Rotational speed [rpm]'])

        # Update the dataframe with the transformed data
        df_transform.dataframe['Rotational speed [rpm]'] = yeo_transformed_data

        # Visualise transformation (if visualisation flag is True)
        if visualisations_on:
            plotter.visualise_transformed_column(
                column='Rotational speed [rpm]',
                original=original_data,
                transformed=yeo_transformed_data
            )

        print('\nRun_skewness_transformations complete..')

    def handle_outlier_detection(self, data):
        """Detect and handle outliers in the data."""
        df_info = DataFrameInfo(data)
        # TODO: Only detects outliers in 'Rotational speed [rpm]
        print("\nDetecting Z-score Outliers:")
        zscore_outliers = df_info.detect_outliers_zscore(
            'Rotational speed [rpm]')
        print(zscore_outliers)

        print("\nDetecting IQR Outliers:")
        iqr_outliers = df_info.detect_outliers_iqr('Rotational speed [rpm]')
        print(iqr_outliers)

        # Visualise outliers
        plotter = Plotter(data)
        plotter.scatter_multiple_plots([
            ('Air temperature [K]', 'Process temperature [K]'),
            ('Rotational speed [rpm]', 'Torque [Nm]'),
            ('Tool wear [min]', 'Rotational speed [rpm]')
        ])
        plotter.plot_boxplots(exclude_columns='UDI')

        print('Outlier detection complete..')

    # -- Further Analysis --

    # Task 1: Operating Ranges Analysis
    def analyse_operating_ranges(self, data):
        """Analyse and display operating ranges across product types."""
        selected_columns = [
            'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
        ]

        # Ensure pre_transform_data is available
        if self.pre_transform_data is None:
            raise ValueError("Pre-transformation data is not available.")

        # Display overall operating ranges
        print("\nOverall Operating Ranges:")
        df_info = DataFrameInfo(self.pre_transform_data)
        overall_ranges = df_info.describe_columns(
            include_columns=selected_columns,
            stats_to_return=['min', '25%', '75%', 'max', 'mean']
        )
        print(overall_ranges)

        # Loop through each product quality type and display the stats
        for product_type in ['L', 'M', 'H']:
            print(f"\nOperating Ranges for Product Type: {product_type}")

            filtered_data = self.pre_transform_data[
                self.pre_transform_data['Type'] == product_type]

            df_info_filtered = DataFrameInfo(filtered_data)

            type_specific_ranges = df_info_filtered.describe_columns(
                include_columns=selected_columns,
                stats_to_return=['min', '25%', '75%', 'max', 'mean']
            )
            print(type_specific_ranges)

    # Task 2: Failures Analysis
    def analyse_failures(self, data):
        """Analyse failures by product quality and failure type."""
        plotter = Plotter(data)
        plotter.failures_by_product_quality()
        plotter.leading_causes_of_failure()
        plotter.failure_causes_by_product_quality()

    # Task 3: Deeper Understanding of Failures
    def analyse_failure_risk_factors(self, data):
        """
        Investigate potential risk factors for machine failures.

        This method explores whether certain machine settings (e.g., torque,
        temperatures, rpm) are correlated with the different types of failures.
        The aim is to identify specific conditions that lead to increased
        failure rates.

        Steps:
        - # TODO: fill in these steps
        """
        plotter = Plotter(data)

        # For failure risk factors with failure comparison
        # TODO: Can these two plot_boxplots be merged?
        plotter.plot_boxplots(
            columns=['Torque [Nm]', 'Process temperature [K]',
                     'Rotational speed [rpm]', 'Air temperature [K]',
                     'Tool wear [min]'],
            x_column='Machine failure')

        plotter.plot_boxplots(
            columns=['Torque [Nm]', 'Process temperature [K]',
                     'Rotational speed [rpm]', 'Air temperature [K]',
                     'Tool wear [min]'],
            x_column='HDF')

        plotter.plot_boxplots(
            columns=['Torque [Nm]', 'Process temperature [K]',
                     'Rotational speed [rpm]', 'Air temperature [K]',
                     'Tool wear [min]'],
            x_column='OSF')

        plotter.correlation_heatmap(include_booleans=True)

        column_pairs = [('Torque [Nm]', 'Process temperature [K]'),
                        ('Air temperature [K]', 'Process temperature [K]'),
                        ('Torque [Nm]', 'HDF'),
                        ('Process temperature [K]', 'HDF'),
                        ('Rotational speed [rpm]', 'PWF')]

        plotter.scatter_multiple_plots(column_pairs)

        selected_columns = [
            'Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]'
        ]

        # Ensure pre_transform_data is available
        if self.pre_transform_data is None:
            raise ValueError("Pre-transformation data is not available.")

        # Loop through each failure type (HDF, OSF, PWF) and True/False statuses
        for failure_type in ['HDF', 'OSF', 'PWF']:
            for failure_status in [True, False]:
                print(f"\nOperating Ranges for {failure_type} = {failure_status}")

                # Filter the pre-transform data based on the failure status
                filtered_data = self.pre_transform_data[
                    self.pre_transform_data[failure_type] == failure_status]

                # Create a DataFrameInfo object for the filtered data
                df_info_filtered = DataFrameInfo(filtered_data)

                # Call the describe_columns method for the filtered dataset
                failure_specific_ranges = df_info_filtered.describe_columns(
                    include_columns=selected_columns,
                    stats_to_return=['min', '25%', '75%', 'max', 'mean']
                )
                print(failure_specific_ranges)

    # Further_analysis umbrella method
    def further_analysis(self, data):
        """Conduct further analysis by calling task-specific methods."""
        self.analyse_operating_ranges(data)
        self.analyse_failures(data)
        self.analyse_failure_risk_factors(data)


if __name__ == "__main__":
    # Flag control system:
    # Each flag corresponds to a different step in the EDA process.
    # Set a flag to True to include that step, or False to skip it.

    run_reformat = True  # Reformat data (e.g., column types, categories)
    run_explore_stats = True  # Explore statistics
    run_visualisation = False  # Generate visualisations for data
    run_null_imputation = True  # Carry out null imputation & visualisation
    run_skewness_transformations = True  # Preview & perform transformation
    run_outlier_detection = False  # Detect and visualise outliers
    run_drop_columns = False  # Drop columns after analysis (if applicable)
    run_save_data = True  # Save transformed data
    run_further_analysis = True  # Carry out more in-depth analysis

    # Load database credentials and connect
    credentials = load_db_credentials('credentials.yaml')
    db_connector = RDSDatabaseConnector(credentials)

    # Create an instance of EDAExecutor & df transform
    eda_executor = EDAExecutor(db_connector)

    # Fetch and save data
    data = eda_executor.fetch_and_save_data(
        query="SELECT * FROM failure_data;"
    )

    # Create an instance of df transform
    df_transform = DataFrameTransform(data)

    if not data.empty:
        if run_reformat:
            # Reformat data as needed (ensure correct formats, types, etc.)
            data = eda_executor.reformat_data(data)

        if run_explore_stats:
            # Perform initial exploration of data
            eda_executor.explore_stats(data)
            # input("\nPress Enter to continue...")

        if run_visualisation:
            # Perform visualisation of data
            eda_executor.visualise_data(data)
            # input("\nPress Enter to continue...")

        if run_null_imputation:
            # Perform null imputation/removal and visualise the result
            eda_executor.run_imputation_and_null_visualisation(
                data, visualisations_on=False  # visualisations on/off
                )
            # input("\nPress Enter to continue...")

        if run_skewness_transformations:
            # Skewness and transformations
            eda_executor.handle_skewness_and_transformations(
                data, visualisations_on=False  # visualisations on/off
                )
            # input("\nPress Enter to continue...")

        if run_outlier_detection:
            # Outlier detection - Currently does not handle outliers
            eda_executor.handle_outlier_detection(data)
            # input("\nPress Enter to continue...")

        if run_drop_columns:
            # Drop columns after analysis (if applicable)
            columns_to_drop = ['Air temperature [K]', 'Rotational speed [rpm]']
            df_transform.drop_columns(columns_to_drop)
            # input("\nPress Enter to continue...")

        if run_save_data:
            # Save the transformed data
            df_transform.save_transformed_data('transformed_failure_data.csv')
            # input("\nPress Enter to continue...")

        if run_further_analysis:
            # Carry out more in-depth analysis
            eda_executor.further_analysis(data)
            # input("\nPress Enter to continue...")

    else:
        print("\nNo data was fetched from the database.")

    # Close the database connection
    db_connector.close_connection()
