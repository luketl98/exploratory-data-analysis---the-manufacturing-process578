import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from sklearn.impute import KNNImputer
from statsmodels.graphics.gofplots import qqplot
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError


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

    def save_data_to_csv(self, data_frame: pd.DataFrame, filename: str):
        """Wrapper for saving data in RDSDatabaseConnector."""
        save_data_to_csv(data_frame, filename)  # TODO: Remove this method?

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

    def __init__(self, dataframe):
        self.dataframe = dataframe

    # --- Utility Methods ---

    def _create_plot(self, title, xlabel=None, ylabel=None, figsize=(8, 6)):
        """Helper method to standardise plot creation."""
        plt.figure(figsize=figsize)
        plt.title(title)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    def _create_subplots(self, num_plots, cols=3, subplot_size=(5, 5)):
        """Helper method to handle subplot creation."""
        rows = (num_plots // cols) + (num_plots % cols > 0)
        plt.figure(figsize=(subplot_size[0] * cols, subplot_size[1] * rows))
        return rows, cols

    # --- Plot Methods ---

    def scatter_plot(self, x_column, y_column):
        """Scatter Plot: To identify outliers."""
        self._create_plot(
            f'Scatter Plot: {x_column} vs {y_column}', x_column, y_column
        )
        plt.scatter(self.dataframe[x_column], self.dataframe[y_column])

    def scatter_multiple_plots(self, column_pairs):
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

    def correlation_heatmap(self):
        """Generate a heatmap of correlations for numerical columns."""
        numeric_cols = self.dataframe.select_dtypes(include=[np.number])
        sns.heatmap(numeric_cols.corr(), annot=True,
                    cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()

    def missing_data_matrix(self):
        """Display a missing data matrix."""
        msno.matrix(self.dataframe, figsize=(14, 6), sparkline=False)
        plt.title('Missing Data Matrix')
        plt.tight_layout()
        plt.show()

    def plot_boxplots(self, exclude_columns=None):
        """Generate boxplots for all numeric columns in one figure."""
        numeric_columns = filter_columns(self.dataframe, np.number,
                                         exclude_columns)
        num_cols = len(numeric_columns)

        rows, cols = self._create_subplots(num_cols, cols=3,
                                           subplot_size=(5, 6))

        for i, col in enumerate(numeric_columns, 1):
            plt.subplot(rows, cols, i)
            sns.boxplot(x=self.dataframe[col])
            plt.title(f'Boxplot of {col}')

        plt.suptitle('Boxplots for numeric columns')
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

    def describe_columns(self) -> pd.DataFrame:
        """Describe all columns in the DataFrame to check their data types."""
        with pd.option_context('display.max_columns', None):
            return self.dataframe.describe(include='all')

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

    def count_null_values(self, exclude_columns=None) -> pd.DataFrame:
        """
        Generate a count and percentage of NULL values in each column,
        excluding specified columns.

        Parameters:
            exclude_columns (list): A list of columns to exclude from
            the null count.

        Returns:
            pd.DataFrame: A DataFrame showing the count and
            percentage of NULL values for each column.
        """
        if exclude_columns is None:
            exclude_columns = []
            # TODO: need to be able to exclude columns here?

        df_filtered = self.dataframe.drop(columns=exclude_columns,
                                          errors='ignore')

        null_count = df_filtered.isnull().sum()
        null_percentage = (null_count / len(df_filtered)) * 100

        null_info = pd.DataFrame({
            'null_count': null_count,
            'null_percentage': null_percentage
        })

        return null_info

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

    def drop_columns_with_nulls(self, threshold: float = 0.5):
        # TODO: use drop_columns method here
        """
        Drop columns where the percentage of missing values
        exceeds the given threshold.

        Parameters:
            threshold (float): The maximum percentage of null values
            allowed (default is 50%) - To adjust: Change threshold.
        """
        # Calculate the percentage of null values in each column
        null_percentage = self.dataframe.isnull().mean()

        # Drop columns exceeding the threshold
        columns_to_drop = null_percentage[null_percentage > threshold].index
        self.dataframe.drop(columns=columns_to_drop, inplace=True)

        return columns_to_drop

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

        print(f"KNN imputation applied to {target_column}"
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
        self.data = None
        self.db_connector = db_connector

    def fetch_and_save_data(self, query):
        """Fetch data from database and save it to CSV."""
        data = self.db_connector.fetch_data(query)
        if not data.empty:
            csv_filename = 'failure_data.csv'
            self.db_connector.save_data_to_csv(data, csv_filename)
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

        return data  # Return the transformed data

    def explore_stats(self, data):
        """Run basic statistical exploration on the data, print to terminal."""
        print("\nprinting statistics...")
        df_info = DataFrameInfo(data)

        # List which columns to exclude
        extract_stats_exclude = ['UDI']
        count_distinct_exclude = ['Product ID']
        count_null_exclude = ['UDI', 'Product ID']
        # TODO: ^ Should data be excluded from null count?

        # Print statistics
        print("\nColumn Descriptions:\n", df_info.describe_columns())
        print("\nExtracted Statistics:\n",
              df_info.extract_stats(exclude_columns=extract_stats_exclude))

        print("\nDistinct Value Counts:\n", df_info.count_distinct_values(
            exclude_columns=count_distinct_exclude))
        df_info.print_shape()
        print("\nNull Value Counts:\n", df_info.count_null_values(
            exclude_columns=count_null_exclude))

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
            ('Tool wear [min]', 'Machine failure')
        ]
        exclude_from_bar_plot = ['Product ID']

        # Call plots
        plotter.scatter_multiple_plots(scatter_plot_column_pairs)
        plotter.plot_histograms(exclude_columns='UDI')
        plotter.plot_bar_plots(exclude_columns=exclude_from_bar_plot)
        plotter.correlation_heatmap()
        plotter.missing_data_matrix()
        plotter.plot_boxplots(exclude_columns='UDI')
        plotter.plot_skewness(exclude_columns='UDI')
        plotter.plot_qq(exclude_columns='UDI')
        print('\nVisualisation complete.')

    def run_imputation_and_null_visualisation(self, data, knn_columns=None):
        """Handle null imputation and visualisation
        of null count comparison."""
        df_transform = DataFrameTransform(data)
        df_info = DataFrameInfo(data)
        plotter = Plotter(data)

        # Specify imputation strategy
        imputation_strategy = {
            'Air temperature [K]': 'mean',
            'Process temperature [K]': 'knn',
            'Tool wear [min]': 'median'
        }

        # Define the columns for KNN imputation (Process Temp uses Air Temp)
        knn_cols = {
            'Process temperature [K]': ['Air temperature [K]']
        }

        # Null count before and after imputation
        initial_null_count = df_info.count_null_values()

        # Pass the strategy and KNN columns to impute_missing_values
        df_transform.impute_missing_values(strategies=imputation_strategy,
                                           knn_columns=knn_cols)

        post_impute_null_count = df_info.count_null_values()
        print("\nPost impute null value counts:\n", post_impute_null_count)

        # Visualise null count comparison
        plotter.plot_null_comparison(initial_null_count,
                                     post_impute_null_count)

    def handle_skewness_and_transformations(self, data):
        """Handle skewness detection and transformation of columns."""
        df_info = DataFrameInfo(data)
        df_transform = DataFrameTransform(data)

        # Preview and visualise transformation
        df_transform.preview_transformations('Rotational speed [rpm]')
        original_data = df_info.dataframe['Rotational speed [rpm]']
        yeo_transformed_data, _ = stats.yeojohnson(
            df_transform.dataframe['Rotational speed [rpm]'])

        # Update and visualise transformation
        df_transform.dataframe['Rotational speed [rpm]'] = yeo_transformed_data
        df_info.visualise_transformed_column(
            column='Rotational speed [rpm]',
            original=original_data,
            transformed=yeo_transformed_data
        )

    def handle_outlier_detection(self, data):
        """Detect and handle outliers in the data."""
        df_info = DataFrameInfo(data)
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


if __name__ == "__main__":
    # Load database credentials and connect
    credentials = load_db_credentials('credentials.yaml')
    db_connector = RDSDatabaseConnector(credentials)

    # Create an instance of EDAExecutor
    eda_executor = EDAExecutor(db_connector)

    # Fetch and save data
    data = eda_executor.fetch_and_save_data(
        query="SELECT * FROM failure_data;"
        )

    if not data.empty:
        # Reformat data as needed (ensure correct formats, types, etc.)
        data = eda_executor.reformat_data(data)

        # Perform initial exploration of data
        eda_executor.explore_stats(data)
        input("\nPress Enter to continue to visualisation...")

        # Perform visualisation of data
        eda_executor.visualise_data(data)
        input("\nPress Enter to continue to null handling...")

        # Perform null imputation/removal and visualise the result
        eda_executor.run_imputation_and_null_visualisation(data)
        input("\nPress Enter to continue to skewness & transformations...")

        # Skewness and transformations
        eda_executor.handle_skewness_and_transformations(data)
        input("\nPress Enter to continue to outlier handling...")

        # Outlier detection
        eda_executor.handle_outlier_detection(data)

        # Save the transformed data
        df_transform = DataFrameTransform(data)
        df_transform.save_transformed_data('transformed_failure_data.csv')

        # Drop columns after analysis
        columns_to_drop = ['Air temperature [K]']  # <-- High collinearity
        df_transform.drop_columns(columns_to_drop)

    else:
        print("\nNo data was fetched from the database.")

    # Close the database connection
    db_connector.close_connection()
