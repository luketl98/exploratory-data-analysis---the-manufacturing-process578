import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sqlalchemy import create_engine


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    # Load the data into a DataFrame
    data = pd.read_csv(file_path)

    return data


# Function to load database credentials from a YAML file
def load_db_credentials(file_path: str = 'credentials.yaml') -> dict:
    with open(file_path, 'r') as file:
        credentials = yaml.safe_load(file)

    return credentials


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
        with self.engine.connect() as connection:
            result = pd.read_sql(query, connection)

        return result

    def save_data_to_csv(self, data_frame, filename):
        # Saves a DataFrame to a CSV file
        data_frame.to_csv(filename, index=False)

    def close_connection(self):
        # Closes the database connection
        self.engine.dispose()


class DataTransform:
    """Class to transform the data as needed"""

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
    """ --- Class to visualise insights from the data --- """

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def scatter_plot(self, x_column, y_column):
        """ Scatter Plot: To identify outliers """
        plt.figure(figsize=(8, 6))
        plt.scatter(self.dataframe[x_column], self.dataframe[y_column])
        plt.title(f'Scatter Plot: {x_column} vs {y_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()

    def scatter_multiple_plots(self, column_pairs):
        num_plots = len(column_pairs)
        cols = 3  # Set number of columns

        # Dynamically calculate rows based on the number of plots and columns
        rows = (num_plots // cols) + (num_plots % cols > 0)

        # Adjust figure size based on number of rows and columns
        plt.figure(figsize=(5 * cols, 5 * rows))

        for i, (x_col, y_col) in enumerate(column_pairs, 1):
            plt.subplot(rows, cols, i)  # Create subplot with dynamic row count
            plt.scatter(self.dataframe[x_col], self.dataframe[y_col])
            plt.title(f'{x_col} vs {y_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)

        plt.tight_layout()  # Ensures spacing between plots
        plt.show()

    def plot_histograms(self):
        """
        Generate histograms for all numerical columns, excluding specific ones.
        """
        numeric_cols = self.dataframe.select_dtypes(
            include=[np.number]).columns
        cols_to_exclude = ['UDI']  # Columns to exclude
        numeric_cols = [
            col for col in numeric_cols if col not in cols_to_exclude
        ]
        self.dataframe[numeric_cols].hist(bins=20, figsize=(15, 10))
        plt.tight_layout()
        plt.show()

    def plot_bar_plots(self):
        """
        Generate bar plots for all categorical columns,
        excluding specific ones.
        """
        categorical_cols = self.dataframe.select_dtypes(
            include=['category', 'object']).columns
        cols_to_exclude = ['Product ID']  # Specify columns to exclude
        categorical_cols = [
            col for col in categorical_cols if col not in cols_to_exclude
        ]

        for col in categorical_cols:
            plt.figure(figsize=(8, 6))
            self.dataframe[col].value_counts().plot(kind='bar')
            plt.title(f'Bar Plot of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()

    def correlation_heatmap(self):
        """
        Generate a heatmap of correlations for numerical columns.
        """
        plt.figure(figsize=(10, 8))
        numeric_cols = self.dataframe.select_dtypes(include=[np.number])

        sns.heatmap(numeric_cols.corr(), annot=True,
                    cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.tight_layout()
        plt.show()

    def missing_data_matrix(self):
        """ Missing Data Matrix (via missingno) """
        msno.matrix(self.dataframe, figsize=(14, 6), sparkline=False)
        plt.show()

    def plot_boxplots(self):
        """ Boxplots for all numeric columns in one figure """
        numeric_columns = self.dataframe.select_dtypes(
            include=[np.number]).columns
        num_cols = len(numeric_columns)

        # Adjust rows and columns based on the number of plots
        cols = 3
        rows = (num_cols // cols) + (num_cols % cols > 0)

        plt.figure(figsize=(12, 6 * rows))

        for i, col in enumerate(numeric_columns, 1):
            plt.subplot(rows, cols, i)
            sns.boxplot(x=self.dataframe[col])
            plt.title(f'Boxplot of {col}')

        plt.tight_layout()
        plt.show()

    def plot_null_comparison(self, null_counts_before, null_counts_after):
        """Plot to compare null counts before and after imputation. """

        # Create a DataFrame with before and after null counts
        null_data = pd.DataFrame({
            'Before Imputation': null_counts_before['null_count'],
            'After Imputation': null_counts_after['null_count']
        })

        # Plot the data
        null_data.plot(kind='bar')
        plt.title('Null Count Before and After Imputation')
        plt.ylabel('Null Count')
        plt.tight_layout()
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
        """
        Describe all columns in the DataFrame to check their data types.

        Returns:
            pd.DataFrame: Summary statistics of all DataFrame columns.
        """
        # Display all columns in the DataFrame
        pd.set_option('display.max_columns', None)

        # Return description of all columns, including non-numeric ones
        return self.dataframe.describe(include='all')

    def extract_statistics(self) -> pd.DataFrame:
        """
        Extract statistical values: median, standard deviation,
        and mean from all numeric columns in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the mean, median,
            and standard deviation for each numeric column.
        """
        # Select only the numerical columns
        numerical_columns = self.dataframe.select_dtypes(include=[np.number])

        # Exclude the 'UDI' column
        numerical_columns = numerical_columns.drop(columns=['UDI'],
                                                   errors='ignore')

        statistics = {
            'mean': numerical_columns.mean(),
            'median': numerical_columns.median(),
            'std_dev': numerical_columns.std()
        }
        return pd.DataFrame(statistics)

    def count_distinct_values(self) -> pd.DataFrame:
        """
        Count distinct values in categorical columns of the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with the count of distinct values
            for each categorical column.
        """
        categorical_columns = self.dataframe.select_dtypes(
            include=['object', 'category', 'bool']
            ).columns
        distinct_values = {
            column: self.dataframe[column].nunique()
            for column in categorical_columns if column != 'Product ID'
        }
        return pd.DataFrame.from_dict(distinct_values,
                                      orient='index',
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
            pd.DataFrame: A DataFrame showing the count and percentage of
            NULL values for each column, excluding 'UDI' and 'Product ID'.
        """
        # Exclude 'UDI' and 'Product ID' columns
        df_filtered = self.dataframe.drop(columns=['UDI', 'Product ID'])

        null_count = df_filtered.isnull().sum()
        null_percentage = (null_count / len(df_filtered)) * 100

        null_info = pd.DataFrame({
            'null_count': null_count,
            'null_percentage': null_percentage
        })
        return null_info


class DataFrameTransform:
    """Class to perform EDA transformations on the DataFrame."""

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def drop_columns_with_nulls(self, threshold: float = 0.5):
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

    def impute_missing_values(self, strategies=None):
        """
        Impute missing numeric values in the DataFrame
        based on the provided strategy.

        Parameters:
            strategies : Best to use a dictionary in main execution block
            where the key is the column name, and the value is the imputation
            strategy ('mean' or 'median').
        """
        # Select numeric columns
        numeric_columns = self.dataframe.select_dtypes(include=[np.number])

        # Find numeric columns with null values
        numeric_columns_with_nulls = numeric_columns.loc[
            :, numeric_columns.isnull().any()
        ]

        if strategies is None:
            strategies = {col: 'median' for col in numeric_columns_with_nulls}

        for column, strategy in strategies.items():
            if column in numeric_columns_with_nulls:
                if strategy == 'mean':
                    self.dataframe[column].fillna(
                        self.dataframe[column].mean(), inplace=True)
                elif strategy == 'median':
                    self.dataframe[column].fillna(
                        self.dataframe[column].median(), inplace=True)


if __name__ == "__main__":
    """
    Main execution block to load data, transform it,
    perform EDA, and visualise results.
    """

    # Load database credentials
    credentials = load_db_credentials('credentials.yaml')
    db_connector = RDSDatabaseConnector(credentials)

    # Define and execute SQL query to fetch data
    query = "SELECT * FROM failure_data;"
    data = db_connector.fetch_data(query)

    if not data.empty:
        csv_filename = 'failure_data.csv'
        db_connector.save_data_to_csv(data, csv_filename)
        print(f"\nData successfully saved to '{csv_filename}'.")

        print("\nLoading data from CSV to verify contents:")
        loaded_data = load_data_from_csv(csv_filename)

        # Initialize DataTransform class for transformations
        transformer = DataTransform(loaded_data)

        # Convert 'Type' to categorical and failure indicators to boolean
        transformer.convert_to_categorical('Type')
        failure_columns = ['Machine failure', 'TWF',
                           'HDF', 'PWF', 'OSF', 'RNF']
        for col in failure_columns:
            transformer.convert_to_boolean(col)

        # DataFrameInfo for basic EDA insights
        data_info = DataFrameInfo(loaded_data)
        print("\nColumn Descriptions:\n", data_info.describe_columns())
        print("\nExtracted Statistics:\n", data_info.extract_statistics())
        print("\nDistinct Value Counts:\n", data_info.count_distinct_values())
        data_info.print_shape()

        # Null count before imputation:
        initial_null_count = data_info.count_null_values()
        print("\nNull Value Counts:\n", initial_null_count)

        # Plotter class for visualisations
        plotter = Plotter(loaded_data)
        print("\nGenerating visualisations...")

        column_pairs = [
            ('Air temperature [K]', 'Process temperature [K]'),
            ('Rotational speed [rpm]', 'Torque [Nm]'),
            ('Tool wear [min]', 'Rotational speed [rpm]'),
            ('Tool wear [min]', 'Process temperature [K]'),
            ('Tool wear [min]', 'Machine failure')
        ]
        plotter.scatter_multiple_plots(column_pairs)
        plotter.plot_histograms()
        plotter.plot_bar_plots()
        plotter.correlation_heatmap()
        plotter.missing_data_matrix()
        plotter.plot_boxplots()
        print('Visualisation complete.')

        # DataFrameTransform for null imputation and checking results
        df_transform = DataFrameTransform(loaded_data)

        # Drop columns with excessive nulls (>50%)
        df_transform.drop_columns_with_nulls()

        # Impute null values with the specified strategies
        imputation_strategy = {
            'Air temperature [K]': 'mean',
            'Process temperature [K]': 'median',
            'Tool wear [min]': 'median'
        }
        df_transform.impute_missing_values(imputation_strategy)

        # Re-run null count after imputation and compare
        post_impute_null_count = data_info.count_null_values()
        print("\nNull Value Counts after imputation:\n",
              post_impute_null_count)

        # Visualise null value imputation comparison
        plotter.plot_null_comparison(
            initial_null_count, post_impute_null_count
                )

    else:
        print("No data was fetched from the database.")

    # Close the database connection
    db_connector.close_connection()
