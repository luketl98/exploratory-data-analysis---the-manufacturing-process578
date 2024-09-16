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
        """
        Convert the specified column to a categorical data type.

        Parameters:
            column_name (str): The name of the column to convert.
        """
        self.dataframe[column_name] = self.dataframe[
            column_name].astype('category')

    def convert_to_boolean(self, column_name: str):
        """
        Convert the specified column to a boolean data type.

        Parameters:
            column_name (str): The name of the column to convert.
        """
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

    def plot_histogram(self, column):
        """ Histogram: to understand distribution of numerical columns """
        plt.figure(figsize=(8, 6))
        self.dataframe[column].hist(bins=20)
        plt.title(f'Histogram of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()

    def bar_plot_categorical(self, column):
        """ Bar Plots for Categorical Data """
        plt.figure(figsize=(8, 6))
        self.dataframe[column].value_counts().plot(kind='bar')
        plt.title(f'Bar Plot of {column}')
        plt.xlabel(column)
        plt.ylabel('Count')
        plt.show()

    def correlation_heatmap(self):
        """ Heatmap of Correlations: For numerical columns """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.dataframe.corr(), annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Heatmap')
        plt.show()

    def missing_data_matrix(self):
        """ Missing Data Matrix (via missingno) """
        plt.figure(figsize=(10, 6))
        msno.matrix(self.dataframe)
        plt.title('Missing Data Matrix')
        plt.show()

    def plot_boxplot(self, column, by=None):
        """ Boxplots: For visualising outliers & variance across categories """
        plt.figure(figsize=(8, 6))
        if by:
            sns.boxplot(x=by, y=column, data=self.dataframe)
        else:
            sns.boxplot(x=self.dataframe[column])
        plt.title(f'Boxplot of {column}')
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


if __name__ == "__main__":
    """
    Initialises the RDSDatabaseConnector with
    the provided database credentials.

    Parameters:
        credentials (dict): A dictionary containing the database credentials.
    """
    credentials = load_db_credentials('credentials.yaml')
    db_connector = RDSDatabaseConnector(credentials)

    # Define the SQL query for the required data
    query = "SELECT * FROM failure_data;"  # Adjust this query as needed

    # Fetch data from the database
    data = db_connector.fetch_data(query)

    # Save the fetched data to a CSV file, if data is not empty
    if not data.empty:
        csv_filename = 'failure_data.csv'
        db_connector.save_data_to_csv(data, csv_filename)
        print("\nData successfully saved to 'failure_data.csv'.")

        # Load the data from the CSV file and print its characteristics
        print("\nLoading data from CSV to verify contents:")
        loaded_data = load_data_from_csv(csv_filename)

        # Pass the loaded data to the DataTransform class
        transformer = DataTransform(loaded_data)

        # --- Apply the transformations: --- #

        # Convert 'Type' to categorical
        transformer.convert_to_categorical('Type')

        # Convert failure indicators to boolean
        failure_columns = ['Machine failure', 'TWF',
                           'HDF', 'PWF', 'OSF', 'RNF']
        for col in failure_columns:
            transformer.convert_to_boolean(col)

        # --- Extract and display information about the DataFrame,
        # Using the DataFrameInfo class ---

        # Initialise the DataFrameInfo class with the loaded data
        data_info = DataFrameInfo(loaded_data)

        # Display the extracted information
        print("\nColumn Descriptions:\n", data_info.describe_columns())
        print("\nExtracted Statistics:\n", data_info.extract_statistics())
        print("\nDistinct Value Counts:\n", data_info.count_distinct_values())
        data_info.print_shape()
        print("\nNull Value Counts:\n", data_info.count_null_values())

        # --- Plotting Section using Plotter class --- #

        # Initialise the Plotter class with loaded_data
        plotter = Plotter(loaded_data)

        # Create visualisations
        print("\nGenerating visualisations...")

        # Scatter Plot to identify outliers
        column_pairs = [
            ('Air temperature [K]', 'Process temperature [K]'),
            ('Rotational speed [rpm]', 'Torque [Nm]'),
            ('Tool wear [min]', 'Rotational speed [rpm]'),
            ('Tool wear [min]', 'Process temperature [K]'),
            ('Tool wear [min]', 'Machine failure')
        ]

        plotter.scatter_multiple_plots(column_pairs)

        """
        # Histogram to understand distribution of numeric columns
        plotter.plot_histogram(column='Air temperature [K]')

        # Bar plot for categorical data
        plotter.bar_plot_categorical(column='Type')

        # Heatmap of correlations
        plotter.correlation_heatmap()

        # Missing data matrix
        plotter.missing_data_matrix()

        # Boxplot for outliers
        plotter.plot_boxplot(column='Torque [Nm]', by='Type')
        """
    else:
        print("No data was fetched from the database.")

    # Close the database connection
    db_connector.close_connection()
