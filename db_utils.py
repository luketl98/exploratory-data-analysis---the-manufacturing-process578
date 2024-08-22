import pandas as pd
import yaml
from sqlalchemy import create_engine


def load_data_from_csv(file_path: str) -> pd.DataFrame:
    # Load the data
    data = pd.read_csv(file_path)

    # Print the shape of the DataFrame
    print("Data shape:", data.shape)

    # Print 1st few rows of DataFrame, data types and info
    # to understand what the data looks like
    print("\nPrint initial data types, then data info:")
    print("\n", data.dtypes)
    print("\n", data.info())

    # Return the DataFrame
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


# Class to transform the data as needed
class DataTransform:
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


class DataFrameInfo:
    """
    A class to extract and display information from a Pandas DataFrame
    for exploratory data analysis (EDA).
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Initialize with a DataFrame.

        Parameters:
            dataframe (pd.DataFrame): The DataFrame to analyze.
        """
        self.dataframe = dataframe

    def describe_columns(self) -> pd.DataFrame:
        """
        Describe all columns in the DataFrame to check their data types.

        Returns:
            pd.DataFrame: Summary statistics of all DataFrame columns.
        """
        return self.dataframe.describe(include='all')

    def extract_statistics(self, column_name: str) -> dict:
        """
        Extract statistical values: median, standard deviation,
        and mean from a specific column.

        Parameters:
            column_name (str): The name of the column to analyze.

        Returns:
            dict: A dictionary containing the median, standard deviation,
            and mean of the column.
        """
        statistics = {
            'mean': self.dataframe[column_name].mean(),
            'median': self.dataframe[column_name].median(),
            'std_dev': self.dataframe[column_name].std()
        }
        return statistics

    def count_distinct_values(self, column_name: str) -> int:
        """
        Count distinct values in a categorical column.

        Parameters:
            column_name (str): The name of the categorical column.

        Returns:
            int: The count of distinct values.
        """
        return self.dataframe[column_name].nunique()

    def print_shape(self) -> None:
        """Print out the shape of the DataFrame."""
        print(f"DataFrame shape: {self.dataframe.shape}")

    def count_null_values(self) -> pd.DataFrame:
        """
        Generate a count and percentage of NULL values in each column.

        Returns:
            pd.DataFrame: A DataFrame showing the count and percentage of
            NULL values for each column.
        """
        null_count = self.dataframe.isnull().sum()
        null_percentage = (null_count / len(self.dataframe)) * 100
        null_info = pd.DataFrame({
            'null_count': null_count,
            'null_percentage': null_percentage
        })
        return null_info


if __name__ == "__main__":
    """
    Initializes the RDSDatabaseConnector with
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

        # Apply the transformations:
        # Convert 'Type' to categorical
        transformer.convert_to_categorical('Type')

        # Convert failure indicators to boolean
        failure_columns = ['Machine failure', 'TWF',
                           'HDF', 'PWF', 'OSF', 'RNF']
        for col in failure_columns:
            transformer.convert_to_boolean(col)

        # Check the result
        print("\nData types after transformations:\n",
              transformer.dataframe.dtypes)

    else:
        print("No data was fetched from the database.")

    # Close the database connection
    db_connector.close_connection()
