import pandas as pd
import yaml


# --- Utility Methods -----


# Function to load database credentials from a YAML file
def load_db_credentials(file_path: str = "credentials.yaml") -> dict:
    with open(file_path, "r") as file:
        credentials = yaml.safe_load(file)

    return credentials


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
