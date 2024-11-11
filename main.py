from db_utils import load_db_credentials, save_data_to_csv
from db_connector import RDSDatabaseConnector
from executor import EDAExecutor
from transformations import DataFrameTransform


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
