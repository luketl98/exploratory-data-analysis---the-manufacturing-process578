import pandas as pd
import numpy as np
import scipy.stats as stats

from db_utils import save_data_to_csv
from transformations import DataTransform, DataFrameTransform
from df_info import DataFrameInfo
from visualisations import Plotter
from calculator import MachineSettingCalculator


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
        df_info = DataFrameInfo(data)

        # Calculations and plots for failure analysis
        df_info.calculate_failure_rate()
        df_info.failures_by_product_quality()
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
