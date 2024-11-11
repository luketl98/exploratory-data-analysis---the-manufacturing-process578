import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import plotly.express as px
from statsmodels.graphics.gofplots import qqplot

from db_utils import filter_columns


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
    ) -> None:
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

    def plot_pie_chart(self, title: str, labels: list, sizes: list) -> None:
        """
        Plot a pie chart with the given title, labels, and sizes.

        Args:
            title (str): The title of the pie chart.
            labels (list): A list of labels for the pie chart.
            sizes (list): A list of sizes corresponding to each label.
        """
        plt.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",  # Display percentage on the pie chart
            startangle=90,  # Start the pie chart at 90 degrees
            colors=["red", "green"],  # Use red for failures and green for non-failures
        )
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def plot_product_quality_chart(
            self,
            failures_by_quality: pd.Series,
            failure_percent_by_quality: pd.Series
            ) -> None:
        """
        Plot a bar chart to visualise failures by product quality type.

        Args:
            failures_by_quality (pd.Series): Series containing failure counts
                                             by product quality type.
            failure_percent_by_quality (pd.Series): Series containing failure
                                                    percentages by product quality type.
        """
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
        type, prints the results, and calls the plotting method for visualisation.
        """
        failure_columns = ["TWF", "HDF", "PWF", "OSF", "RNF"]

        # Sum the failures for each failure type
        cause_of_failure_counts = self.dataframe[failure_columns].sum()
        total_failures = cause_of_failure_counts.sum()

        # Print the total number of failures for each cause
        print("\nLeading Causes of Failure:\n", cause_of_failure_counts.to_string())

        # Calculate the percentage of failures for each cause
        failure_percent_by_cause = (cause_of_failure_counts / total_failures) * 100

        # Call plot_failure_causes_chart method for visualisation
        self.plot_failure_causes_chart(
            cause_of_failure_counts, failure_percent_by_cause
            )

    def plot_failure_causes_chart(
            self,
            cause_of_failure_counts: pd.Series,
            failure_percent_by_cause: pd.Series
            ) -> None:
        """
        Plot a bar chart to visualise the leading causes of failure.

        Args:
            cause_of_failure_counts (pd.Series): Series containing failure counts by
                                                 failure type.
            failure_percent_by_cause (pd.Series): Series containing failure percentages
                                                  by failure type.
        """
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
