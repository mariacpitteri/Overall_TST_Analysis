import pandas as pd
import numpy as np
import glob
import os
from scipy.stats import norm, kruskal, mannwhitneyu, pearsonr, ttest_ind


# Function to load all data from the different parts of the study
def load_data(directory_path):
    """
    Loads all CSV files from the given directory.
    Returns a dictionary of DataFrames with keys as filenames (without extensions).
    """
    file_paths = glob.glob(os.path.join(directory_path, "*.csv"))
    file_paths.sort()

    dataframes = {
        os.path.splitext(os.path.basename(path))[0]: pd.read_csv(path)
        for path in file_paths
    }

    return dataframes


def group_mb_data(df, group_col, score_col):
    result = (
        df.groupby(group_col)[score_col]
        .agg(
            count="count",  # number of trials
            mean="mean",  # mean of score_col
            std="std",  # standard deviation
            se=lambda x: x.std() / np.sqrt(x.count()),  # standard error of the mean
            prob_Stay=lambda x: (x == 1).mean(),  # proportion of score_col == 1
        )
        .reset_index()
    )

    return result


def cohen_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt(
        (np.std(x, ddof=1) ** 2 + np.std(y, ddof=1) ** 2) / 2
    )


def pearson_correlation(participant_df, x_variable, y_variable):
    # Drop rows where either x or y is NaN
    valid_data = participant_df[[x_variable, y_variable]].dropna()

    n = len(valid_data)

    if n < 2:
        print(
            f"Not enough valid data to compute correlation between {x_variable} and {y_variable}."
        )
        return

    r, p = pearsonr(valid_data[x_variable], valid_data[y_variable])
    print(
        f"Correlation ({x_variable} vs {y_variable}): r = {r:.2f}, p = {p:.3g}, n = {n}"
    )


def indep_t_test(participant_df, group_score_col, score_col):
    group_col = f"group_{group_score_col}"

    # Filter rows with non-NaN scores AND valid group labels
    valid_data = participant_df[[score_col, group_col]].dropna()

    n = len(valid_data)
    if n < 2:
        print(f"Not enough valid data to compute {score_col}.")
        return

    # Split groups
    Top_25 = valid_data[valid_data[group_col] == "Top_25"][score_col]
    Bottom_25 = valid_data[valid_data[group_col] == "Bottom_25"][score_col]

    # Convert to NumPy arrays
    x1 = Top_25.values
    x2 = Bottom_25.values
    n1 = len(x1)
    n2 = len(x2)

    # T-test (assuming equal variances)
    t_stat, p_val = ttest_ind(x1, x2, equal_var=True)

    # Degrees of freedom for equal variances
    df = n1 + n2 - 2

    print(
        f"T-statistic: {t_stat:.3f}, df: {df}, p-value: {p_val:.3g}, "
        f"Cohen's d: {cohen_d(x1, x2):.3f}, "
        f"n_Bottom_25: {n2}, n_Top_25: {n1}"
    )


def group_subjects(individuals_df, score_col):
    # Drop rows with NaN in score_col
    filtered_df = individuals_df.dropna(subset=[score_col]).copy()

    # Calculate the 25th and 75th percentiles of OCI-R scores
    p25 = np.percentile(filtered_df[score_col], 25)
    p75 = np.percentile(filtered_df[score_col], 75)

    # Assign groups based on percentile cutoffs
    conditions = [
        filtered_df[score_col] >= p75,
        filtered_df[score_col] <= p25,
    ]
    choices = ["Top_25", "Bottom_25"]

    # Label top 25% as 'OCD', bottom 25% as 'Control', and others as 'Other'
    filtered_df[f"group_{score_col}"] = np.select(conditions, choices, default="Other")

    # Assign the new column back to the original dataframe using .loc
    individuals_df.loc[filtered_df.index, f"group_{score_col}"] = filtered_df[
        f"group_{score_col}"
    ]

    return individuals_df
