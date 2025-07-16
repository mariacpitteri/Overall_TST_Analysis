import pandas as pd
import numpy as np
import glob
import os


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
