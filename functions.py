import pandas as pd
import numpy as np
import glob
import os


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
