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


def get_failed_participants(questionnaires_dfs, tst_dfs):
    failed_participants = {
        # Questionnaire checks
        "catch_response": set(),
        "catch_infrequent_response": set(),
        # TST checks
        "catch_side_response": set(),
        "catch_missed_trials": set(),
        "catch_RT": set(),
    }

    # Check questionnaire failures
    for key, df in questionnaires_dfs.items():
        if key.startswith("mh_calculated_scores"):
            if "catch_response" in df.columns:
                failed_participants["catch_response"].update(
                    pid
                    for pid, group in df.groupby("participant_id")
                    if (group["catch_response"] == "Fail").any()
                )
            if "catch_infrequent_response" in df.columns:
                failed_participants["catch_infrequent_response"].update(
                    pid
                    for pid, group in df.groupby("participant_id")
                    if (group["catch_infrequent_response"] == "Fail").any()
                )

    # Check TST failures
    for key, df in tst_dfs.items():
        if "catch_side_response" in df.columns:
            failed_participants["catch_side_response"].update(
                pid
                for pid, group in df.groupby("participant_id")
                if (group["catch_side_response"] == "Fail").any()
            )

        if "catch_missed_trials" in df.columns:
            failed_participants["catch_missed_trials"].update(
                pid
                for pid, group in df.groupby("participant_id")
                if (group["catch_missed_trials"] == "Fail").any()
            )

        if "catch_RT" in df.columns:
            failed_participants["catch_RT"].update(
                pid
                for pid, group in df.groupby("participant_id")
                if (group["catch_RT"] == "Fail").any()
            )

    # Optional: Add total set of failed participants across all checks
    failed_participants["any_failure"] = set().union(*failed_participants.values())

    # Print summary
    for test_name, failed_ids in failed_participants.items():
        if failed_ids:
            print(f"Participants who failed {test_name}: {sorted(failed_ids)}")
        else:
            print(f"No participants failed {test_name}")

    return failed_participants


def merge_dataframes(demographic_dfs, questionnaires_dfs, tst_dfs):
    # Extract relevant demographic dataframe
    demographic_df = next(
        df
        for key, df in demographic_dfs.items()
        if key.startswith("demographic_results")
    )[
        [
            "participant_id",
            "study",
            "age",
            "ethnicity",
            "mh_past",
            "mh_current",
            "fam_mh_past",
        ]
    ]

    # Extract relevant questionnaire scores dataframe
    questionnaires_df = next(
        df
        for key, df in questionnaires_dfs.items()
        if key.startswith("mh_overall_scores")
    )[
        [
            "participant_id",
            "ocir_overall",
            "alcohol_overall",
            "social_overall",
            "bis_overall",
            "neutral_overall",
            "depress_overall",
            "eat_overall",
            "schizo_overall",
            "iq_overall",
            "anxiety_overall",
            "apathy_overall",
        ]
    ]

    # Extract TST MB dataframe
    tst_df = next(df for key, df in tst_dfs.items() if key.startswith("MB"))[
        ["participant_id", "lastWinRew_lastTranUnc"]
    ]

    # Merge them all on participant_id
    merged_df = demographic_df.merge(
        questionnaires_df, on="participant_id", how="inner"
    )
    merged_df = merged_df.merge(tst_df, on="participant_id", how="inner")

    # Reorder columns as specified
    ordered_columns = [
        "study",
        "participant_id",
        "age",
        "lastWinRew_lastTranUnc",
        "ethnicity",
        "ocir_overall",
        "alcohol_overall",
        "social_overall",
        "bis_overall",
        "neutral_overall",
        "depress_overall",
        "eat_overall",
        "schizo_overall",
        "iq_overall",  # Note: column is iq_overall in source, not iq_iq_overall
        "anxiety_overall",
        "apathy_overall",
        "mh_past",
        "mh_current",
        "fam_mh_past",
    ]

    merged_df = merged_df[ordered_columns]

    return merged_df
