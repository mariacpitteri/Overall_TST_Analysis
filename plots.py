import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_centiles(df):
    # Create figure and set background color
    fig, ax = plt.subplots(figsize=(9, 6))
    # fig.patch.set_facecolor("#F0F8EB")  # Background of the figure
    # ax.set_facecolor("#F0F8EB")  # Background of the plot area

    # Shaded areas for central ranges
    ax.fill_between(
        df["Age"],
        df["25th"],
        df["75th"],
        alpha=0.3,
        label="IQR (25–75%)",
        color="indianred",
    )
    ax.fill_between(
        df["Age"],
        df["10th"],
        df["90th"],
        alpha=0.2,
        label="Central 80% (10–90%)",
        color="lightcoral",
    )
    ax.fill_between(
        df["Age"],
        df["5th"],
        df["95th"],
        alpha=0.15,
        label="Central 90% (5–95%)",
        color="PeachPuff",
    )
    ax.fill_between(
        df["Age"],
        df["1th"],
        df["99th"],
        alpha=0.1,
        label="Central 98% (1–99%)",
        color="lightgrey",
    )

    # Median and centile lines
    ax.plot(df["Age"], df["50th"], label="Median MB", color="black", linewidth=2)
    ax.plot(df["Age"], df["5th"], linestyle="--", color="grey", linewidth=1)
    ax.plot(df["Age"], df["95th"], linestyle="--", color="grey", linewidth=1)
    ax.plot(df["Age"], df["1th"], linestyle=":", color="darkgrey", linewidth=1)
    ax.plot(df["Age"], df["99th"], linestyle=":", color="darkgrey", linewidth=1)

    # Plot formatting
    ax.set_ylim(-0.5, 1.8)
    # ax.legend(fontsize=14)
    ax.set_xlabel("Age", fontsize=14)
    ax.set_ylabel("Model-Based Score", fontsize=14)
    ax.set_title("Normative Centile Curves of MB Control Across Age", fontsize=16)
    # ax.grid(True, alpha=0.2)
    ax.tick_params(axis="both", labelsize=20)

    plt.tight_layout()
    plt.show()


def scatter_plots(participants_df, x_col, y_col, title, dot_color, line_color):
    plt.figure(figsize=(8, 6))

    # Simple scatter and regression line
    sns.scatterplot(data=participants_df, x=x_col, y=y_col, color=dot_color)
    sns.regplot(
        data=participants_df,
        x=x_col,
        y=y_col,
        scatter=False,
        color=line_color,
        line_kws={"linewidth": 2},
    )

    plt.title(title, fontsize=14)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    # Set custom y-axis ticks
    # plt.yticks(np.arange(-0.5, 1.6, 0.5), fontsize=25)  # stops at 1.5
    # plt.ylim(-0.5, 1.75)
    plt.tight_layout()
    plt.show()


def point_plot(participants_df):
    # Filter out (Intercept)
    df_filtered = participants_df[
        participants_df["Predictor"].str.strip().str.lower() != "(intercept)"
    ]

    plt.figure(figsize=(8, 6))

    # Plot points using seaborn (for styling)
    sns.scatterplot(data=df_filtered, x="Predictor", y="Estimate", color="blue", s=100)

    # Add error bars manually using Std. Error
    plt.errorbar(
        x=df_filtered["Predictor"],
        y=df_filtered["Estimate"],
        yerr=df_filtered["Std. Error"],
        fmt="none",  # No extra markers
        ecolor="black",  # Error bar color
        elinewidth=1.2,
        capsize=4,
    )

    plt.ylim(min(0, df_filtered["Estimate"].min()), df_filtered["Estimate"].max() + 0.3)
    plt.xticks(rotation=45)
    plt.title("Predictor Estimates ± SE")
    plt.tight_layout()
    plt.show()


def box_plots(participants_df, group_col, score_col, hue_col, title, palette):
    plt.figure(figsize=(8, 5))

    # Create boxplot
    ax = sns.boxplot(
        x=group_col,
        y=score_col,
        hue=hue_col,
        data=participants_df,
        palette=palette,
        dodge=True,
    )

    # Set alpha for each box patch
    # Each hue level per group_col adds one box, so we iterate over ax.patches
    for patch in ax.patches:
        patch.set_facecolor(patch.get_facecolor()[:3] + (0.7,))  # Set alpha to 0.3

    # Add stripplot on top
    sns.stripplot(
        x=group_col,
        y=score_col,
        hue=hue_col,
        data=participants_df,
        dodge=True,
        palette=palette,
        alpha=0.7,
        size=6,
        jitter=True,
        legend=False,
    )

    # Clean up duplicated legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles[: len(participants_df[hue_col].unique())],
        labels[: len(participants_df[hue_col].unique())],
        title=hue_col,
    )

    plt.title(title, fontsize=14)
    plt.xlabel(group_col)
    plt.ylabel(score_col)
    plt.tight_layout()
    plt.show()


def bar_plots(
    participants_df,
    group_col,
    score_col,
    title,
    hue=None,
    palette="crest",
    legend=False,
):
    sns.barplot(
        x=group_col,
        y=score_col,
        hue=hue,
        data=participants_df,
        estimator="mean",
        errorbar="sd",
        palette=palette,
        legend=legend,
    )
    plt.title(title)
    plt.show()


def strip_chart_with_mean(
    participants_df, group_col, score_col, title, bottom_color, top_color
):
    # Filter to include only Bottom_25 and Top_25 groups and drop NaNs
    filtered_df = participants_df[
        participants_df[group_col].isin(["Bottom_25", "Top_25"])
    ].dropna(subset=[score_col])

    # Define palette directly
    custom_palette = {
        "Bottom_25": bottom_color,
        "Top_25": top_color,
    }

    plt.figure(figsize=(8, 6))
    sns.stripplot(
        x=group_col,
        y=score_col,
        data=filtered_df,
        order=["Bottom_25", "Top_25"],
        jitter=True,
        alpha=0.7,
        palette=custom_palette,
        hue=group_col,
        legend=False,
    )

    # Plot means
    group_means = filtered_df.groupby(group_col)[score_col].mean()
    group_std = filtered_df.groupby(group_col)[score_col].std()
    for i, (group, mean_val) in enumerate(group_means.items()):
        plt.scatter(group, mean_val, color="black", marker="D", s=100, zorder=10)
        print(f"The mean value for the {group} is {mean_val}")

    for i, (group, std_val) in enumerate(group_std.items()):
        print(f"The std value for the {group} is {std_val}")

    plt.title(title)
    plt.xlabel(group_col)
    plt.ylabel(score_col)
    plt.tight_layout()
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    # Show legend if necessary
    handles, labels = plt.gca().get_legend_handles_labels()
    if "Mean" in labels:
        plt.legend(handles=[handles[labels.index("Mean")]])

    plt.show()
