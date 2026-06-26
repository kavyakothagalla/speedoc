import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_theme(style="whitegrid")

def _save(fig, plots_dir, filename):
    """Create plots_dir if needed, save figure, then close it."""
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_visit_duration_histogram(df, plots_dir):
    """Histogram with KDE overlay showing the distribution of visit durations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["visit_duration_minutes"].dropna(), bins=30, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Distribution of Visit Duration (minutes)")
    ax.set_xlabel("Visit Duration (minutes)")
    ax.set_ylabel("Number of Visits")
    return _save(fig, plots_dir, "visit_duration_histogram.png")

def plot_travel_duration_histogram(df, plots_dir):
    """Histogram with KDE overlay showing the distribution of travel durations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["travel_duration_minutes"].dropna(), bins=30, kde=True, ax=ax, color="#DD8452")
    ax.set_title("Distribution of Travel Duration (minutes)")
    ax.set_xlabel("Travel Duration (minutes)")
    ax.set_ylabel("Number of Visits")
    return _save(fig, plots_dir, "travel_duration_histogram.png")

def plot_avg_duration_by_service_type(df, plots_dir):
    """Horizontal bar chart of average visit duration per service type."""
    avg_by_type = (df.groupby("service_type")["visit_duration_minutes"].mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=avg_by_type.values, y=avg_by_type.index, ax=ax, color="#55A868")
    ax.set_title("Average Visit Duration by Service Type")
    ax.set_xlabel("Average Visit Duration (minutes)")
    ax.set_ylabel("Service Type")
    return _save(fig, plots_dir, "avg_duration_by_service_type.png")

def plot_avg_duration_by_location(df, plots_dir):
    """Horizontal bar chart of average visit duration per location zone."""
    avg_by_zone = (df.groupby("visit_location")["visit_duration_minutes"].mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=avg_by_zone.values, y=avg_by_zone.index, ax=ax, color="#C44E52")
    ax.set_title("Average Visit Duration by Location Zone")
    ax.set_xlabel("Average Visit Duration (minutes)")
    ax.set_ylabel("Visit Location")
    return _save(fig, plots_dir, "avg_duration_by_location.png")

def plot_duration_boxplot_by_service_type(df, plots_dir):
    """Box plot showing the spread of visit durations for each service type."""
    order = (df.groupby("service_type")["visit_duration_minutes"].median().sort_values(ascending=False).index)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="visit_duration_minutes", y="service_type", order=order, ax=ax, color="#8172B2")
    ax.set_title("Visit Duration Distribution by Service Type")
    ax.set_xlabel("Visit Duration (minutes)")
    ax.set_ylabel("Service Type")
    return _save(fig, plots_dir, "duration_boxplot_by_service_type.png")


def plot_duration_boxplot_by_location(df, plots_dir):
    """Box plot showing the spread of visit durations for each location zone."""
    order = (df.groupby("visit_location")["visit_duration_minutes"].median().sort_values(ascending=False).index )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=df, x="visit_duration_minutes", y="visit_location", order=order, ax=ax, color="#937860")
    ax.set_title("Visit Duration Distribution by Location Zone")
    ax.set_xlabel("Visit Duration (minutes)")
    ax.set_ylabel("Visit Location")
    return _save(fig, plots_dir, "duration_boxplot_by_location.png")


def plot_avg_travel_duration_by_nurse(df, plots_dir):
    """Bar chart of average travel duration for every nurse meeting the
    minimum-observation threshold, sorted descending.
    This complements plot_top_bottom_nurses_travel by showing the full
    nurse population rather than just the extremes, so the question of
    "who consistently has longer/shorter travel" is visualised across
    all nurses, not only the top 3 and bottom 3.
    """
    nurse_avg = (df.dropna(subset=["nurse_id", "travel_duration_minutes"])
                 .groupby("nurse_id")["travel_duration_minutes"]
                 .agg(["mean", "count"]))
    nurse_avg = nurse_avg[nurse_avg["count"] >= 2].sort_values("mean", ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.3 * len(nurse_avg))))
    sns.barplot(x=nurse_avg["mean"].values, y=nurse_avg.index, ax=ax, color="#64B5CD")
    ax.set_title("Average Travel-Duration Gap by Nurse (all nurses, min. 2 observations)")
    ax.set_xlabel("Average Travel-Duration Gap (minutes)")
    ax.set_ylabel("Nurse ID")
    return _save(fig, plots_dir, "avg_travel_duration_by_nurse.png")


def plot_top_bottom_nurses_travel(top_bottom_df, overall_avg, plots_dir):
    """Horizontal bar chart comparing top-3 and bottom-3 nurses by travel duration.
    A dashed vertical line marks the overall fleet average for reference.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = top_bottom_df["rank_group"].map({"Top (longest travel)": "#C44E52", "Bottom (shortest travel)": "#55A868"})
    ax.barh(top_bottom_df.index, top_bottom_df["avg_travel_duration_minutes"], color=colors)
    ax.axvline(overall_avg, color="black", linestyle="--", linewidth=1, label=f"Overall avg = {overall_avg:.1f} min")
    ax.set_title("Top 3 & Bottom 3 Nurses by Average Travel Duration Gap")
    ax.set_xlabel("Average Travel-Duration Gap (minutes)")
    ax.set_ylabel("Nurse ID")
    ax.legend()
    return _save(fig, plots_dir, "top_bottom_nurses_travel.png")
