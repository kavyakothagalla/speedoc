from scipy import stats as scipy_stats
import pandas as pd

def descriptive_stats(df):
    """Return descriptive statistics for both duration columns."""
    return df[["visit_duration_minutes", "travel_duration_minutes"]].describe()

def overall_averages(df):
    """Return the mean visit and travel duration across all records."""
    return df[["visit_duration_minutes", "travel_duration_minutes"]].mean()

def avg_duration_by_service_type(df):
    """Compute mean, median, std and count of visit duration grouped by service type. Sorted by mean descending so the longest-average service type appears first."""
    result = (df.groupby("service_type")["visit_duration_minutes"]
              .agg(["mean", "median", "std", "count"])
              .sort_values("mean", ascending=False))
    return result

def service_type_extremes(by_service_df):
    """Return the service types with the longest and shortest average visit durations.
    Parameters"""
    longest = by_service_df.index[0]
    shortest = by_service_df.index[-1]
    return longest, shortest

def avg_duration_by_location(df):
    """Compute mean, median, std and count of visit duration grouped by location zone. Sorted by mean descending."""
    result = (df.groupby("visit_location")["visit_duration_minutes"].agg(["mean", "median", "std", "count"])\
              .sort_values("mean", ascending=False))
    return result

def location_anova(df):
    """One-way ANOVA: test whether visit duration differs significantly across zones. Groups with fewer than 2 observations are excluded."""
    groups = [group["visit_duration_minutes"].dropna().values
              for _, group in df.groupby("visit_location")
              if len(group) > 1]
    f_stat, p_value = scipy_stats.f_oneway(*groups)
    return f_stat, p_value


def top_bottom_nurses_by_travel(df, n = 3):
    """Identify the top-n and bottom-n nurses by average travel duration. Only nurses with at least 5 travel-duration observations are included
    to avoid artefacts from nurses with very few recorded trips."""
    overall_avg = df["travel_duration_minutes"].mean()
    nurse_travel = (df.dropna(subset=["nurse_id", "travel_duration_minutes"]).groupby("nurse_id")["travel_duration_minutes"]\
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "avg_travel_duration_minutes", "count": "n_observations"}))
    nurse_travel = nurse_travel[nurse_travel["n_observations"] >= 2]
    nurse_travel["diff_from_overall_avg"] = (nurse_travel["avg_travel_duration_minutes"] - \
                                             overall_avg)
    nurse_travel = nurse_travel.sort_values("avg_travel_duration_minutes", ascending=False)
    top_n = nurse_travel.head(n).copy()
    top_n["rank_group"] = "Top (longest travel)"
    bottom_n = nurse_travel.tail(n).copy()
    bottom_n["rank_group"] = "Bottom (shortest travel)"
    return pd.concat([top_n, bottom_n])

def note_outcome_summary(df, flag_names, flags_col="nurse_note_flags"):
    """Summarise visit outcomes and patient needs inferred from nurse_notes."""
    rows = []
    n_total = len(df)
    flags_series = df[flags_col].fillna("none")
    for flag in flag_names:
        has_flag = flags_series.str.contains(rf"\b{flag}\b", regex=True)
        n_with_flag = int(has_flag.sum())
        rows.append({
            "outcome_flag": flag,
            "n_visits": n_with_flag,
            "pct_of_all_visits": round(100 * n_with_flag / n_total, 2) if n_total else 0.0})
    return pd.DataFrame(rows).sort_values("n_visits", ascending=False)

def note_flag_duration_impact(df, flag_names, flags_col="nurse_note_flags"):
    rows = []
    overall_mean = df["visit_duration_minutes"].mean()
    flags_series = df[flags_col].fillna("none")
    for flag in flag_names:
        has_flag = flags_series.str.contains(rf"\b{flag}\b", regex=True)
        with_flag = df.loc[has_flag, "visit_duration_minutes"]
        without_flag = df.loc[~has_flag, "visit_duration_minutes"]
        rows.append({"note_flag": flag,
                     "n_visits_with_flag": int(has_flag.sum()),
                     "avg_duration_with_flag": with_flag.mean(),
                     "avg_duration_without_flag": without_flag.mean(),
                     "diff_vs_overall_avg": with_flag.mean() - overall_mean})
    return pd.DataFrame(rows).sort_values("diff_vs_overall_avg", ascending=False)
