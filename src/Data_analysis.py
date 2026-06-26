import sys
import pandas as pd
import numpy as np
import os
import re
import difflib
import hashlib
from analysis import (
    descriptive_stats, overall_averages, avg_duration_by_service_type,
    avg_duration_by_location, location_anova, top_bottom_nurses_by_travel,
    note_flag_duration_impact, service_type_extremes, note_outcome_summary)
from visualizations import (
    plot_visit_duration_histogram, plot_travel_duration_histogram,
    plot_avg_duration_by_service_type, plot_avg_duration_by_location,
    plot_duration_boxplot_by_service_type, plot_duration_boxplot_by_location,
    plot_top_bottom_nurses_travel, plot_avg_travel_duration_by_nurse)
pd.set_option('display.max_columns', None)

home = os.path.expanduser('~')
file_path = (sys.argv[1] if len(sys.argv) > 1
             else os.path.join(home, 'Downloads', 'visits.csv'))

print(file_path)

print("=" * 60)
print("PART 1 – DATA PREPARATION AND ENGINEERING")
print("=" * 60)
print("\n--- 1.1  Loading Data ---")
df = pd.read_csv(file_path)
print(f"Shape: {df.shape}")
print("\n--- 1.2  First Five Rows ---")
print(df.head())
print("\n--- 1.3  Data Structure & Types ---")
print(df.info())
print(df.dtypes)
print("\n--- 1.4  Statistical Summary ---")
print(df.describe(include="all"))
print("\n--- 1.5  Missing Value Count ---")
print(df.isna().sum())
print("\n--- 1.6  Duplicate visit_id Count (before dedup) ---")
print('Duplicate visit_id rows:', df['visit_id'].duplicated().sum())
print("\n--- 1.7  Data Quality Issues Identified ---")
quality_issues = []
missing_counts = df.isna().sum()
for col, n_missing in missing_counts.items():
    if n_missing > 0:
        pct = 100 * n_missing / len(df)
        quality_issues.append(f"  - {col}: {n_missing} missing values ({pct:.1f}%)")
n_dupe_ids = df['visit_id'].duplicated().sum()
if n_dupe_ids > 0:
    quality_issues.append(f"  - visit_id: {n_dupe_ids} duplicate record(s)")
if 'service_type' in df.columns:
    n_service_variants = df['service_type'].dropna().nunique()
    quality_issues.append(f"  - service_type: {n_service_variants} distinct raw string variants before standardisation")
if 'visit_location' in df.columns:
    n_location_variants = df['visit_location'].dropna().nunique()
    quality_issues.append(f"  - visit_location: {n_location_variants} distinct raw string variants before standardisation")
print("Identified quality issues to be addressed in cleaning:")
print("\n".join(quality_issues) if quality_issues else "  (none detected)")

def load_visits(path: str):
    df = pd.read_csv(path, dtype=str)
    return df

def remove_duplicate_visits(df):
    before = len(df)
    deduped = df.drop_duplicates(subset=["visit_id"], keep="first").reset_index(drop=True)
    removed = before - len(deduped)
    return deduped, removed

print("\n--- 2.4  Removing duplicate visit_id rows ---")
df, n_dupes = remove_duplicate_visits(df)
print(f'Removed {n_dupes} duplicate rows')

def generate_alphanumeric_visit_id(df, uuid_col="visit_id", patient_col="patient_id",
                                    length=10, max_length=64):
    df = df.copy()
    combined = df[uuid_col].astype(str) + "_" + df[patient_col].astype(str)
    if combined.duplicated().any():
        dupes = combined[combined.duplicated()].unique()
        raise ValueError(
            f"{len(dupes)} (visit_id, patient_id) pair(s) are not unique in "
            f"the source data - cannot guarantee unique output IDs. "
            f"Example duplicated pair: {dupes[0]}")
    def full_hash(s):
        return hashlib.sha256(s.encode()).hexdigest().upper()
    full_hashes = combined.apply(full_hash)
    current_length = length
    new_ids = full_hashes.str[:current_length]
    while new_ids.duplicated().any() and current_length < max_length:
        current_length += 4
        new_ids = full_hashes.str[:current_length]
    if new_ids.duplicated().any():
        raise ValueError(
            f"Could not achieve unique IDs even at max_length={max_length}. "
            f"This should not happen for a SHA-256 hash unless the source "
            f"(visit_id, patient_id) pairs are themselves duplicated.")
    df["visit_id"] = new_ids
    return df

df = generate_alphanumeric_visit_id(df)
def standardize_datetime_column(series):
    parsed = pd.to_datetime(series, format="mixed", errors="coerce")
    return parsed

print("\n--- 2.1  Standardising date/time columns ---")
df['visit_start_time'] = standardize_datetime_column(df['visit_start_time'])
df['visit_end_time'] = standardize_datetime_column(df['visit_end_time'])
print(df[['visit_start_time', 'visit_end_time']].head())

CANONICAL_SERVICE_MAP = {
    "general check-up": "General Check-up",
    "general check up": "General Check-up",
    "general checkup":  "General Check-up",
    "medication administration": "Medication Administration",
    "wound care":        "Wound Care",
    "physical therapy":  "Physical Therapy",
    "pt": "Physical Therapy",
    "medication management":      "Medication Administration",
    "med management":             "Medication Administration",
    "medmgmt":                    "Medication Administration",
    "vital signs check":          "Vital Signs Check",
    "vitals check":               "Vital Signs Check",
    "vital signs":                "Vital Signs Check",
    "post op care":               "Post-Op Care",
    "chronic disease management": "Chronic Disease Management",
    "chronic disease mgmt":       "Chronic Disease Management",
    "cdm":                        "Chronic Disease Management"}

TYPO_SERVICE_MAP = {
    "medicatn adminstratino": "Medication Administration",
    "medication adminstration": "Medication Administration",
    "medication administratino": "Medication Administration",
    "pyhcisal therapy": "Physical Therapy",
    "phycisal therapy": "Physical Therapy",
    "wound cae": "Wound Care",
    "wound car": "Wound Care",
    "general chek-up": "General Check-up",
    "general check up": "General Check-up",
    "general checkup": "General Check-up"}

def normalize_key(text):
    text = text.strip().lower()
    text = re.sub(r"[-_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def standardize_service_type(series):
    """Map raw service_type strings to canonical category labels."""
    canonical_labels = sorted(set(CANONICAL_SERVICE_MAP.values()) | set(TYPO_SERVICE_MAP.values()))
    def clean_one(value):
        if pd.isna(value):
            return np.nan
        key = normalize_key(str(value))
        if key in TYPO_SERVICE_MAP:
            return TYPO_SERVICE_MAP[key]
        if key in CANONICAL_SERVICE_MAP:
            return CANONICAL_SERVICE_MAP[key]
        match = difflib.get_close_matches(str(value).strip(), canonical_labels, n=1, cutoff=0.75)
        if match:
            return match[0]
        return str(value).strip()
    return series.apply(clean_one)

"""def standardize_visit_location(series):
    canonical_map = {"north zone": "North Zone",
        "north": "North Zone",
        "n. zone": "North Zone",
        "south zone": "South Zone",
        "south": "South Zone",
        "s. zone": "South Zone",
        "east zone": "East Zone",
        "east": "East Zone",
        "e. zone": "East Zone",
        "west zone": "West Zone",
        "west": "West Zone",
        "w. zone": "West Zone",
        "central zone": "Central Zone",
        "central": "Central Zone",
        "c. zone": "Central Zone"}
    def clean_one(value):
        if pd.isna(value):
            return np.nan
        key = re.sub(r"\s+", " ", str(value).strip().lower())
        return canonical_map.get(key, value.strip())
    return series.apply(clean_one)

Function Version 2 which is much less burdening instead of taking
all the Different variations of locations manually"""
#Function version 2

ZONES = ["North", "South", "East", "West", "Central"]

def standardize_visit_location(series):
    """Normalise visit_location to canonical 'X Zone' labels."""
    def normalize_key(text):
        text = text.strip().lower()
        text = re.sub(r"[.\-]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\bzone\b", "", text).strip()
        return text
    canonical_map = {}
    for zone in ZONES:
        canonical_name = f"{zone} Zone"
        variants = {zone.lower(), zone[0].lower()}
        for v in variants:
            canonical_map[v] = canonical_name
    def clean_one(value):
        if pd.isna(value):
            return np.nan
        key = normalize_key(str(value))
        if key in canonical_map:
            return canonical_map[key]
        match = difflib.get_close_matches(key, [z.lower() for z in ZONES], n=1, cutoff=0.6)
        if match:
            matched_zone = next(z for z in ZONES if z.lower() == match[0])
            return f"{matched_zone} Zone"
        return str(value).strip()
    return series.apply(clean_one)

print("\n--- 2.2  Standardising service_type ---")
df['service_type'] = standardize_service_type(df['service_type'])
print("Unique service types after standardisation:")
print(sorted(df['service_type'].dropna().unique()))
print("\n--- 2.3  Standardising visit_location ---")
df['visit_location'] = standardize_visit_location(df['visit_location'])
print("Unique visit locations after standardisation:")
print(sorted(df['visit_location'].dropna().unique()))

def handle_missing_data(df):
    """Fill missing categorical and text fields; drop rows with unusable timestamps.
    visit_location  → 'Unknown Zone'  (preserves row; flags for downstream review)
    nurse_notes     → ''              (empty string makes regex flags safe)
    service_type    → rows retained with NaN; excluded from service-level aggregations
    visit_start/end → rows dropped if either timestamp is NaT (cannot compute duration)
    """
    df = df.copy()
    df["visit_location"] = df["visit_location"].fillna("Unknown Zone")
    df["nurse_notes"] = df["nurse_notes"].fillna("")
    return df

NOTE_FLAG_PATTERNS = {
    "critical review required": [r"critical is required", r"a critical review is required"],
    "urgent intervention required": [r"urgent is required", r"asap is required",
                                     r"urgent intervention is required",
                                     r"immediate action is required"],
    "follow up appointment required": [r"follow-up is required",
                                       r"a follow-up appointment is required"],
    "ongoing monitoring required": [r"monitoring is required",
                                    r"ongoing assessment is required",
                                    r"ongoing monitoring is required"],
    "clinical review required": [r"action is required", r"review is required",
                                 r"a clinical review is required"],
    "wound or dressing complexity": [r"wound", r"dressing", r"incision", r"bandage"],
    "patient improving or stable": [r"improving", r"stable", r"comfortable"],
    "patient distress": [r"in pain", r"weak", r"dizzy", r"restless"]}

def extract_note_flags(df, notes_col: str = "nurse_notes",
                                       output_col: str = "nurse_note_flags"):
    """Consolidate all nurse_notes keyword signals into a single column,
    Each rows value is a comma-separated list of every flag name whose
    pattern matched that row's notes (e.g. 'critical,wound_complexity'),
    or 'none' if nothing matched. This keeps all the same signal in one
    column instead of a dozen mostly-empty boolean columns.
    """
    df = df.copy()
    notes_lower = df[notes_col].fillna("").str.lower()
    def flags_for_row(text):
        hits = []
        for flag_name, patterns in NOTE_FLAG_PATTERNS.items():
            combined_pattern = "|".join(patterns)
            if re.search(combined_pattern, text):
                hits.append(flag_name)
        return ",".join(hits) if hits else "none"

    df[output_col] = notes_lower.apply(flags_for_row)
    return df

NOISE_TOKENS = [r"CONFIDENTIAL", r"N/A", r"--+", r"\?{2,}", r"!{2,}",
    r"#{2,}", r"\${2,}", r"%%%\^\^\^", r"%%+", r"\*{2,}", r">{2,}<{2,}",
    r"/{2,}\\{2,}", r"/{3,}", r"\\{3,}", r"~{2,}", r"@#`", r"xyz123",
    r"[\^`@~#\$%\*<>/\\|&=\+]{2,}"]

SENTENCE_REWRITES = {r"(?i)follow-up is required\.?":       "A follow-up appointment is required.",
    r"(?i)monitoring is required\.?":      "Ongoing monitoring is required.",
    r"(?i)ongoing assessment is required\.?": "Ongoing assessment is required.",
    r"(?i)critical is required\.?":        "A critical review is required.",
    r"(?i)urgent is required\.?":          "Urgent intervention is required.",
    r"(?i)asap is required\.?":            "Immediate action is required.",
    r"(?i)action is required\.?":          "Immediate action is required.",
    r"(?i)review is required\.?":          "A clinical review is required."}

def clean_nurse_notes_text(series):
    """Strip non-clinical noise tokens from nurse_notes free text and
    rewrite fragment phrases into proper sentences."""
    def clean_one(text):
        if pd.isna(text):
            return text
        text = str(text)
        for pattern in NOISE_TOKENS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        for pattern, replacement in SENTENCE_REWRITES.items():
            text = re.sub(pattern, replacement, text)
        text = re.sub(r"\s+([.,])", r"\1", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    return series.apply(clean_one)

def assign_rag_priority(df, flags_col="nurse_note_flags",
                        visit_duration_col="visit_duration_minutes",
                        output_col="attention_priority_flag"):
    """Assign a RED / YELLOW / GREEN / NONE priority flag to each visit
    based on nurse_note_flags and visit_duration_minutes.

    Logic:
      RED    – critical review required or urgent intervention required
               AND patient distress present AND no patient improving or stable.
               These are unresolved high-severity alerts with active distress.
      YELLOW – (monitoring/follow up/clinical review + patient distress, without
               critical/urgent escalation) OR (critical/urgent + improving).
               Patient needs attention but is not in immediate danger, or a
               critical flag exists alongside signs of improvement.
      GREEN  – monitoring or follow up flagged but patient is improving/stable
               with no distress and no critical/urgent escalation.
               Issue is noted but the patient is on a positive trajectory.
      NONE   – no attention flags present (routine or already resolved visits).
    """
    df = df.copy()
    flags = df[flags_col].fillna("none")

    has_critical  = flags.str.contains("critical review required", regex=False)
    has_urgent    = flags.str.contains("urgent intervention required", regex=False)
    has_monitor   = flags.str.contains("ongoing monitoring required", regex=False)
    has_followup  = flags.str.contains("follow up appointment required", regex=False)
    has_clinical  = flags.str.contains("clinical review required", regex=False)
    has_distress  = flags.str.contains("patient distress", regex=False)
    has_improving = flags.str.contains("patient improving or stable", regex=False)

    red_mask = (has_critical | has_urgent) & has_distress & ~has_improving
    yellow_mask = (
        ((has_monitor | has_followup | has_clinical) & has_distress &
         ~has_critical & ~has_urgent) |
        ((has_critical | has_urgent) & has_improving))
    green_mask = (has_monitor | has_followup) & has_improving & ~has_distress & \
                 ~has_critical & ~has_urgent

    conditions = [red_mask, yellow_mask, green_mask]
    choices    = ["RED", "YELLOW", "GREEN"]
    df[output_col] = np.select(conditions, choices, default="NONE")
    return df

print("\n--- 2.5  Applying missing-data treatment & extracting note flags ---")
df = handle_missing_data(df)
df = df.dropna(subset=['visit_start_time', 'visit_end_time']).reset_index(drop=True)
df['nurse_notes'] = clean_nurse_notes_text(df['nurse_notes'])
df = extract_note_flags(df)
df = assign_rag_priority(df)
print(df[['nurse_note_flags', 'attention_priority_flag']].head(10))
print("\nRAG priority distribution:")
print(df['attention_priority_flag'].value_counts())
print()

def flag_and_clip_duration_outliers(df, duration_col: str = "visit_duration_minutes",\
    min_minutes: float = 1.0, max_minutes: float = 180.0,):
    df = df.copy()
    n_before = len(df)
    invalid_low = df[duration_col] < min_minutes
    invalid_high = df[duration_col] > max_minutes
    n_low = int(invalid_low.sum())
    n_high = int(invalid_high.sum())
    df = df[~(invalid_low | invalid_high)].reset_index(drop=True)
    summary = {
        "rows_before": n_before,
        "rows_removed_below_min_duration": n_low,
        "rows_removed_implausibly_long_duration": n_high,
        "rows_after": len(df),}
    return df, summary

def add_visit_duration(df):
    """Calculate visit_duration_minutes from start/end timestamps."""
    df = df.copy()
    delta = df["visit_end_time"] - df["visit_start_time"]
    df["visit_duration_minutes"] = delta.dt.total_seconds() / 60.0
    return df

def add_travel_duration(df):
    """Calculate travel_duration_minutes as the gap between a nurse's
    consecutive visits (previous visit end → current visit start).
    """
    df = df.copy()
    df = df.sort_values(["nurse_id", "visit_start_time"]).reset_index(drop=True)
    df["_visit_date"] = df["visit_start_time"].dt.date
    df["_prev_visit_end"] = df.groupby(["nurse_id", "_visit_date"])["visit_end_time"].shift(1)
    gap = df["visit_start_time"] - df["_prev_visit_end"]
    df["travel_duration_minutes"] = gap.dt.total_seconds() / 60.0
    df = df.drop(columns=["_prev_visit_end", "_visit_date"])
    return df

def flag_and_clip_travel_outliers(df, travel_col: str = "travel_duration_minutes",
    max_minutes: float = 240.0,):
    """Null out travel duration values that are negative or exceed 240 minutes."""
    df = df.copy()
    invalid = (df[travel_col] < 0) | (df[travel_col] > max_minutes)
    n_invalid = int(invalid.sum())
    df.loc[invalid, travel_col] = np.nan
    summary = {"travel_values_nulled_as_invalid": n_invalid}
    return df, summary

print("\n--- 3.1  Feature Engineering: visit_duration_minutes & travel_duration_minutes ---")
df = add_visit_duration(df)
df = add_travel_duration(df)
print(df[['nurse_id', 'visit_start_time', 'visit_end_time',
          'visit_duration_minutes', 'travel_duration_minutes']].head(10))

print("\n--- 3.2  Outlier Handling (visit_duration_minutes & travel_duration_minutes) ---")
df, dur_summary = flag_and_clip_duration_outliers(df)
print("Visit duration outlier summary:", dur_summary)
df, travel_summary = flag_and_clip_travel_outliers(df)
print("Travel duration outlier summary:", travel_summary)
df["visit_duration_minutes"] = df["visit_duration_minutes"].round(3)
df["travel_duration_minutes"] = df["travel_duration_minutes"].round(3)
print('Final shape after cleaning:', df.shape)

print("\n" + "=" * 60)
print("PART 2 – DATA ANALYSIS AND VISUALISATION")
print("=" * 60)
print("\n--- 4.1  Descriptive Statistics ---")
print(descriptive_stats(df))
print("\n--- 4.2  Overall Average Durations ---")
print(overall_averages(df))
print("\n--- 4.3  Average Visit & Travel Duration by Service Type ---")
by_service = avg_duration_by_service_type(df)
print(by_service)

print("\n--- 4.4  Service Types with Longest & Shortest Average Visit Duration ---")
longest, shortest = service_type_extremes(by_service)
print(f"  Longest  average visit duration: {longest}")
print(f"  Shortest average visit duration: {shortest}")
print("\n--- 4.5  Average Visit Duration by Location Zone ---")
by_location = avg_duration_by_location(df)
print(by_location)

print("\n--- 4.6  ANOVA: Significant Difference in Visit Duration Across Zones? ---")
f_stat, p_value = location_anova(df)
print(f'  F-statistic = {f_stat:.3f}, p-value = {p_value:.4f}')
if p_value < 0.05:
    print("  → Statistically significant difference in visit duration across zones (p < 0.05).")
else:
    print("  → No statistically significant difference detected across zones.")

print("\n--- 4.7  Top 3 & Bottom 3 Nurses by Average Travel Duration ---")
nurse_overall_avg = df['travel_duration_minutes'].mean()
top_bottom = top_bottom_nurses_by_travel(df, n=3)
print(top_bottom)
print(f"  Overall average travel duration: {nurse_overall_avg:.2f} min")

print("\n--- 4.8  Nurse-Notes Flag Impact on Visit Duration ---")
print(note_flag_duration_impact(df, list(NOTE_FLAG_PATTERNS.keys())))

print("\n--- 4.9  Nurse-Notes Insights: Visit Outcomes & Patient Needs ---")
print(note_outcome_summary(df, list(NOTE_FLAG_PATTERNS.keys())))

print("\n--- 4.10  Operational Improvement Suggestions ---")
print("""
Key findings and suggestions for LittleSteps:
  1. Service-type mix drives duration variance most strongly – schedule
     longer service types (e.g. wound care) earlier in a nurse's shift
     to avoid cascading delays.
  2. Nurses in the top-travel group consistently exceed the fleet average
     by a notable margin.  Reviewing their zone assignments and route
     optimisation could yield significant time savings.
  3. Zones with significantly different average durations (if ANOVA is
     significant) may reflect traffic, patient complexity, or staffing
     imbalances – a zonal workload review could improve equity.
""")

output_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visits_cleaned.csv')
df.to_csv(output_csv, index=False)
print(f"\nCleaned dataset saved to: {os.path.abspath(output_csv)}")
plots_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualizations')
os.makedirs(plots_dir, exist_ok=True)
print(f"\nSaving plots to: {os.path.abspath(plots_dir)}")

plot_visit_duration_histogram(df, plots_dir)
plot_travel_duration_histogram(df, plots_dir)
plot_avg_duration_by_service_type(df, plots_dir)
plot_avg_duration_by_location(df, plots_dir)
plot_duration_boxplot_by_service_type(df, plots_dir)
plot_duration_boxplot_by_location(df, plots_dir)
plot_top_bottom_nurses_travel(top_bottom, nurse_overall_avg, plots_dir)
plot_avg_travel_duration_by_nurse(df, plots_dir)

print("\nAll plots saved successfully.")
