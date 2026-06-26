# LittleSteps – Patient Visit Data Analysis

## Introduction

This project was completed as part of a take-home data engineering and analysis assignment for LittleSteps, an at-home healthcare startup. The goal is to support operational efficiency by analysing patient visit durations and nurse travel times, and surfacing actionable insights for scheduling and resource planning.

The analysis covers three areas:

- **Data Engineering** — loading, cleaning, standardising, and feature engineering on raw visit data
- **Data Analysis** — descriptive statistics, service-type and location breakdowns, nurse travel profiling, and nurse-notes intelligence
- **Visualisation** — charts and plots to support all key findings

---

## Repository Structure

```
.
├── Data_analysis.py        # Main pipeline: runs all cleaning, analysis, and visualisation
├── analysis.py             # Analysis functions (stats, groupings, ANOVA, nurse travel)
├── visualizations.py       # Plotting functions (histograms, bar charts, box plots)
├── visits_cleaned.csv      # Output: cleaned and engineered dataset
├── README.md               # This file
└── visualizations/         # Output: all generated plots
    ├── visit_duration_histogram.png
    ├── travel_duration_histogram.png
    ├── avg_duration_by_service_type.png
    ├── avg_duration_by_location.png
    ├── duration_boxplot_by_service_type.png
    ├── duration_boxplot_by_location.png
    ├── top_bottom_nurses_travel.png
    └── avg_travel_duration_by_nurse.png
```

---

## Environment Setup

**Python version:** 3.8 or higher

Install all required libraries using pip:

```bash
pip install pandas numpy scipy matplotlib seaborn
```

---

## How to Run

Pass the path to the raw `visits.csv` file as a command-line argument:

```bash
python Data_analysis.py path/to/visits.csv
```

If no argument is provided, the script defaults to `~/Downloads/visits.csv`.

The script will:
1. Print a full pipeline log to the terminal (data quality checks, cleaning steps, analysis results)
2. Save the cleaned dataset as `visits_cleaned.csv` in the same directory as the scripts
3. Save all plots to a `visualizations/` folder in the same directory as the scripts

---

## Dataset Overview

| Attribute | Detail |
|---|---|
| Source file | `visits.csv` |
| Raw columns | `visit_id`, `patient_id`, `nurse_id`, `visit_start_time`, `visit_end_time`, `service_type`, `visit_location`, `nurse_notes` |
| Records after cleaning | 866 visits |
| Nurses | 100 unique nurses |
| Patients | 408 unique patients |
| Date range | 19 August 2025 – 18 September 2025 |
| Locations | North Zone, South Zone, East Zone, West Zone |
| Service types | Medication Administration, Wound Care, Physical Therapy, General Check-up |

---

## Part 1 – Data Preparation and Engineering

### 1.1 Initial Exploration

On load, the pipeline reports shape, data types, a full statistical summary, missing value counts, duplicate `visit_id` counts, and a list of all identified quality issues before any cleaning begins.

### 1.2 Data Cleaning Steps

**Duplicate removal**
Duplicate rows were identified by `visit_id` and removed, keeping the first occurrence.

**Visit ID standardisation**
All `visit_id` values were re-hashed to consistent 10-character alphanumeric strings using SHA-256 on the `(visit_id, patient_id)` pair, guaranteeing global uniqueness.

**Date/time standardisation**
`visit_start_time` and `visit_end_time` were parsed using `pd.to_datetime` with `format="mixed"` to handle inconsistent input formats. Rows where either timestamp could not be parsed were dropped.

**Service type standardisation**
Raw `service_type` values contained both abbreviated forms and misspellings. These were resolved in two passes:

- An explicit typo map corrected known misspellings (e.g. `Medicatn Adminstratino` → `Medication Administration`, `Pyhcisal Therapy` → `Physical Therapy`, `Wound Cae` → `Wound Care`, `General Chek-up` → `General Check-up`)
- A fuzzy matcher (`difflib.get_close_matches`, cutoff 0.75) caught any remaining variants not in the explicit map

In total, 69 rows had their service type corrected (19 + 20 + 16 + 14).

**Visit location standardisation**
Location values were normalised to canonical `X Zone` format (e.g. `n. zone`, `north` → `North Zone`) using a zone-name extractor and fuzzy fallback.

**Missing data treatment**

| Column | Treatment | Justification |
|---|---|---|
| `visit_location` | Filled with `Unknown Zone` | Preserves the row for duration analysis; flags it for downstream review |
| `nurse_notes` | Filled with empty string | Makes regex flag extraction safe without affecting other columns |
| `service_type` | Retained as NaN | Rows are excluded from service-level aggregations only |
| `visit_start_time` / `visit_end_time` | Rows dropped if either is NaT | Duration cannot be computed without both timestamps |

**Nurse notes text cleaning**
Noise tokens and symbol clusters were stripped from free-text notes using regex patterns covering: `####`, `***`, `$$$`, `>>><<<`, `///\\`, `~@#`, `xyz123`, `%%%^^^`, and any sequence of 2 or more non-alphanumeric symbols. Fragment phrases were also rewritten into proper sentences, for example:

| Before | After |
|---|---|
| `Critical is required.` | `A critical review is required.` |
| `Follow-up is required.` | `A follow-up appointment is required.` |
| `ASAP is required.` | `Immediate action is required.` |
| `Monitoring is required.` | `Ongoing monitoring is required.` |

### 1.3 Feature Engineering

**`visit_duration_minutes`**
Computed as `(visit_end_time - visit_start_time)` in minutes. Visits shorter than 1 minute or longer than 180 minutes were removed as implausible.

**`travel_duration_minutes`**
Computed as the gap between a nurse's previous visit end time and the current visit start time, within the same nurse and same calendar day. Values below 0 or above 240 minutes were nulled out as invalid. All duration values are rounded to 3 decimal places.

**`nurse_note_flags`**
Each nurse note was scanned against a set of keyword patterns and assigned a comma-separated list of clinical flags:

| Flag | Triggered by |
|---|---|
| `critical review required` | "a critical review is required" |
| `urgent intervention required` | "urgent intervention is required", "immediate action is required" |
| `follow up appointment required` | "a follow-up appointment is required" |
| `ongoing monitoring required` | "ongoing monitoring is required" |
| `clinical review required` | "a clinical review is required", "immediate action is required" |
| `wound or dressing complexity` | "wound", "dressing", "incision", "bandage" |
| `patient improving or stable` | "improving", "stable", "comfortable" |
| `patient distress` | "in pain", "weak", "dizzy", "restless" |

**`attention_priority_flag`**
A RAG (Red / Amber / Green) triage column derived from `nurse_note_flags`:

| Priority | Logic | Count |
|---|---|---|
| **RED** | Critical or urgent flag + patient distress + no improvement | 147 visits |
| **YELLOW** | Monitoring/follow-up + distress (no critical escalation), or critical + improving | 276 visits |
| **GREEN** | Monitoring/follow-up + improving, no distress, no critical escalation | 94 visits |
| **NONE** | No attention flags — routine or already resolved | 349 visits |

---

## Part 2 – Data Analysis and Visualisation

### 2.1 Descriptive Statistics

| Metric | Visit Duration (min) | Travel Duration (min) |
|---|---|---|
| Count | 866 | 31 |
| Mean | 63.07 | 124.43 |
| Std Dev | 33.10 | 66.72 |
| Min | 1.00 | 12.12 |
| 25th percentile | 34.55 | 63.89 |
| Median | 62.00 | 125.12 |
| 75th percentile | 92.00 | 177.73 |
| Max | 121.87 | 238.92 |

Travel duration has limited coverage (31 non-null values) as it requires at least two consecutive same-day visits by the same nurse.

### 2.2 Average Duration by Service Type

| Service Type | Avg Visit Duration (min) |
|---|---|
| Physical Therapy | 64.66 |
| Wound Care | 64.00 |
| Medication Administration | 63.07 |
| General Check-up | 60.46 |

**Longest:** Physical Therapy — hands-on sessions naturally require more time per visit.
**Shortest:** General Check-up — typically assessment-only with no procedure component.

### 2.3 Average Visit Duration by Location Zone

| Location | Avg Visit Duration (min) |
|---|---|
| North Zone | 65.58 |
| South Zone | 64.46 |
| East Zone | 63.76 |
| West Zone | 58.61 |

West Zone visits are notably shorter on average. This may reflect patient complexity differences, visit type mix, or more efficient routing in that area.

### 2.4 ANOVA — Significant Difference Across Zones?

A one-way ANOVA was performed to test whether visit duration differs significantly across zones.

- **F-statistic: 1.936**
- **p-value: 0.1221**

The result is **not statistically significant** at the 0.05 level, meaning the observed differences between zones cannot be confidently distinguished from random variation at this sample size.

### 2.5 Top 3 and Bottom 3 Nurses by Average Travel Duration

Overall fleet average travel duration: **124.43 minutes**

| Rank | Nurse ID | Avg Travel (min) | Observations |
|---|---|---|---|
| Top 1 | N2273 | 177.50 | 2 |
| Top 2 | N5881 | 166.62 | 2 |
| Top 3 | N7325 | 165.88 | 3 |
| Bottom 1 | N5275 | 72.27 | 2 |
| Bottom 2 | N8880 | 78.81 | 2 |
| Bottom 3 | N1148 | 107.99 | 2 |

Note: travel duration data is sparse in the current dataset (only 31 recorded values across 866 visits). Findings should be treated as indicative until more travel data is captured.

### 2.6 Nurse Notes Insights

Across 866 visits, the most common flag combinations were:

- **patient_improving_or_stable** (180 visits) — the largest single group, indicating positive care outcomes
- **ongoing_monitoring_required + patient_distress** (97 visits) — patients requiring continued attention
- **urgent_intervention_required + clinical_review_required + patient_distress** (79 visits) — the highest-severity group

Visits with distress flags (`patient_distress`) have a slightly higher average visit duration than visits without them, consistent with more complex care needs requiring additional nurse time.

---

## Part 3 – Operational Improvement Suggestions

**1. Prioritise high-complexity service types early in shifts**
Physical Therapy and Wound Care have the longest average durations. Scheduling these earlier in a nurse's day reduces the risk of cascading delays when earlier visits run over time.

**2. Investigate West Zone scheduling**
West Zone has the shortest average visit duration (58.61 min vs 65.58 min in North Zone). If this reflects genuine patient complexity differences it may allow for tighter scheduling in that zone; if it reflects rushed care delivery a clinical audit is warranted.

**3. Act on RED-flagged visits**
147 visits are flagged RED (critical or urgent alert with active distress and no improvement signal). A clinical review process should be triggered for these visits to confirm care was completed and outcomes were followed up.

**4. Route optimisation for high-travel nurses**
Nurses N2273, N5881, and N7325 average over 165 minutes of travel between consecutive visits — significantly above the fleet average of 124 minutes. Reviewing their zone assignments or introducing route optimisation tooling could recover meaningful hours per week.

**5. Expand travel duration data capture**
Only 31 of 866 visits have a recorded travel duration. This is because the metric requires consecutive same-day visits per nurse, which is rarely met in the current dataset. Ensuring nurses are scheduled with back-to-back visits where possible will both improve data coverage and reduce idle time.

**6. Use the RAG flag for proactive scheduling buffers**
Visits flagged YELLOW or RED could be automatically allocated a 10–15 minute scheduling buffer to account for the higher likelihood of extended care delivery, reducing the knock-on effect of overruns.

---

## Assumptions and Challenges

**Assumptions**
- Travel duration is defined as the gap between a nurse's previous visit end and the current visit start, same day and same nurse only. Cross-day travel is excluded as it cannot be reliably distinguished from overnight rest time.
- Visits shorter than 1 minute were treated as data entry errors and removed. Visits longer than 180 minutes were treated as implausible and removed.
- Travel gaps exceeding 240 minutes were nulled as they likely reflect scheduling gaps rather than genuine travel.
- Where `nurse_notes` was missing, an empty string was substituted to allow flag extraction to run without errors.

**Challenges**
- The raw dataset contained misspelled service type values (`Medicatn Adminstratino`, `Pyhcisal Therapy`, `Wound Cae`, `General Chek-up`) which required an explicit typo correction map on top of the fuzzy matcher.
- Nurse notes contained inconsistent noise symbols (`####`, `***`, `xyz123`, etc.) requiring a multi-pattern cleaning pass before text could be used for flag extraction.
- Travel duration coverage is very low (3.6% of visits) due to the sparse scheduling of consecutive same-day visits per nurse, limiting the reliability of nurse-level travel analysis.

---
**Note** 
- only 31 inter-visit gaps were recorded; per-nurse rankings are based on 2 observations each and should be treated as indicative.

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, transformation, and aggregation |
| `numpy` | Numerical operations and conditional column assignment |
| `scipy.stats` | One-way ANOVA for location zone significance testing |
| `matplotlib` | Plot rendering and figure management |
| `seaborn` | Statistical visualisations (histograms, bar charts, box plots) |
| `re` | Regex-based text cleaning and flag extraction |
| `difflib` | Fuzzy string matching for service type standardisation |
| `hashlib` | SHA-256 based visit ID generation |
