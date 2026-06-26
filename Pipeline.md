# Pipeline Overview (Plain English)

This project turns raw profile and job-history data into analysis-ready files in three major steps.

## Why There Are 3 Scripts

Each script has one clear job:

1. Build a clean base dataset.
2. Match companies to SCP legal-entity records.
3. Build final analysis panels and Stata outputs.

This ordering matters because each later step depends on files created earlier.

## Correct Run Order

The pipeline should run in this order:

1. `Create_Pandas_Datasets.py`
2. `Match_to_SCP_Data.py`
3. `Create_Stata_Analysis_Panel.py`

This is exactly how `run_pipeline.sh` is now arranged.

---

## Step 1: Create_Pandas_Datasets.py

### Goal
Create the core "all_experience" dataset from raw sources and engineer baseline features.

### What it does
- Loads and cleans education and experience records.
- Creates founder/owner style indicators.
- Adds franchise and cofounder flags.
- Merges company-level scores and other non-SCP enrichments.
- Saves the main file:
  - `all_experience_AnalysisFile_latest.pkl`

### Why it runs first
`Match_to_SCP_Data.py` needs this file as input. Without it, SCP matching cannot run.

---

## Step 2: Match_to_SCP_Data.py

### Goal
Find which companies in `all_experience` match SCP records, and capture SCP founding information.

### What it does
- Reads `all_experience_AnalysisFile_latest.pkl`.
- Reads `SCP_dataset_minimal.csv`.
- Normalizes company names so matching is more reliable.
- Picks the earliest SCP incorporation year per normalized company key.
- Outputs matched rows to:
  - `all_experience_AnalysisFile_scp.pkl`

### Why it runs second
It depends on the output from Step 1.

---

## Step 3: Create_Stata_Analysis_Panel.py

### Goal
Create person-level and event-level analysis datasets for Stata, including SCP-based timing windows.

### What it does
- Loads education + base experience data.
- Loads SCP match output from Step 2 (`all_experience_AnalysisFile_scp.pkl`).
- Merges SCP fields into the analysis frame.
- Creates 3/5/10-year variables using different founding-date logic:
  - `_o` = Original timing logic
  - `_p` = Page-based founding-year logic
  - `_s` = SCP-based founding-year logic (SCP-matched firms only)
- Exports Stata datasets and labeled variants.

### Why it runs last
It needs both:
- Base processed data from Step 1, and
- SCP match data from Step 2.

---

## In Short

Think of the pipeline as:

- Build clean base data ->
- Attach SCP legal/founding match data ->
- Produce final analysis outputs.

If you swap Steps 1 and 2, matching fails or becomes stale, because Step 2 depends on Step 1's latest output.
