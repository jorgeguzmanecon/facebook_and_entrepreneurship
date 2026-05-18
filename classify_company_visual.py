"""
Classify founder/owner company names as visual / non_visual / unknown
using OpenAI's gpt-4.1-nano model.

Usage:
    1. Set your API key:  export OPENAI_API_KEY="sk-..."
    2. Run:  python classify_company_visual.py

The script:
- Loads all_experience, extracts unique company names where is_founder_or_owner_with_url is True
- Sends batches to the API with retries and rate-limit handling
- Saves progress incrementally to a JSON-lines file so it can resume after interruption
- Produces a final pickle file that can be merged back into all_experience
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
import pdb

# --------------- configuration ------------------------------------------------
DATA_PATH = "/shared/share_scp/coresignal/all_experience_AnalysisFile_latest.pkl"
OUTPUT_DIR = Path("/shared/share_scp/coresignal/gitrepo_facebook2")
PROGRESS_FILE = OUTPUT_DIR / "classify_company_visual_progress.jsonl"
FINAL_OUTPUT = OUTPUT_DIR / "company_visual_classification.pkl"

MODEL = "gpt-4.1-nano"          # change if your account uses a different name
BATCH_SIZE = 500                 # company names per API call
MAX_RETRIES = 5
INITIAL_BACKOFF = 2              # seconds; doubles on each retry
REQUEST_DELAY = 0.2              # minimum seconds between requests
# ------------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You will receive a list of company names.

Task:
For each company, classify whether its likely primary offering is directly \
advertisable through a visual online post.

Label definitions:
- visual: the offering is typically easy to market through an image or short video
- non_visual: the offering is mainly intangible, technical, contractual, backend, \
or not easily understood visually
- unknown: the company name is too ambiguous to infer the offering

Instructions:
- Use only the company name and any text explicitly provided with it.
- Do not rely on outside knowledge about real firms.
- Prefer "unknown" over weak guessing.
- Confidence is the probability that your assigned label is correct given only \
the provided information.
- Provide a brief reason in 5–15 words.

Return valid JSON as:
[
  {
    "company_name": "...",
    "label": "visual | non_visual | unknown",
    "confidence": 0.00,
    "reason": "..."
  }
]

Return ONLY the JSON array. No markdown fences, no extra text."""


def load_completed_names(progress_path: Path) -> set:
    """Load company names already classified from the progress file."""
    done = set()
    if progress_path.exists():
        with open(progress_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done.add(rec["company_name"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def call_openai(client, names: list[str]) -> list[dict]:
    """Send one batch to the API and return parsed results."""
    user_msg = json.dumps(names)

    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3].strip()
            # Parse – the model may wrap the array in an object key
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                # find the first list value
                for v in parsed.values():
                    if isinstance(v, list):
                        parsed = v
                        break
                else:
                    # dict has no list value – treat each value as one record
                    # e.g. {"company_a": {"label": ...}, ...}
                    coerced = []
                    for k, v in parsed.items():
                        if isinstance(v, dict):
                            v.setdefault("company_name", k)
                            coerced.append(v)
                    if coerced:
                        parsed = coerced
            if isinstance(parsed, dict):
                # single-item response: wrap it
                parsed = [parsed]
            if not isinstance(parsed, list):
                print(f"  [WARN] Unexpected response type {type(parsed)}, raw: {raw[:200]}")
                return []
            return parsed

        except (ValueError, json.JSONDecodeError, KeyError) as e:
            # Parse/format error – retrying won't help
            print(f"  [PARSE ERROR] {type(e).__name__}: {e}")
            return []

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = "rate" in err_str or "429" in err_str
            is_server = "500" in err_str or "502" in err_str or "503" in err_str

            if attempt == MAX_RETRIES:
                print(f"  [FAIL] Batch failed after {MAX_RETRIES} attempts: {e}")
                return []

            if is_rate_limit or is_server:
                wait = backoff * (2 if is_rate_limit else 1)
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] {type(e).__name__}: sleeping {wait:.0f}s")
                time.sleep(wait)
                backoff *= 2
            else:
                print(f"  [RETRY {attempt}/{MAX_RETRIES}] {type(e).__name__}: {e}")
                time.sleep(backoff)
                backoff *= 2


def main():
    # --- lazy import so the script fails fast on missing key ----
    #api_key = os.environ.get("OPENAI_API_KEY")
    api_key = "NONE"
    if not api_key:
        raise RuntimeError(
            "Set OPENAI_API_KEY environment variable before running.\n"
            "  export OPENAI_API_KEY='sk-...'"
        )
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # --- load data ---
    print(f"Loading data from {DATA_PATH} ...")
    all_exp = pd.read_pickle(DATA_PATH)

    mask = all_exp["is_founder_or_owner_with_url"].fillna(False).astype(bool)
    unique_names = (
        all_exp.loc[mask, "company_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda s: s != ""]
        .unique()
    )
    print(f"Unique company names to classify: {len(unique_names):,}")

    # --- resume support ---
    already_done = load_completed_names(PROGRESS_FILE)
    remaining = [n for n in unique_names if n not in already_done]
    print(f"Already classified: {len(already_done):,}  |  Remaining: {len(remaining):,}")

    if not remaining:
        print("Nothing to do — all names are already classified.")
    else:
        # --- batch loop ---
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        for batch_idx in range(total_batches):
            start = batch_idx * BATCH_SIZE
            batch = remaining[start : start + BATCH_SIZE]

            print(f"Batch {batch_idx + 1}/{total_batches}  ({len(batch)} names) ...", end=" ", flush=True)
            results = call_openai(client, batch)

            # write progress
            with open(PROGRESS_FILE, "a") as f:
                for rec in results:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"got {len(results)} results")
            time.sleep(REQUEST_DELAY)

    # --- build final output ---
    print("Building final output ...")
    all_results = []
    with open(PROGRESS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    all_results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    df_class = pd.DataFrame(all_results)

    # Deduplicate keeping last (most recent run wins)
    df_class = df_class.drop_duplicates(subset=["company_name"], keep="last")

    df_class.to_pickle(FINAL_OUTPUT)
    print(f"Saved {len(df_class):,} classifications to {FINAL_OUTPUT}")
    print("Done.")


if __name__ == "__main__":
    main()
