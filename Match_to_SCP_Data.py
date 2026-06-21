#!/usr/bin/env python
# coding: utf-8

"""
Match_to_SCP_Data.py

Create a matched all_experience dataset against SCP firm records using normalized
company names. For each normalized SCP company key, keep the earliest
incorporation year and match all_experience to that earliest SCP record.

Output:
    all_experience_AnalysisFile_scp.pkl

Example:
    conda run -n jgpriv python Match_to_SCP_Data.py
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

import pandas as pd


DROP_TOKENS = {
    "a", "an", "the",
    "and", "&", "of", "for", "to",
    "co", "company",
    "inc", "incorporated", "corp", "corporation",
    "ltd", "limited", "llc", "plc", "lp", "llp",
    "gmbh", "ag", "sa", "sas", "sarl", "bv", "nv",
    "pte", "pty", "oy", "ab", "as",
}


@dataclass(frozen=True)
class Config:
    all_experience_path: str
    scp_path: str
    output_path: str


def normalize_company_name(name: object) -> object:
    """Normalize company names using deterministic exact-key rules."""
    if pd.isna(name):
        return pd.NA

    s = str(name).lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    tokens = [tok for tok in s.split() if tok and tok not in DROP_TOKENS]

    if not tokens:
        return pd.NA
    return " ".join(tokens)


def build_config(args: argparse.Namespace) -> Config:
    script_dir = os.path.dirname(os.path.abspath(__file__))

    all_experience_path = args.all_experience or os.path.join(
        script_dir, "all_experience_AnalysisFile_latest.pkl"
    )
    scp_path = args.scp or os.path.join(script_dir, "SCP_dataset_minimal.csv")
    output_path = args.output or os.path.join(
        script_dir, "all_experience_AnalysisFile_scp.pkl"
    )

    return Config(
        all_experience_path=all_experience_path,
        scp_path=scp_path,
        output_path=output_path,
    )


def load_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading all_experience:", cfg.all_experience_path)
    all_experience = pd.read_pickle(cfg.all_experience_path)
    print("Loading SCP data:", cfg.scp_path)
    scp_data = pd.read_csv(
        cfg.scp_path,
        on_bad_lines="skip",
        sep="\t",
        engine="python",
    )
    return all_experience, scp_data


def filter_founder_owner_rows(all_experience: pd.DataFrame) -> pd.DataFrame:
    owner_founder_columns = [
        col for col in all_experience.columns
        if "own" in col.lower() or "found" in col.lower()
    ]

    print(f"Potential founder/owner columns found: {len(owner_founder_columns)}")

    if not owner_founder_columns:
        print("No founder/owner columns found. Keeping all rows.")
        return all_experience.copy()

    mask = all_experience[owner_founder_columns].any(axis=1)
    filtered = all_experience[mask].copy()

    print(f"Rows before founder/owner filter: {len(all_experience):,}")
    print(f"Rows after founder/owner filter:  {len(filtered):,}")
    return filtered


def prepare_scp_earliest(scp_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    scp = scp_data.copy()

    # Parse incdate and create a robust numeric incyear for earliest-row selection.
    if "incdate" in scp.columns:
        scp["incdate"] = pd.to_datetime(scp["incdate"], format="%m/%d/%Y", errors="coerce")

    if "incyear" in scp.columns:
        scp["incyear"] = pd.to_numeric(scp["incyear"], errors="coerce")
    else:
        scp["incyear"] = pd.NA

    if "incdate" in scp.columns:
        scp["incyear_resolved"] = scp["incyear"].fillna(scp["incdate"].dt.year)
    else:
        scp["incyear_resolved"] = scp["incyear"]

    scp["scp_company_name_norm"] = scp["entityname"].map(normalize_company_name)

    scp_match_counts = (
        scp.loc[scp["scp_company_name_norm"].notna(), ["scp_company_name_norm", "entityname"]]
        .drop_duplicates()
        .groupby("scp_company_name_norm")["entityname"]
        .nunique()
    )

    scp_earliest = (
        scp.loc[scp["scp_company_name_norm"].notna()].copy()
        .sort_values(
            by=["scp_company_name_norm", "incyear_resolved", "incdate", "entityname"],
            na_position="last",
        )
        .drop_duplicates(subset=["scp_company_name_norm"], keep="first")
        .rename(
            columns={
                "entityname": "entityname_matched",
                "incyear_resolved": "scp_earliest_incyear",
            }
        )
    )

    keep_cols = [
        "scp_company_name_norm",
        "entityname_matched",
        "scp_earliest_incyear",
    ]
    if "jurisdiction" in scp_earliest.columns:
        keep_cols.append("jurisdiction")
    if "incdate" in scp_earliest.columns:
        keep_cols.append("incdate")

    return scp_earliest[keep_cols], scp_match_counts


def match_and_save(
    all_experience: pd.DataFrame,
    scp_earliest: pd.DataFrame,
    scp_match_counts: pd.Series,
    output_path: str,
) -> pd.DataFrame:
    ae = all_experience.copy()
    ae["cs_company_name_norm"] = ae["company_name"].map(normalize_company_name)

    ae["scp_total_matches_pre_dedup"] = (
        ae["cs_company_name_norm"].map(scp_match_counts).fillna(0).astype("int32")
    )
    ae["scp_has_match_pre_dedup"] = ae["scp_total_matches_pre_dedup"] > 0
    ae["scp_ambiguous_match_pre_dedup"] = ae["scp_total_matches_pre_dedup"] > 1

    matched = ae.merge(
        scp_earliest,
        left_on="cs_company_name_norm",
        right_on="scp_company_name_norm",
        how="left",
        validate="m:1",
    )

    matched_only = matched[matched["entityname_matched"].notna()].copy()

    matched_only.to_pickle(output_path)
    print(f"Saved matched output: {output_path}")

    return matched_only


def print_diagnostics(
    all_experience_raw: pd.DataFrame,
    all_experience_filtered: pd.DataFrame,
    matched_only: pd.DataFrame,
    scp_match_counts: pd.Series,
) -> None:
    total_raw = len(all_experience_raw)
    total_filtered = len(all_experience_filtered)
    total_matched = len(matched_only)

    print("\n" + "=" * 80)
    print("DIAGNOSTICS")
    print("=" * 80)
    print(f"Rows in all_experience (raw):              {total_raw:,}")
    print(f"Rows in all_experience (owner/founder):    {total_filtered:,}")
    print(f"Rows matched to SCP (saved):               {total_matched:,}")
    print(f"Match rate after filter:                   {total_matched / total_filtered:.2%}" if total_filtered else "Match rate after filter:                   n/a")

    n_valid_norm = all_experience_filtered["company_name"].map(normalize_company_name).notna().sum()
    print(f"Rows with valid normalized company name:   {n_valid_norm:,}")

    ambiguous_keys = (scp_match_counts > 1).sum()
    print(f"SCP normalized keys with >1 entity name:   {ambiguous_keys:,}")

    if total_matched:
        if "scp_earliest_incyear" in matched_only.columns:
            non_null_years = matched_only["scp_earliest_incyear"].dropna()
            print(f"Matched rows with non-null earliest year:  {len(non_null_years):,}")
            if not non_null_years.empty:
                print(f"Earliest matched incyear (min/max):        {int(non_null_years.min())} / {int(non_null_years.max())}")

        print("\nTop 10 matched SCP entity names:")
        print(matched_only["entityname_matched"].value_counts().head(10).to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Match all_experience firms to SCP using earliest incorporation year per normalized name."
    )
    parser.add_argument(
        "--all-experience",
        dest="all_experience",
        default=None,
        help="Path to all_experience pickle. Defaults to ./all_experience_AnalysisFile_latest.pkl",
    )
    parser.add_argument(
        "--scp",
        dest="scp",
        default=None,
        help="Path to SCP tab-delimited CSV. Defaults to ./SCP_dataset_minimal.csv",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=None,
        help="Output pickle path. Defaults to ./all_experience_AnalysisFile_scp.pkl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    all_experience_raw, scp_raw = load_data(cfg)
    all_experience_filtered = filter_founder_owner_rows(all_experience_raw)

    scp_earliest, scp_match_counts = prepare_scp_earliest(scp_raw)
    matched_only = match_and_save(
        all_experience=all_experience_filtered,
        scp_earliest=scp_earliest,
        scp_match_counts=scp_match_counts,
        output_path=cfg.output_path,
    )

    print_diagnostics(
        all_experience_raw=all_experience_raw,
        all_experience_filtered=all_experience_filtered,
        matched_only=matched_only,
        scp_match_counts=scp_match_counts,
    )


if __name__ == "__main__":
    main()
