#!/usr/bin/env python
# coding: utf-8

"""
Create_Pandas_Datasets.py

Refactored script version of Create_Pandas_Datasets.ipynb.
Organization follows the original notebook section titles and keeps print-heavy
progress updates and data previews.

Run format:
    conda run -n jgpriv python Create_Pandas_Datasets.py --start-section <N>

Where:
    <N> is one of 1, 2, 3, 4

Examples:
    conda run -n jgpriv python Create_Pandas_Datasets.py --start-section 1
    conda run -n jgpriv python Create_Pandas_Datasets.py --start-section 4
"""

from __future__ import annotations

import gc
import glob
import argparse
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd


# =============================
# Overview
# =============================


@dataclass(frozen=True)
class Config:
    coresignal_root: str = "/shared/share_scp/coresignal"
    repo_facebook: str = "/shared/share_scp/coresignal/gitrepo_facebook"
    repo_facebook2: str = "/shared/share_scp/coresignal/gitrepo_facebook2"

    education_glob: str = "processed_data2/coresignal_member_education_*linkedin*.pkl"
    experience_glob: str = "processed_data/coresignal_member_experience_*START*.pkl"

    education_analysis_path: str = "/shared/share_scp/coresignal/coresignal_member_education_AnalysisFile_latest.pkl"
    experience_analysis_path: str = "/shared/share_scp/coresignal/all_experience_AnalysisFile_latest.pkl"

    split_education_out: str = "/shared/share_scp/coresignal/gitrepo_facebook2/coresignal_member_education_AnalysisFile_latest.pkl.gz"
    split_experience_out: str = "/shared/share_scp/coresignal/gitrepo_facebook2/all_experience_AnalysisFile_latest.pkl.gz"

    company_scores_path: str = "/shared/share_scp/coresignal/company_marketing_scores_latest.pkl"
    companies_csv_path: str = "/shared/share_scp/coresignal/coresignal_company.csv"


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100, flush=True)


def preview_df(df: pd.DataFrame, name: str, n: int = 5) -> None:
    print(f"\n[{name}] shape={df.shape}")
    if df.empty:
        print(f"[{name}] DataFrame is empty")
        return
    print(df.head(n).to_string(index=False))


def preview_counts(series: pd.Series, name: str, n: int = 10) -> None:
    print(f"\n[{name}] top {n} value counts")
    vc = series.value_counts(dropna=False).head(n)
    print(vc.to_string())


def load_pickle_parallel(filepaths: list[str], max_workers: int = 8) -> list[pd.DataFrame]:
    def _read(path: str) -> pd.DataFrame:
        return pd.read_pickle(path)

    if not filepaths:
        return []

    workers = max(1, min(max_workers, len(filepaths)))
    print(f"Using {workers} workers for parallel pickle loading", flush=True)

    frames: list[pd.DataFrame] = []
    start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_file = {executor.submit(_read, f): f for f in filepaths}
        for i, future in enumerate(as_completed(future_to_file), start=1):
            frames.append(future.result())
            if i % 10 == 0 or i == len(filepaths):
                elapsed = time.time() - start
                rate = i / elapsed if elapsed > 0 else 0.0
                left = len(filepaths) - i
                eta = left / rate if rate > 0 else 0.0
                print(
                    f"Progress: {i:,}/{len(filepaths):,} files ({(100*i/len(filepaths)):.1f}%) | "
                    f"Rate: {rate:.1f}/sec | ETA: {eta:.0f}s | Elapsed: {elapsed:.0f}s",
                    flush=True,
                )
    return frames


# =============================
# Setup
# =============================


def setup_environment(cfg: Config) -> Any:
    section("Setup")

    if cfg.repo_facebook2 not in sys.path:
        sys.path.insert(0, cfg.repo_facebook2)

    if cfg.repo_facebook not in sys.path:
        sys.path.insert(0, cfg.repo_facebook)

    from helper_functions import to_pickle_split_by_size  # pylint: disable=import-error
    from university_name_matcher import university_name_matcher  # pylint: disable=import-error

    print(f"repo_facebook2: {cfg.repo_facebook2}")
    print(f"repo_facebook:  {cfg.repo_facebook}")
    print(f"coresignal_root: {cfg.coresignal_root}")

    umatcher = university_name_matcher()
    return to_pickle_split_by_size, umatcher


# =============================
# 1. Convert the raw education and experience files into analysis files
# =============================

# -----------------------------
# 1.1 Read the files on education experience
# -----------------------------


def load_university_reference(umatcher: Any) -> pd.DataFrame:
    section("1.1 Read the files on education experience - university reference")
    universities = umatcher.load_university_data()
    dup_names = universities[universities["instnm"].duplicated()]["instnm"]

    print("Number of duplicate university IDs:", universities[universities["instnm"].isin(dup_names)].shape[0])
    print("Number of NA in name:", universities["instnm"].isna().sum())
    preview_df(universities[["instnm"]].drop_duplicates().head(10), "universities sample", n=10)
    return universities


def load_education_data(cfg: Config) -> pd.DataFrame:
    section("1.1 Read the files on education experience - load education files")
    os.chdir(cfg.coresignal_root)
    files = sorted(glob.glob(cfg.education_glob))
    print(f"Found {len(files):,} education files")
    if not files:
        raise FileNotFoundError(f"No education files matched pattern: {cfg.education_glob}")

    education = pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
    print(f"Loaded education rows: {len(education):,}")

    if "member_id" in education.columns:
        education["member_id"] = pd.to_numeric(education["member_id"], errors="coerce").astype("Int64")
        print(f"Unique member IDs in education: {education['member_id'].nunique(dropna=True):,}")

    before = len(education)
    education = education.drop_duplicates()
    print(f"Dropped exact duplicate rows: {before - len(education):,}")

    preview_df(education, "education raw")
    return education


def clean_education_data(education: pd.DataFrame) -> pd.DataFrame:
    section("1.1 Read the files on education experience - clean education")
    out = education.copy()

    if "title" in out.columns:
        before = len(out)
        out = out[
            ~out["title"].str.contains("university of phoenix|devry university", case=False, na=False)
        ]
        print(f"Filtered phoenix/devry rows: {before - len(out):,}")

    if "id" in out.columns:
        dups = out.duplicated(["id"]).sum()
        print(f"Duplicate IDs before filtering: {dups:,}")
        out = out[~out.duplicated(["id"])]
        print(f"Rows after ID de-dup: {len(out):,}")

    if "school_url" in out.columns:
        before = len(out)
        out = out[~out["school_url"].astype(str).str.endswith("linkedin.com/edu/school", na=False)]
        print(f"Filtered generic school_url rows: {before - len(out):,}")

    if "subtitle" in out.columns:
        out = out[out["subtitle"].notna()].copy()

        mask_ba_ma = (
            out["subtitle"].str.contains(r"Bachelor'?s?|Undergrad|\sB\.?A\.?|B\.?S\.?", na=False)
            | out["subtitle"].str.contains(r"Master'?s|MS|M\.S|MBA|M\.B\.A", na=False)
        )
        mask = mask_ba_ma
        mask &= ~out["subtitle"].str.contains(r"Associate", na=False)
        mask &= ~out["subtitle"].str.contains(r"Ph\.?D\.?|Doctor|MD|M\.D\.", na=False)
        mask &= ~out["subtitle"].str.contains(r"Nursing|Medical", na=False)
        mask &= ~out["subtitle"].str.contains(r"Law|J\.?D\.?", na=False)
        mask &= ~out["subtitle"].str.contains(r"Technician", na=False)

        before = len(out)
        out = out[mask].copy()
        print(f"Filtered to bachelors/masters profile rows: {before:,} -> {len(out):,}")

    preview_df(out, "education cleaned")
    if "instnm" in out.columns:
        preview_counts(out["instnm"], "top institutions", n=15)

    gc.collect()
    return out


# -----------------------------
# 1.2 Read the files on employment experience
# -----------------------------


def load_experience_data(cfg: Config, education: pd.DataFrame) -> pd.DataFrame:
    section("1.2 Read the files on employment experience")
    os.chdir(cfg.coresignal_root)

    files = sorted(glob.glob(cfg.experience_glob))
    print(f"Found {len(files):,} experience files")
    if not files:
        raise FileNotFoundError(f"No experience files matched pattern: {cfg.experience_glob}")

    frames = load_pickle_parallel(files, max_workers=os.cpu_count() or 8)
    experience = pd.concat(frames, ignore_index=True)
    print(f"All experience rows before filtering: {len(experience):,}")

    if "member_id" in experience.columns and "member_id" in education.columns:
        before = len(experience)
        keep_ids = set(education["member_id"].dropna().astype(int).tolist())
        experience = experience[experience["member_id"].isin(keep_ids)]
        print(f"Filtered to education-member_id set: {before:,} -> {len(experience):,}")

    if "id" in experience.columns:
        before = len(experience)
        experience = experience.drop_duplicates(subset=["id"])
        print(f"Dropped duplicate experience IDs: {before - len(experience):,}")

    preview_df(experience, "experience cleaned base")
    return experience


# -----------------------------
# 1.3 Testing and validations
# -----------------------------


def run_validations(education: pd.DataFrame, experience: pd.DataFrame) -> None:
    section("1.3 Testing and validations")

    print("Education memory (GB):", round(education.memory_usage(deep=True).sum() / 1e9, 4))
    print("Experience memory (GB):", round(experience.memory_usage(deep=True).sum() / 1e9, 4))

    if "year_to" in education.columns:
        yr = education["year_to"].dropna()
        if not yr.empty:
            print(f"Education year_to range: {int(yr.min())} - {int(yr.max())}")

    if "company_name" in experience.columns:
        preview_counts(experience["company_name"].astype(str), "top experience companies", n=20)


# -----------------------------
# 1.4 Output and save the analysis files
# -----------------------------


def save_analysis_files(
    cfg: Config,
    to_pickle_split_by_size: Any,
    education: pd.DataFrame,
    experience: pd.DataFrame,
) -> None:
    section("1.4 Output and save the analysis files")

    education.to_pickle(cfg.education_analysis_path)
    print(f"Saved education analysis pickle: {cfg.education_analysis_path}")
    to_pickle_split_by_size(education, cfg.split_education_out)
    print(f"Saved split education pickle: {cfg.split_education_out}")

    experience.to_pickle(cfg.experience_analysis_path)
    print(f"Saved experience analysis pickle: {cfg.experience_analysis_path}")
    to_pickle_split_by_size(experience, cfg.split_experience_out)
    print(f"Saved split experience pickle: {cfg.split_experience_out}")


# =============================
# 2. Add additional information for the all_experience dataset.
# =============================

# -----------------------------
# 2.1 Load file
# -----------------------------


def load_all_experience(cfg: Config) -> pd.DataFrame:
    section("2.1 Load file")
    df = pd.read_pickle(cfg.experience_analysis_path)
    print(f"Loaded all_experience rows: {len(df):,}")

    if "date_from" in df.columns:
        before = len(df)
        df = df[~df["date_from"].isna()].copy()
        print(f"Dropped null date_from rows: {before - len(df):,}")

    preview_df(df, "all_experience loaded")
    return df


# -----------------------------
# 2.2 Create owner variables etc
# 2.2.1 Founder/owner construction — REVISED 2026-06-02 (v3)
# -----------------------------


def build_founder_owner_flags(df: pd.DataFrame) -> pd.DataFrame:
    section("2.2.1 Founder/owner construction — REVISED 2026-06-02 (v3)")
    out = df.copy()

    out["title"] = out["title"].astype(str)
    out["job_from"] = out["date_from"].astype(str).str.extract(r"(\d{4})").astype(float)

    out["is_founder_only_raw"] = out["title"].str.contains("founder", case=False, na=False)
    out["is_owner_only_raw"] = out["title"].str.contains(r"\bowner\b", case=False, na=False)
    out["is_cofounder_only_title_raw"] = out["title"].str.contains(r"co[-\s]?founder", case=False, na=False)
    out["is_coowner_only_title_raw"] = out["title"].str.contains(r"co[-\s]?owner", case=False, na=False)

    title = out["title"]

    mask_not_real_owner = (
        title.str.contains(r"\b(?:product|assistant\s+to|business\s+process|program|process)\s+owner\b", case=False, na=False)
        | title.str.contains(
            r"\b(?:system|service|application|content|account|feature|platform|portfolio|"
            r"project|brand|risk|policy|capability|workstream|domain|module|epic|"
            r"sprint|asset|story|test|quality|knowledge|workflow|outcome|stakeholder|"
            r"client|customer|architecture|document|incident|problem|requirement|"
            r"specification|release|change|metric|model|data|technology|technical|"
            r"it|solution|experience|engagement)\s+owner\b",
            case=False,
            na=False,
        )
        | title.str.contains(
            r"\bowner\s+(?:services?|relations?|support|liaison|portal|advocate|"
            r"representative|advisor|adviser|consultant|coordinator|specialist|"
            r"claims?|builder|architect|engagement|experience|onboarding|coach|"
            r"recruiter|recruitment)\b",
            case=False,
            na=False,
        )
        | title.str.contains(
            r"owner'?s?\s+(?:assistant|representative|advocate|aide)|"
            r"\bassistant\s+(?:to\s+)?(?:the\s+)?owner|"
            r"\bto\s+the\s+owner\b|"
            r"\baide\s+to\s+(?:the\s+)?owner",
            case=False,
            na=False,
            regex=True,
        )
        | title.str.contains(
            r"\b(?:future|aspiring|wannabe|wanna[ -]?be|prospective|hopeful|soon[ -]?to[ -]?be)\s+(?:co[- ]?)?owner",
            case=False,
            na=False,
        )
        | title.str.contains(r"\b(?:pet|dog|cat|horse|reptile|parent)\s+owner|\bpet[ -]?parent", case=False, na=False)
        | title.str.contains(r"\bemployee[- ]?owner|\besop\s+owner", case=False, na=False)
    )

    mask_not_real_founder = (
        title.str.contains(r"\bfounding\s+(?:member|engineer|partner|team|employee|associate|leader)", case=False, na=False)
        | title.str.contains(
            r"\bfounders\s+(?:fellow|fellowship|award|forum|institute|grant|scholarship|"
            r"fund|club|program|society|circle|guild|associate|associates|day|"
            r"committee|board|meeting|event|50|100)",
            case=False,
            na=False,
        )
        | title.str.contains(
            r"\b(?:future|aspiring|wannabe|wanna[ -]?be|prospective|hopeful|soon[ -]?to[ -]?be)\s+(?:co[- ]?)?founder",
            case=False,
            na=False,
        )
        | title.str.contains(
            r"\bassistant\s+(?:to\s+)?(?:the\s+)?(?:co[- ]?)?founder|"
            r"\bto\s+the\s+(?:co[- ]?)?founder|"
            r"\baide\s+to\s+(?:the\s+)?founder|"
            r"founder'?s\s+(?:assistant|aide)",
            case=False,
            na=False,
            regex=True,
        )
        | title.str.contains(r"\bconfounder", case=False, na=False)
    )

    weak_founder_patterns = [
        r"\bto\s+(?:the\s+)?(?:co[- ]?)?founder\b",
        r"\b(?:future|aspiring|wannabe|wanna[ -]?be|prospective|hopeful|soon[ -]?to[ -]?be)\s+(?:co[- ]?)?founder\b",
        r"founder'?s\s+(?:assistant|aide)",
        r"\baide\s+to\s+(?:the\s+)?founder\b",
        r"\bassistant\s+(?:to\s+)?(?:the\s+)?(?:co[- ]?)?founder\b",
        r"\bfounders\s+(?:fellow|fellowship|award|forum|institute|grant|scholarship|fund|club|program|society|circle|guild|associate|associates|day|committee|board|meeting|event|50|100)",
        r"\bconfounder\b",
    ]

    title_stripped = title.copy()
    for pat in weak_founder_patterns:
        title_stripped = title_stripped.str.replace(pat, "", regex=True, case=False)

    has_strong_founder = title_stripped.str.contains(r"\b(?:co[-\s]?)?founder\b", case=False, na=False, regex=True)
    mask_not_real_founder = mask_not_real_founder & ~has_strong_founder

    out["is_founder_only"] = out["is_founder_only_raw"] & ~mask_not_real_founder
    out["is_owner_only"] = out["is_owner_only_raw"] & ~mask_not_real_owner
    out["is_cofounder_only_title"] = out["is_cofounder_only_title_raw"] & ~mask_not_real_founder
    out["is_coowner_only_title"] = out["is_coowner_only_title_raw"] & ~mask_not_real_owner

    out["is_founder_or_owner"] = out["is_founder_only"] | out["is_owner_only"]
    out["is_cofounder_or_coowner_title"] = out["is_cofounder_only_title"] | out["is_coowner_only_title"]
    out["is_founder_or_owner_with_url"] = out["is_founder_or_owner"] & out["company_url"].notna()

    out["is_founder_or_owner_inc"] = out["is_founder_or_owner"] & out["company_name"].astype(str).str.contains(
        r"\b(?:inc|corp|corporation|co|incorporated)\b", na=False, case=False
    )

    out = out.drop(
        columns=[
            "is_founder_only_raw",
            "is_owner_only_raw",
            "is_cofounder_only_title_raw",
            "is_coowner_only_title_raw",
        ],
        errors="ignore",
    )

    print("Founder/Owner flag means:")
    print(
        out[
            [
                "is_founder_only",
                "is_owner_only",
                "is_founder_or_owner",
                "is_founder_or_owner_with_url",
            ]
        ]
        .mean(numeric_only=True)
        .to_string()
    )

    sample_cols = [c for c in ["title", "company_name", "is_founder_only", "is_owner_only", "is_founder_or_owner"] if c in out.columns]
    if sample_cols:
        print("\nFounder/owner sample rows:")
        print(out[sample_cols].head(20).to_string(index=False))

    return out


# -----------------------------
# 2.1.3 Create flag for franchising
# -----------------------------


def create_franchise_flags(all_experience: pd.DataFrame) -> pd.DataFrame:
    section("2.1.3 Create flag for franchising")
    out = all_experience.copy()

    out["company_name"] = out["company_name"].astype("string")

    founders = (
        out[out["is_founder_only"]]
        .groupby("company_name", observed=True)["member_id"]
        .nunique()
        .rename("num_unique_founders")
    )

    owners = (
        out[out["is_owner_only"]]
        .groupby("company_name", observed=True)["member_id"]
        .nunique()
        .rename("num_unique_owners")
    )

    founder_or_owner = (
        out[out["is_founder_only"] | out["is_owner_only"]]
        .groupby("company_name", observed=True)["member_id"]
        .nunique()
        .rename("num_unique_founders_or_owners")
    )

    company_ownership = (
        pd.concat([founders, owners, founder_or_owner], axis=1)
        .fillna(0)
        .reset_index()
        .sort_values("num_unique_founders_or_owners", ascending=False)
    )

    company_ownership = company_ownership[
        company_ownership["company_name"].notna() & (company_ownership["company_name"] != "-")
    ]

    company_ownership = company_ownership[
        ~company_ownership["company_name"].str.match(
            r"self[-\s]?employed|^freelance$|^consultant$|^independent consultant$|^business owner$|^private practice$|^stealth startup$|^stealth|^startup|^entrepreneur$",
            case=False,
            na=False,
        )
    ]

    company_ownership = company_ownership[
        ~company_ownership["company_name"].isin([".", "owner", "confidential", "none"])
    ]

    company_ownership["share_founders_vs_owners"] = (
        company_ownership["num_unique_founders"]
        / company_ownership["num_unique_founders_or_owners"].replace(0, pd.NA)
    )

    company_ownership = company_ownership[~(company_ownership["share_founders_vs_owners"] > 0.7)]

    franchises = (
        (company_ownership["num_unique_founders_or_owners"] > 15)
        & (company_ownership["share_founders_vs_owners"] < 0.1)
    )

    franchise_company_names = set(company_ownership.loc[franchises, "company_name"].dropna().tolist())
    print(f"Franchise-like companies flagged: {len(franchise_company_names):,}")

    out["franchise_founder_or_owner"] = out["company_name"].isin(franchise_company_names) & out["is_founder_or_owner"]
    out["franchise_owner"] = out["company_name"].isin(franchise_company_names) & out["is_owner_only"]

    print("Franchise flag means:")
    print(out[["franchise_founder_or_owner", "franchise_owner"]].mean(numeric_only=True).to_string())

    return out


# -----------------------------
# 2.1.4 Develop different definitions of co-owner / co-founder.
# -----------------------------


def create_cofounder_flags(all_experience: pd.DataFrame) -> pd.DataFrame:
    section("2.1.4 Develop different definitions of co-owner / co-founder")
    out = all_experience.copy()

    out["company_url"] = out["company_url"].astype("string")
    found_own = out[out["is_founder_or_owner"]]

    ownership_by_url = (
        found_own.groupby("company_url", as_index=False, observed=True)
        .agg({"is_founder_only": "sum", "is_owner_only": "sum"})
        .sort_values(["is_founder_only", "is_owner_only"], ascending=False)
    )

    ownership_by_url = ownership_by_url[ownership_by_url["company_url"].notna()].copy()
    ownership_by_url["total_founders_owners"] = ownership_by_url["is_founder_only"] + ownership_by_url["is_owner_only"]

    cofounder_companies = ownership_by_url[ownership_by_url["total_founders_owners"].between(2, 10)]
    cofounder_urls = set(cofounder_companies["company_url"].tolist())

    out["cofound_coown_same_url"] = out["is_founder_or_owner"] & out["company_url"].isin(cofounder_urls)
    print(f"Rows flagged cofound_coown_same_url: {int(out['cofound_coown_same_url'].sum()):,}")

    return out


# -----------------------------
# 2.4 Make all_experience a smaller dataset
# -----------------------------


def shrink_experience_dataset(all_experience: pd.DataFrame) -> pd.DataFrame:
    section("2.4 Make all_experience a smaller dataset")
    out = all_experience.copy()

    columns_to_drop = [
        "description",
        "order_in_profile",
        "deleted",
        "last_updated",
        "created",
        "Index",
        "location",
    ]

    for col in columns_to_drop:
        if col in out.columns:
            out = out.drop(columns=[col])
            print(f"Dropped column: {col}")

    if "id" in out.columns:
        out = out.set_index("id", drop=True)
        print("Set index to id")

    mem_gb = out.memory_usage(deep=True).sum() / 1e9
    print(f"Memory usage after shrink: {mem_gb:.3f} GB")
    return out


# -----------------------------
# 2.x Add the continuous measure of add likelihood
# -----------------------------


def merge_company_scores(cfg: Config, all_experience: pd.DataFrame) -> pd.DataFrame:
    section("2.x Add the continuous measure of ad likelihood")
    out = all_experience.copy()

    if not os.path.exists(cfg.company_scores_path):
        print(f"company_scores file not found, skipping merge: {cfg.company_scores_path}")
        return out

    scores = pd.read_pickle(cfg.company_scores_path)
    print(f"Loaded company scores rows: {len(scores):,}")

    before_cols = set(out.columns)
    out = pd.merge(out, scores, on=["company_name"], how="left")
    new_cols = sorted(set(out.columns) - before_cols)
    print(f"Merged company scores. Added columns: {new_cols[:10]}{' ...' if len(new_cols) > 10 else ''}")

    return out


# =============================
# 3. Incorporate company founding date from URL
# =============================


def clean_url_col(df: pd.DataFrame, col: str) -> None:
    df[col] = df[col].astype("string")
    mask = df[col].notna()

    print(f"Cleaning {col}: {int(mask.sum()):,} non-null entries", flush=True)
    unique_urls = pd.Index(df.loc[mask, col].unique())
    print(f"Unique {col}: {len(unique_urls):,}", flush=True)

    cleaned = (
        pd.Series(unique_urls, index=unique_urls)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[\?#].*$", "", regex=True)
        .str.replace(r"^.*\.linkedin\.com", "linkedin.com", regex=True)
        .str.replace(r"/$", "", regex=True)
    )

    df.loc[mask, col] = df.loc[mask, col].map(cleaned)


def merge_company_founding_year(cfg: Config, all_experience: pd.DataFrame) -> pd.DataFrame:
    section("3. Incorporate company founding date from URL")

    out = all_experience.copy()
    if not os.path.exists(cfg.companies_csv_path):
        print(f"companies CSV not found, skipping founding-year merge: {cfg.companies_csv_path}")
        return out

    start = time.perf_counter()
    companies = pd.read_csv(cfg.companies_csv_path)
    print(f"Loaded companies rows: {len(companies):,}")

    clean_url_col(companies, "url")
    clean_url_col(out, "company_url")

    companies = companies[companies["url"].notna()].copy()
    companies["founded"] = pd.to_numeric(companies["founded"], errors="coerce")

    companies_merge = (
        companies.groupby("url", as_index=False)["founded"]
        .min()
        .rename(columns={"url": "company_url", "founded": "company_page_founded_year"})
    )

    if "company_page_founded_year" not in out.columns:
        out = out.merge(companies_merge, on="company_url", how="left")

    bad = out["company_page_founded_year"].notna() & ~out["company_page_founded_year"].between(1900, 2030)
    out.loc[bad, "company_page_founded_year"] = pd.NA

    elapsed = time.perf_counter() - start
    print(f"Merged founding year in {timedelta(seconds=round(elapsed))}")
    print(f"Non-null company_page_founded_year: {int(out['company_page_founded_year'].notna().sum()):,}")

    return out


# =============================
# 4. Do filtering of founder measures following Isaac's idea
# =============================


def apply_isaac_filter(all_experience: pd.DataFrame) -> pd.DataFrame:
    section("4. Do filtering of founder measures following Isaac's idea")
    out = all_experience.copy()

    ih_keywords = ["entrepreneur", r"(co)?[\- ]?founder", r"(co)?[\- ]?owner"]
    out["ih_is_founder_or_owner"] = out["title"].astype(str).str.contains("|".join(ih_keywords), case=False, na=False)

    if "flt_company_employee_start_year" not in out.columns:
        out["founder_job_from"] = out["job_from"].where(out["is_founder_or_owner"].fillna(False))

        company_start_dates = (
            out.groupby("company_name", as_index=False)
            .agg({"job_from": "min", "founder_job_from": "min"})
            .rename(
                columns={
                    "job_from": "flt_company_employee_start_year",
                    "founder_job_from": "flt_company_founder_start_year",
                }
            )
        )
        out = pd.merge(out, company_start_dates, on="company_name", how="left").reset_index(drop=True)

    out["flt_employee_before_founder"] = out["flt_company_employee_start_year"] < out["flt_company_founder_start_year"]
    out["flt_is_founder_or_owner"] = out["is_founder_or_owner"].where(~out["flt_employee_before_founder"], False)
    out["ih_flt_is_founder_or_owner"] = out["ih_is_founder_or_owner"].where(~out["flt_employee_before_founder"], False)

    cols = [
        "is_founder_or_owner",
        "flt_is_founder_or_owner",
        "ih_flt_is_founder_or_owner",
        "is_founder_or_owner_with_url",
    ]
    existing_cols = [c for c in cols if c in out.columns]

    print("Means (share True):")
    print(out[existing_cols].mean(numeric_only=True).to_string())

    return out


# =============================
# Save helper
# =============================


def persist_all_experience(cfg: Config, to_pickle_split_by_size: Any, all_experience: pd.DataFrame) -> None:
    all_experience.to_pickle(cfg.experience_analysis_path)
    print(f"Saved all_experience: {cfg.experience_analysis_path}")
    to_pickle_split_by_size(all_experience, cfg.split_experience_out)
    print(f"Saved split all_experience: {cfg.split_experience_out}")


@dataclass
class DatasetRepository:
    """Centralizes read/write operations for intermediate and final datasets."""

    cfg: Config
    to_pickle_split_by_size: Any

    def save_analysis_files(self, education: pd.DataFrame, experience: pd.DataFrame) -> None:
        save_analysis_files(self.cfg, self.to_pickle_split_by_size, education, experience)

    def load_all_experience(self) -> pd.DataFrame:
        return load_all_experience(self.cfg)

    def persist_all_experience(self, all_experience: pd.DataFrame) -> None:
        persist_all_experience(self.cfg, self.to_pickle_split_by_size, all_experience)


@dataclass
class SectionProcessors:
    """Groups section-level transformations using notebook-aligned steps."""

    cfg: Config
    umatcher: Any
    repository: DatasetRepository

    def run_section_1(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        section("Pipeline Section 1")
        _ = load_university_reference(self.umatcher)
        education = load_education_data(self.cfg)
        education = clean_education_data(education)

        experience = load_experience_data(self.cfg, education)
        run_validations(education, experience)
        self.repository.save_analysis_files(education, experience)
        return education, experience

    def run_section_2(self) -> pd.DataFrame:
        section("Pipeline Section 2")
        all_experience = self.repository.load_all_experience()
        all_experience = build_founder_owner_flags(all_experience)
        all_experience = create_franchise_flags(all_experience)
        all_experience = create_cofounder_flags(all_experience)
        all_experience = shrink_experience_dataset(all_experience)
        all_experience = merge_company_scores(self.cfg, all_experience)
        self.repository.persist_all_experience(all_experience)
        return all_experience

    def run_section_3(self, all_experience: pd.DataFrame) -> pd.DataFrame:
        section("Pipeline Section 3")
        all_experience = merge_company_founding_year(self.cfg, all_experience)
        self.repository.persist_all_experience(all_experience)
        return all_experience

    def run_section_4(self, all_experience: pd.DataFrame) -> pd.DataFrame:
        section("Pipeline Section 4")
        all_experience = apply_isaac_filter(all_experience)
        self.repository.persist_all_experience(all_experience)
        return all_experience


@dataclass
class CoresignalDatasetPipeline:
    """Orchestrates the script using notebook-aligned section steps."""

    cfg: Config
    to_pickle_split_by_size: Any
    umatcher: Any
    repository: DatasetRepository = field(init=False)
    processors: SectionProcessors = field(init=False)

    def __post_init__(self) -> None:
        self.repository = DatasetRepository(self.cfg, self.to_pickle_split_by_size)
        self.processors = SectionProcessors(self.cfg, self.umatcher, self.repository)

    def run_section_1(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.processors.run_section_1()

    def run_section_2(self) -> pd.DataFrame:
        return self.processors.run_section_2()

    def run_section_3(self, all_experience: pd.DataFrame) -> pd.DataFrame:
        return self.processors.run_section_3(all_experience)

    def run_section_4(self, all_experience: pd.DataFrame) -> pd.DataFrame:
        return self.processors.run_section_4(all_experience)

    def run_all(self) -> pd.DataFrame:
        self.run_section_1()
        all_experience = self.run_section_2()
        all_experience = self.run_section_3(all_experience)
        all_experience = self.run_section_4(all_experience)
        return all_experience

    def run_from_section(self, start_section: int = 1) -> pd.DataFrame:
        """Run pipeline starting at a given section number (1-4)."""
        if start_section not in {1, 2, 3, 4}:
            raise ValueError(f"Invalid start_section={start_section}. Choose from 1, 2, 3, 4.")

        print(f"Starting pipeline from section {start_section}")

        if start_section == 1:
            return self.run_all()

        if start_section == 2:
            all_experience = self.run_section_2()
            all_experience = self.run_section_3(all_experience)
            all_experience = self.run_section_4(all_experience)
            return all_experience

        all_experience = self.repository.load_all_experience()
        if start_section == 3:
            all_experience = self.run_section_3(all_experience)
            all_experience = self.run_section_4(all_experience)
            return all_experience

        all_experience = self.run_section_4(all_experience)
        return all_experience


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Create_Pandas_Datasets pipeline with section controls."
    )
    parser.add_argument(
        "--start-section",
        type=int,
        default=1,
        choices=[1, 2, 3, 4],
        help="Section number to start from (1-4). Example: --start-section 4",
    )
    return parser.parse_args()


# =============================
# Main
# =============================


def main() -> None:
    args = parse_args()
    cfg = Config()
    to_pickle_split_by_size, umatcher = setup_environment(cfg)

    pipeline = CoresignalDatasetPipeline(
        cfg=cfg,
        to_pickle_split_by_size=to_pickle_split_by_size,
        umatcher=umatcher,
    )
    all_experience = pipeline.run_from_section(start_section=args.start_section)

    section("Pipeline complete")
    print(f"Final rows: {len(all_experience):,}")
    print(f"Final columns: {len(all_experience.columns):,}")


if __name__ == "__main__":
    os.chdir("/shared/share_scp/coresignal")
    main()
