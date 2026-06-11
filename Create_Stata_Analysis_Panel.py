#!/usr/bin/env python
# coding: utf-8

"""
Create_Stata_Analysis_Panel.py

Refactored script version of Create_Stata_Analysis_Panel.ipynb.
This script is organized by notebook section titles and uses classes to keep
logic modular and maintainable.

Run format:
  conda run -n jgpriv python Create_Stata_Analysis_Panel.py --start-section <N>

Where:
  <N> in {2, 3}

Examples:
  conda run -n jgpriv python Create_Stata_Analysis_Panel.py --start-section 2
  conda run -n jgpriv python Create_Stata_Analysis_Panel.py --start-section 3
"""

from __future__ import annotations

import argparse
import gc
import os
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100, flush=True)


@dataclass(frozen=True)
class Config:
    coresignal_root: str = "/shared/share_scp/coresignal"

    education_path: str = "coresignal_member_education_AnalysisFile_latest.pkl"
    experience_path: str = "all_experience_AnalysisFile_latest.pkl"
    member_profiles_path: str = "coresignal_member_profiles_AnalysisFile_latest.pkl"

    company_marketing_scores_path: str = "/shared/share_scp/coresignal/company_marketing_scores_latest.pkl"


@dataclass
class DataRepository:
    cfg: Config

    def set_cwd(self) -> None:
        os.chdir(self.cfg.coresignal_root)
        print(f"Working directory: {os.getcwd()}")

    def load_core_datasets(self) -> dict[str, pd.DataFrame]:
        section("2.1.1 Load datasets")
        self.set_cwd()

        print("Loading coresignal_member_education...", end=" ")
        education = pd.read_pickle(self.cfg.education_path)
        print("Loaded")

        print("Loading all_experience...", end=" ")
        experience = pd.read_pickle(self.cfg.experience_path)
        print("Loaded")

        print("Loading coresignal_member_profiles...", end=" ")
        members = pd.read_pickle(self.cfg.member_profiles_path)
        print("Loaded")

        return {
            "education": education,
            "experience": experience,
            "members": members,
        }

    def load_graduate_job_level(self, today_str: str) -> pd.DataFrame:
        path = f"graduates_with_education_job_level_{today_str}.pkl"
        print(f"Loading graduate-job-level file: {path}")
        return pd.read_pickle(path)

    def save_graduate_job_level(self, df: pd.DataFrame, today_str: str) -> str:
        path = f"graduates_with_education_job_level_{today_str}.pkl"
        df.to_pickle(path)
        print(f"Saved: {path}")
        return path

    def save_graduates_person_level_pickle(self, df: pd.DataFrame, today_str: str) -> str:
        path = f"graduates_person_level_with_majors_{today_str}.pkl"
        df.to_pickle(path)
        print(f"Saved: {path}")
        return path

    def save_graduates_person_level_stata(self, df: pd.DataFrame, today_str: str) -> str:
        path = f"graduates_person_level_{today_str}.dta"
        df.to_stata(path, version=118, write_index=False)
        print(f"Saved: {path}")
        return path

    def save_graduates_person_level_labeled_stata(
        self,
        df: pd.DataFrame,
        today_str: str,
        variable_labels: dict[str, str],
    ) -> str:
        path = f"graduates_person_level_with_labels_{today_str}.dta"
        df.to_stata(path, variable_labels=variable_labels, version=118, write_index=False)
        print(f"Saved: {path}")
        return path

    def save_cofounder_events_stata(self, df: pd.DataFrame, today_str: str, clean: bool = False) -> str:
        suffix = "clean_" if clean else ""
        path = f"cofounder_events_{suffix}{today_str}.dta"
        df.to_stata(path, version=118)
        print(f"Saved: {path}")
        return path


@dataclass
class MajorCategorizer:
    majors_categories: dict[str, dict[str, Any]] = field(default_factory=dict)

    @staticmethod
    def default_categories() -> dict[str, dict[str, Any]]:
        x = [
            "asian", "hispanic", "african", "latin american", "gender", "feminist",
            "asian american", "african american", "frech", "russian", "middle eastern",
            "european", "caribbean", "women's", "chicano", "jewish",
        ]
        studies_groups_social_science = [g + " studies" for g in x]

        return {
            "Engineering or Computer": {
                "keywords": [
                    "engineering", "computer", "software", "electronic", "information systems",
                    "information technology", "informatics", "robotics", "machine learning",
                    "artificial intelligence", "cybersecurity", "architecture", "urban planning",
                ],
                "variable_name": "engineering_or_computer",
            },
            "Natural Science": {
                "keywords": [
                    "biology", "biological", "chemistry", "physics", "environmental", "geology",
                    "earth", "astronomy", "astrophysics", "meteorology", "biotechnology",
                    "biochemistry", "biotech", "biochem", "neuroscience", "marine", "oceanography",
                    "ecology", "genetics",
                ],
                "variable_name": "natural_science",
            },
            "Math": {
                "keywords": ["math", "mathematics", "statistics", "statistical", "stats", "data science", "analytics"],
                "variable_name": "math",
            },
            "Education": {
                "keywords": ["education", "teacher", "teaching", "instructional", "curriculum", "pedagogy", "educational", "speech therapy"],
                "variable_name": "education",
            },
            "Clinical Work": {
                "keywords": ["social work", "pre-med", "pharmacy", "nursing", "health", "mental", "therapy", "clinical", "counseling"],
                "variable_name": "clinical_work",
            },
            "Law / Climinology": {
                "keywords": ["law", "legal", "criminology", "criminal", "justice", "landscape"],
                "variable_name": "law_climinology",
            },
            "Economics and Finance": {
                "keywords": ["economics", "econ", "finance", "financial", "banking", "investment", "econometrics"],
                "variable_name": "economics_and_finance",
            },
            "Business (not Economics / Finance)": {
                "keywords": [
                    "public relations", "business", "management", "accounting", "mba", "m.b.a.",
                    "marketing", "administration", "advertising", "human resources", "operations",
                    "supply chain", "organizational behavior",
                ],
                "variable_name": "business_not_economics_finance",
            },
            "Social Science (not Economics)": {
                "keywords": [
                    "social science", "history", "sociology", "anthropology", "international relations",
                    "political science", "government", "policy", "ethnic", "cultural", "religion",
                    "philosophy", "liberal art",
                ] + studies_groups_social_science,
                "variable_name": "social_science_not_economics",
            },
            "Arts": {
                "keywords": ["fine art", "design", "graphic", "music", "theater", "film", "cinema", "photography", "fashion", "visual", "dance", "performing"],
                "variable_name": "arts",
            },
            "Communications": {
                "keywords": ["communication", "communications", "media", "journalism", "broadcasting"],
                "variable_name": "communications",
            },
            "English": {
                "keywords": ["english", "literature", "writing"],
                "variable_name": "english",
            },
            "Psychology": {
                "keywords": ["psychology"],
                "variable_name": "psychology",
            },
        }

    def __post_init__(self) -> None:
        if not self.majors_categories:
            self.majors_categories = self.default_categories()

    def assign_major(self, subtitle: str | None) -> tuple[str, str]:
        if subtitle is None or pd.isna(subtitle):
            return "Other", "major_other"

        subtitle_lower = str(subtitle).lower()
        for major_name, details in self.majors_categories.items():
            for keyword in details["keywords"]:
                if keyword in subtitle_lower:
                    return major_name, "major_" + details["variable_name"]
        return "Other", "major_other"

    def apply(self, graduates_person_level: pd.DataFrame) -> pd.DataFrame:
        section("2.4.3 Categorize majors by keywords")
        out = graduates_person_level.copy()

        major_results = out["university_major_clean"].apply(self.assign_major)
        major_categories = [result[0] for result in major_results]
        major_variables = [result[1] for result in major_results]

        out["university_major_categorized"] = [
            "; ".join(sorted(set(cat.split("; ")))) if cat else "Other"
            for cat in major_categories
        ]

        for _, details in self.majors_categories.items():
            var_name = "major_" + details["variable_name"]
            out[var_name] = [1 if var_name in str(major_var) else 0 for major_var in major_variables]

        out["major_other"] = [1 if mv == "major_other" else 0 for mv in major_variables]

        print("Major categorization complete")
        print(out["university_major_categorized"].value_counts().head(20).to_string())
        return out


@dataclass
class StataPanelBuilder:
    cfg: Config
    repository: DataRepository

    @staticmethod
    def _make_time_window_flags(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
        out = df.copy()
        max_cols: dict[str, str] = {}

        for col in cols:
            for yrs in [3, 5, 10]:
                out[f"{col}_{yrs}_years_o"] = out[col] & (out["job_from"] <= (out["year_to"] + yrs))

                founded_year = out["company_page_founded_year"]
                using_page_year = founded_year.notna()
                page_cond = founded_year <= (out["year_to"] + yrs)
                fallback_cond = out["job_from"] <= (out["year_to"] + yrs)

                out[f"{col}_{yrs}_years_p"] = out[col] & ((using_page_year & page_cond) | (~using_page_year & fallback_cond))

                max_cols[f"{col}_{yrs}_years_o"] = "max"
                max_cols[f"{col}_{yrs}_years_p"] = "max"

        return out, max_cols

    def build_graduate_panel(self) -> pd.DataFrame:
        section("2.2-2.6 Build graduate-level panel")

        datasets = self.repository.load_core_datasets()
        education = datasets["education"]
        experience = datasets["experience"]

        print("Merging experience and education on member_id...")
        g = experience.merge(education, on="member_id", suffixes=("_experience", "_education"))
        print(f"Rows after merge: {len(g):,}")

        duplicate_cols = [
            "member_id", "title_experience", "company_url", "date_from_experience", "date_to_experience",
            "title_education", "date_from_education", "date_to_education", "unitid",
        ]
        pre = len(g)
        g = g.drop_duplicates(subset=duplicate_cols, keep="first")
        print(f"Dropped duplicates: {pre - len(g):,}")

        g["worked_as_engineer"] = g["title_experience"].str.contains("engineer", case=False, na=False)
        g["worked_in_sales"] = g["title_experience"].str.contains("sales", case=False, na=False)

        companies = pd.read_pickle(self.cfg.company_marketing_scores_path)
        g = g.merge(companies, on=["company_name"], how="left")
        print("Merged marketing scores")

        companies_probs = experience[["company_name", "fb_ad_prob"]].drop_duplicates(subset=["company_name"])
        g = g.merge(companies_probs, on="company_name", how="left")

        found_own_columns = [c for c in experience.columns if "found" in c.lower() or "own" in c.lower()]
        cols = ["worked_as_engineer", "worked_in_sales"] + found_own_columns
        cols = [c for c in cols if c in g.columns]
        print("Using columns for time-window flags:")
        print(", ".join(cols))

        g, max_cols = self._make_time_window_flags(g, cols)

        if "avg_prob_finance_broad" in g.columns:
            max_cols["avg_prob_finance_broad"] = "mean"
        if "avg_prob_wallstreet" in g.columns:
            max_cols["avg_prob_wallstreet"] = "mean"

        graduates_with_education_job_level = g

        group_keys = ["member_id", "year_to", "year_from", "title_education", "subtitle", "unitid"]
        graduates_person_level = graduates_with_education_job_level.groupby(group_keys, dropna=False).agg(max_cols).reset_index()

        graduates_person_level.rename(
            columns={
                "year_from": "year_start_college",
                "year_to": "year_end_college",
                "member_id": "linkedin_member_id",
                "title_education": "university_title",
                "subtitle": "university_major_raw",
            },
            inplace=True,
        )

        self._clean_majors(graduates_person_level)

        categorizer = MajorCategorizer()
        graduates_person_level = categorizer.apply(graduates_person_level)

        self._print_summary(graduates_person_level)
        return_data =  {
            "graduates_with_education_job_level": graduates_with_education_job_level,
            "graduates_person_level": graduates_person_level
            }

        return return_data



    @staticmethod
    def _clean_majors(graduates_person_level: pd.DataFrame) -> None:
        section("2.4 Clean major text")
        generic_patterns = [
            r"^\s*bachelor'?s?\s*degree", r"^\s*b\.?s\.?", r"^\s*b\.?a\.?", r"^\s*bachelor'?s?", r"^undergrad",
            r"^\s*bachelor of science(\s*\(b\.?s\.?\))?$", r"^bachelor of arts(\s*\(b\.?a\.?\))?$", r"bachelor",
        ]

        graduates_person_level["bachelors_degree"] = False
        for pat in generic_patterns:
            mask = graduates_person_level["university_major_raw"].astype(str).str.lower().str.match(pat, na=False)
            graduates_person_level.loc[mask, "bachelors_degree"] = True

        patterns_to_remove = [
            r"^bachelor of (applied\s+)?science(\s*\(b\.?s\.?\))?\s*,?\s*",
            r"^bachelor of arts(\s*\(b\.?a\.?\))?\s*,?\s*",
            r"^bachelor'?s? degree\w?,",
            r"^BS",
            r"^master of( science)?(\(MS\))?\s*,?\s*",
            r"master'?s( degree)?[\s,]*",
        ]

        combined_pattern = "|".join(patterns_to_remove)
        graduates_person_level["university_major_clean"] = graduates_person_level["university_major_raw"].astype(str).str.replace(
            combined_pattern, "", case=False, regex=True
        ).str.strip()

        graduates_person_level["university_major_clean"] = (
            graduates_person_level["university_major_clean"].str.replace(r"\b[Mm]inor\b[ \w]*", "", regex=True).str.strip()
        )

        # Remove entries that are just short parenthetical notes.
        mask_note = graduates_person_level["university_major_clean"].str.contains(r"^\(.{0,6}\)$", regex=True, na=False)
        before = len(graduates_person_level)
        graduates_person_level.drop(index=graduates_person_level[mask_note].index, inplace=True)
        print(f"Removed parenthetical-only major rows: {before - len(graduates_person_level):,}")

    @staticmethod
    def _print_summary(graduates_person_level: pd.DataFrame) -> None:
        section("2.5-2.6 Summary checks")
        print(f"Total graduates: {len(graduates_person_level):,}")

        major_dummy_cols = [c for c in graduates_person_level.columns if c.startswith("major_")]
        if major_dummy_cols:
            print("Major category summary:")
            total = len(graduates_person_level)
            for col in major_dummy_cols:
                count = graduates_person_level[col].sum()
                pct = (count / total) * 100 if total else 0
                print(f"{col:<40}: {count:>10,} ({pct:>5.1f}%)")


@dataclass
class CofounderBuilder:
    cfg: Config
    repository: DataRepository

    def build_from_graduate_panel(self, graduates_with_education_job_level: pd.DataFrame) -> pd.DataFrame:
        section("3 Create cofounder dataset")

        mask = (~graduates_with_education_job_level["company_url"].isna()) & (
            graduates_with_education_job_level["is_founder_or_owner_5_years"] == True
        )
        graduate_founders = graduates_with_education_job_level[mask].copy()
        print(f"Graduate founders rows: {len(graduate_founders):,}")

        members_per_company_url = graduate_founders.groupby(["company_url"], observed=True).agg({"member_id": "nunique"}).reset_index()
        members_per_company_url = members_per_company_url.rename(columns={"member_id": "number_of_founders"})

        cofounder_counts = members_per_company_url[members_per_company_url["number_of_founders"].between(2, 10)]
        cofounder_events = graduate_founders[graduate_founders["company_url"].isin(cofounder_counts["company_url"])].copy()
        print(f"Cofounder events rows: {len(cofounder_events):,}")

        cofounder_events.rename(
            columns={
                "year_from": "year_start_college",
                "year_to": "year_end_college",
                "member_id": "linkedin_member_id",
                "title_education": "university_title",
                "subtitle": "university_major_raw",
                "date_from_experience": "start_founder_date",
                "date_to_experience": "end_founder_date",
            },
            inplace=True,
        )

        cols = [
            "linkedin_member_id", "company_name", "company_url", "title_experience", "start_founder_date",
            "end_founder_date", "year_start_college", "year_end_college", "university_title", "unitid",
        ]
        cols = [c for c in cols if c in cofounder_events.columns]
        out = cofounder_events[cols].drop_duplicates()
        print(f"Final cofounder dataset rows: {len(out):,}")
        return out


@dataclass
class StataPanelPipeline:
    cfg: Config
    repository: DataRepository
    panel_builder: StataPanelBuilder
    cofounder_builder: CofounderBuilder

    def run_section_2(self) -> pd.DataFrame:
        data = self.panel_builder.build_graduate_panel()
        graduates_with_education_job_level = data["graduates_with_education_job_level"]
        graduates_person_level = data["graduates_person_level"]
        today_str = date.today().strftime("%Y%m%d")

        self.repository.save_graduate_job_level(graduates_with_education_job_level, today_str)

        self.repository.save_graduates_person_level_pickle(graduates_person_level, today_str)
        self.repository.save_graduates_person_level_stata(graduates_person_level, today_str)

        variable_labels = {
            "linkedin_member_id": "Unique LinkedIn member identifier",
            "unitid": "University identification code from IPEDS database",
            "university_title": "Official name of the university",
            "university_major_raw": "Raw major/degree description as reported",
            "university_major_clean": "Cleaned major description with degree prefixes removed",
            "university_major_categorized": "Major classified into standardized categories",
            "year_start_college": "Year the individual started college",
            "year_end_college": "Year the individual graduated from college",
        }

        existing = set(graduates_person_level.columns)
        final_labels = {k: v for k, v in variable_labels.items() if k in existing}
        self.repository.save_graduates_person_level_labeled_stata(graduates_person_level, today_str, final_labels)
        return graduates_person_level

    def run_section_3(self, graduates_person_level: pd.DataFrame | None = None) -> pd.DataFrame:
        today_str = date.today().strftime("%Y%m%d")

        if graduates_person_level is None:
            graduates_with_education_job_level = self.repository.load_graduate_job_level(today_str)
        else:
            # We need the event-level dataset for cofounders, so load staged event-level data.
            graduates_with_education_job_level = self.repository.load_graduate_job_level(today_str)

        cofounder_events = self.cofounder_builder.build_from_graduate_panel(graduates_with_education_job_level)
        self.repository.save_cofounder_events_stata(cofounder_events, today_str, clean=False)
        return cofounder_events

    def run_from_section(self, start_section: int) -> None:
        if start_section not in {2, 3}:
            raise ValueError("start_section must be 2 or 3")

        if start_section == 2:
            graduates_person_level = self.run_section_2()
            _ = self.run_section_3(graduates_person_level)
            return

        _ = self.run_section_3(None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stata analysis panel and cofounder datasets.")
    parser.add_argument(
        "--start-section",
        type=int,
        choices=[2, 3],
        default=2,
        help="Section to start from: 2 (panel build) or 3 (cofounder build)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    gc.collect()
    print("Garbage collection complete")

    cfg = Config()
    repo = DataRepository(cfg)
    panel_builder = StataPanelBuilder(cfg, repo)
    cofounder_builder = CofounderBuilder(cfg, repo)

    pipeline = StataPanelPipeline(
        cfg=cfg,
        repository=repo,
        panel_builder=panel_builder,
        cofounder_builder=cofounder_builder,
    )

    pipeline.run_from_section(args.start_section)

    section("Pipeline complete")
    print("Create_Stata_Analysis_Panel pipeline finished successfully")


if __name__ == "__main__":
    main()
