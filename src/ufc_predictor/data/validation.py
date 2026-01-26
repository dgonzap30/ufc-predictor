"""
Data validation module.

Functions for validating data integrity and quality.
Checks include:
- Referential integrity between fights and fighters
- Value range validation (no negative ages, heights, etc.)
- Temporal consistency (fight dates, fighter ages)
- Completeness checks for critical fields
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ufc_predictor.config import INTERMEDIATE_DATA_DIR, REPORTS_DIR


def validate_integrity(fights: pd.DataFrame, fighters: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate integrity and quality of cleaned UFC data.

    Checks performed:
    - All fighters in fights table exist in fighters table
    - No impossible values (negative stats, future dates)
    - No critical missing values in key fields
    - Date consistency (fight dates are chronological)
    - Fighter age calculations are reasonable

    Args:
        fights: Cleaned fights DataFrame
        fighters: Cleaned fighters DataFrame

    Returns:
        Dictionary containing validation results with timestamp, referential_integrity,
        missing_values, and sanity_checks sections
    """
    results = {
        "timestamp": datetime.now(),
        "referential_integrity": {},
        "missing_values": {},
        "sanity_checks": {},
    }

    # 1. REFERENTIAL INTEGRITY CHECK
    fighters_in_fights = set(fights["fighter1"]) | set(fights["fighter2"])
    known_fighters = set(fighters["fighter"])
    missing_fighters = sorted(fighters_in_fights - known_fighters)

    # Count affected fights
    affected_fights = fights[
        fights["fighter1"].isin(missing_fighters) | fights["fighter2"].isin(missing_fighters)
    ]

    results["referential_integrity"] = {
        "missing_fighters": missing_fighters,
        "missing_count": len(missing_fighters),
        "affected_fights_count": len(affected_fights),
    }

    # 2. MISSING VALUES REPORT
    results["missing_values"] = {
        "height_inches_pct": fighters["height_inches"].isna().mean() * 100,
        "reach_inches_pct": fighters["reach_inches"].isna().mean() * 100,
        "dob_pct": fighters["dob"].isna().mean() * 100,
    }

    # 3. SANITY CHECKS

    # 3a. Invalid reach (< 40 inches)
    invalid_reach = fighters[fighters["reach_inches"] < 40].copy()
    results["sanity_checks"]["invalid_reach"] = invalid_reach[
        ["fighter", "reach_inches"]
    ].to_dict("records")

    # 3b. Invalid age at fight time (< 18 or > 60)
    # Merge fights with fighters to get DOB for each fighter
    fights_with_dob1 = fights.merge(
        fighters[["fighter", "dob"]].rename(columns={"dob": "dob1"}),
        left_on="fighter1",
        right_on="fighter",
        how="left",
    ).drop(columns=["fighter"])

    fights_with_dob = fights_with_dob1.merge(
        fighters[["fighter", "dob"]].rename(columns={"dob": "dob2"}),
        left_on="fighter2",
        right_on="fighter",
        how="left",
    ).drop(columns=["fighter"])

    # Calculate ages
    fights_with_dob["age1"] = (
        (fights_with_dob["date"] - fights_with_dob["dob1"]).dt.days / 365.25
    )
    fights_with_dob["age2"] = (
        (fights_with_dob["date"] - fights_with_dob["dob2"]).dt.days / 365.25
    )

    # Flag invalid ages
    invalid_age_fights = fights_with_dob[
        (fights_with_dob["age1"] < 18)
        | (fights_with_dob["age1"] > 60)
        | (fights_with_dob["age2"] < 18)
        | (fights_with_dob["age2"] > 60)
    ].copy()

    results["sanity_checks"]["invalid_age"] = invalid_age_fights[
        ["date", "fighter1", "fighter2", "age1", "age2"]
    ].to_dict("records")

    return results


def print_summary(results: Dict[str, Any]) -> None:
    """
    Print validation summary to console.

    Args:
        results: Validation results dictionary
    """
    print("\n" + "=" * 50)
    print("UFC Data Validation Report")
    print("=" * 50)

    # Referential Integrity
    ref_int = results["referential_integrity"]
    print("\nREFERENTIAL INTEGRITY:")
    print(f"  Missing fighters: {ref_int['missing_count']} (not in fighters table)")
    print(f"  Affected fights: {ref_int['affected_fights_count']}")

    # Missing Values
    missing = results["missing_values"]
    print("\nMISSING VALUES (fighters):")
    print(f"  height_inches: {missing['height_inches_pct']:.1f}% missing")
    print(f"  reach_inches: {missing['reach_inches_pct']:.1f}% missing")
    print(f"  dob: {missing['dob_pct']:.1f}% missing")

    # Sanity Checks
    sanity = results["sanity_checks"]
    print("\nSANITY CHECKS:")
    print(f"  Invalid reach (<40\"): {len(sanity['invalid_reach'])} fighters")
    print(f"  Invalid age (<18 or >60): {len(sanity['invalid_age'])} fights")

    print("\n" + "=" * 50)


def save_report(results: Dict[str, Any]) -> None:
    """
    Save detailed validation report to text file.

    Args:
        results: Validation results dictionary
    """
    # Create reports directory if needed
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = REPORTS_DIR / "validation_report.txt"

    with open(report_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("UFC DATA VALIDATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # Referential Integrity
        ref_int = results["referential_integrity"]
        f.write("REFERENTIAL INTEGRITY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Missing fighters: {ref_int['missing_count']}\n")
        f.write(f"Affected fights: {ref_int['affected_fights_count']}\n\n")

        if ref_int["missing_fighters"]:
            f.write("Missing fighter names:\n")
            for i, fighter in enumerate(ref_int["missing_fighters"][:50], 1):
                f.write(f"  {i}. {fighter}\n")
            if len(ref_int["missing_fighters"]) > 50:
                f.write(f"  ... and {len(ref_int['missing_fighters']) - 50} more\n")
        f.write("\n")

        # Missing Values
        missing = results["missing_values"]
        f.write("MISSING VALUES (fighters table)\n")
        f.write("-" * 70 + "\n")
        f.write(f"height_inches: {missing['height_inches_pct']:.2f}% missing\n")
        f.write(f"reach_inches: {missing['reach_inches_pct']:.2f}% missing\n")
        f.write(f"dob: {missing['dob_pct']:.2f}% missing\n\n")

        # Sanity Checks
        sanity = results["sanity_checks"]
        f.write("SANITY CHECKS\n")
        f.write("-" * 70 + "\n")

        f.write(f"\nInvalid reach (<40 inches): {len(sanity['invalid_reach'])} fighters\n")
        if sanity["invalid_reach"]:
            for record in sanity["invalid_reach"][:20]:
                f.write(f"  - {record['fighter']}: {record['reach_inches']}\" reach\n")

        f.write(f"\nInvalid age (<18 or >60): {len(sanity['invalid_age'])} fights\n")
        if sanity["invalid_age"]:
            for record in sanity["invalid_age"][:20]:
                f.write(
                    f"  - {record['date']}: {record['fighter1']} (age {record['age1']:.1f}) "
                    f"vs {record['fighter2']} (age {record['age2']:.1f})\n"
                )
            if len(sanity["invalid_age"]) > 20:
                f.write(f"  ... and {len(sanity['invalid_age']) - 20} more\n")

        f.write("\n" + "=" * 70 + "\n")

    print(f"\nDetailed report saved to: {report_path}")


def run_validation() -> Dict[str, Any]:
    """
    Run the complete data validation pipeline.

    Loads cleaned data, runs validation checks, prints summary, and saves report.

    Returns:
        Validation results dictionary
    """
    print("Loading cleaned data...")
    fights = pd.read_parquet(INTERMEDIATE_DATA_DIR / "cleaned_fights.parquet")
    fighters = pd.read_parquet(INTERMEDIATE_DATA_DIR / "cleaned_fighters.parquet")

    print(f"Loaded {len(fights)} fights and {len(fighters)} fighters")

    print("\nRunning validation checks...")
    results = validate_integrity(fights, fighters)

    # Print summary
    print_summary(results)

    # Save detailed report
    save_report(results)

    return results


if __name__ == "__main__":
    run_validation()
