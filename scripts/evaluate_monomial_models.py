"""Evaluate SystemDiscovery with poly_degree 0, 1, 2 over all BioModel timecourses.

Output CSV written to data/evaluate_monomial_models.csv with columns:
    model_name, num_species, num_reactions,
    deg0_min, deg0_median, deg0_max,
    deg1_min, deg1_median, deg1_max,
    deg2_min, deg2_median, deg2_max
"""

import os
import sys

import pandas as pd  # type: ignore

import src.constants as cn  # type: ignore
from src.system_discovery import SystemDiscovery  # type: ignore
from src.timecourse_iterator import TimecourseIterator  # type: ignore

DEGREES = [0, 1, 2]
OUTPUT_PATH = os.path.join(cn.DATA_DIR, "evaluate_monomial_models.csv")



def _score_cols(deg: int) -> tuple[str, str, str]:
    return f"deg{deg}_min", f"deg{deg}_median", f"deg{deg}_max"


def _evaluate_model(model_name: str, timecourse, *,
        threshold: float=0.01) -> dict:
    row: dict = {
        cn.COL_MODEL_NAME: model_name,
        cn.COL_NUM_SPECIES: timecourse.model.num_species,
        cn.COL_NUM_REACTION: timecourse.model.num_reaction,
    }
    for deg in DEGREES:
        for col in _score_cols(deg):
            row[col] = float("nan")

    for deg in DEGREES:
        try:
            sd = SystemDiscovery(
                timecourse.timecourse_df,
                poly_degree=deg,
                include_bias=True,
                threshold=threshold,
            )
            sd.fit()
            info = sd.score()
            min_col, median_col, max_col = _score_cols(deg)
            row[min_col] = info.min
            row[median_col] = info.median
            row[max_col] = info.max
        except Exception as exc:
            print(f"  [deg={deg}] {model_name}: {exc}", file=sys.stderr)

    return row


def main(threshold: float = 0.01, is_initialize: bool = True) -> None:
    rows = []
    if is_initialize:
        if os.path.isfile(OUTPUT_PATH):
            print(f"Loading existing results from {OUTPUT_PATH}...")
            df = pd.read_csv(OUTPUT_PATH)
            rows = df.to_dict(orient="records")
        else:
            print(f"Initializing {OUTPUT_PATH} with header...")
            df = pd.DataFrame(rows)
            df.to_csv(OUTPUT_PATH, index=False) 
    # Process models not already present
    df = pd.DataFrame(rows)
    for item in TimecourseIterator(num_model=50):
        if len(df) > 0 and item.model_name in df[cn.COL_MODEL_NAME].values:
            print(f"Skipping {item.model_name} (already processed)", flush=True)
            continue
        print(f"Processing {item.model_name}...", flush=True)
        row = _evaluate_model(item.model_name, item.timecourse, threshold=threshold)
        rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
