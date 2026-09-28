# bench.py
import itertools
import pandas as pd
from datetime import datetime
from pipeline import run_pipeline

CSV_PATH = "coffee_analysis.csv"
TEXT_COL = "desc_1"

GRID = {
    "index":   ["kd", "quad", "range", "rtree"],
    "bands":    [12, 16, 20],
    "rows":     [4],
    "kshingle": [3],
    "knn":      [0, 20],
    "sample":   [None],   # e.g., [3000, 8000, None]
}

FILTERS = dict(
    price_min=None, price_max=None,
    rating_min=None, rating_max=None,
    year_min=None,  year_max=None,
    country=None,
)

def grid_dicts(grid):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))

def main():
    rows = []
    for params in grid_dicts(GRID):
        kwargs = {
            "csv_path": CSV_PATH,
            "col_text": TEXT_COL,
            "quiet": True,
            **FILTERS,
            **{k:v for k,v in params.items() if v is not None}
        }
        metrics = run_pipeline(**kwargs)
        rows.append({
            "index": params["index"],
            "bands": params["bands"],
            "rows": params["rows"],
            "nhashes": metrics["params"]["nhashes"],
            "kshingle": params["kshingle"],
            "knn": params["knn"],
            "sample": params["sample"],
            "n_cand": metrics["n_cand"],
            "topn": metrics["topn"],
            "time_load_prep": metrics["times"]["load_prep"],
            "time_index_prep": metrics["times"]["index_prep"],
            "time_range_filter": metrics["times"]["range_filter"],
            "time_lsh_add": metrics["times"]["lsh_add"],
        })
    df = pd.DataFrame(rows).sort_values(["index","sample","bands","rows","nhashes","knn"])
    out_name = f"bench_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out_name, index=False, encoding="utf-8")
    print(df)
    print(f"\nSaved: {out_name}")

if __name__ == "__main__":
    main()
