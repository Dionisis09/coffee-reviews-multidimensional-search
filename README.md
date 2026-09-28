# Coffee Reviews: Multidimensional Search

Python implementation of a search pipeline for coffee reviews. It combines
numeric filters, multidimensional indexes, and text similarity with shingling,
MinHash, and locality-sensitive hashing (LSH). The repository also includes
scripts for timing measurements and recall evaluation.

## Setup

Requires Python 3.10 or newer. From the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The original review CSV files are not redistributed here. Obtain the
[Coffee Reviews Dataset](https://www.kaggle.com/datasets/schmoyote/coffee-reviews-dataset)
and place `coffee_analysis.csv` in the repository directory. The pipeline can
also read another CSV with review text and numeric fields; use `--col_text` to
select its text column. The provided `coffee_analysis.csv` uses `desc_1` for
review text.

## Run

Search with the default index and return up to 10 similar reviews:

```powershell
python pipeline.py --csv coffee_analysis.csv --col_text desc_1 --topn 10
```

Apply country, price, rating, and year filters:

```powershell
python pipeline.py --csv coffee_analysis.csv --col_text desc_1 --country "Hong Kong" --price_min 4 --price_max 10 --rating_min 94 --year_min 2017 --year_max 2021 --topn 10
```

Use an index for k-nearest-neighbor refinement (`kd`, `quad`, `range`, or
`rtree`):

```powershell
python pipeline.py --csv coffee_analysis.csv --col_text desc_1 --index kd --knn 20 --topn 10
```

The pipeline writes `results_lsh.csv`. It also accepts `--zip` for an archive
containing a CSV and `--help` for the full option list.

## Evaluate

`bench.py` runs the index and LSH parameter grid defined at the top of that
script. Its default input is `coffee_analysis.csv`; change `CSV_PATH` and
`TEXT_COL` there to use another dataset.

```powershell
python bench.py
python plot_bench.py "bench_results*.csv"
python recall_baseline.py --csv coffee_analysis.csv --col_text desc_1 --sample 2000 -k 10 --bands 32 --rows 8 --thres 0.35
```

Benchmarks write `bench_results_<timestamp>.csv`. The plotting script reads
those files and saves PNG charts. The recall script compares LSH candidates
with a TF-IDF cosine-similarity baseline.

## Project files

- `pipeline.py`: input loading, filtering, MinHash/LSH, index refinement, ranking.
- `indexes/`: KD-tree, quadtree, range-tree, and R-tree implementations.
- `bench.py`: timing measurements across parameter combinations.
- `plot_bench.py`: charts from benchmark CSV files.
- `recall_baseline.py`: recall evaluation against a brute-force baseline.
