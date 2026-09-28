# plot_bench.py
import pandas as pd
import matplotlib.pyplot as plt
import sys

def main(path="bench_results*.csv"):
    import glob
    files = glob.glob(path)
    if not files:
        print("No bench_results*.csv found.")
        return
    for f in files:
        df = pd.read_csv(f)
        print(f"Loaded {f} with {len(df)} rows")
        for col in ["time_load_prep","time_index_prep","time_range_filter","time_lsh_add"]:
            plt.figure()
            df.groupby("index")[col].mean().plot.bar()
            plt.title(f"{col} ({f})")
            plt.ylabel("seconds")
            plt.tight_layout()
            plt.savefig(f"{f}_{col}.png", dpi=160)

            plt.show()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv)>1 else "bench_results*.csv"
    main(path)
