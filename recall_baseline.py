# recall_baseline.py
import argparse, pandas as pd, numpy as np
from pipeline import run_pipeline
from datasketch import MinHash, MinHashLSH

def brute_force(texts, q, k):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    vec = TfidfVectorizer().fit_transform(texts)
    sims = cosine_similarity(vec.getrow(q), vec).ravel()
    return np.argsort(sims)[-k:][::-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--col_text", required=True)
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("-k", "--k", type=int, default=10)
    ap.add_argument("--bands", type=int, default=16)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--thres", type=float, default=0.4)
    args = ap.parse_args()

    df = pd.read_csv(args.csv).dropna(subset=[args.col_text])
    N = len(df); n = min(args.sample, N)
    if n < args.sample:
        print(f"[info] sample capped to {n} (available {N})")
    df = df.sample(n=n, random_state=0)
    texts = df[args.col_text].astype(str).tolist()

    # build LSH
    mh = [MinHash(num_perm=args.bands * args.rows) for _ in texts]
    for t, m in zip(texts, mh):
        for sh in [t[i:i+3] for i in range(len(t) - 2)]:
            m.update(sh.encode("utf8"))
    lsh = MinHashLSH(threshold=args.thres, num_perm=args.bands * args.rows)
    for i, m in enumerate(mh):
        lsh.insert(i, m)

    # test recall
    recalls = []
    for q in range(10):
        brute = set(brute_force(texts, q, args.k))
        cand = lsh.query(mh[q])
        inter = len(brute.intersection(cand))
        recalls.append(inter / args.k)
    print(f"Mean recall@{args.k}: {np.mean(recalls):.3f}")



if __name__ == "__main__":
    main()
