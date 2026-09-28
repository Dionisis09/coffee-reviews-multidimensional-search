# pipeline.py
from indexes.base import MultiIndex
import argparse, io, zipfile, os, re, math, random, glob, time, pickle, hashlib
from collections import defaultdict
from typing import Optional, List, Dict, Any, Tuple

import pandas as pd
import numpy as np
from unidecode import unidecode



PREFERRED_NAMES = ["coffee_analysis.csv","simplified_coffee.csv","coffeereviewcom.csv","coffee_reviews.csv","reviews.csv"]

def find_local_csv(base_dir: str) -> Optional[str]:
    candidates: List[str] = []
    for folder in [base_dir, os.path.join(base_dir, "archive")]:
        candidates += glob.glob(os.path.join(folder, "*.csv"))
    if not candidates: return None
    name_map = {os.path.basename(p).lower(): p for p in candidates}
    for pref in PREFERRED_NAMES:
        if pref.lower() in name_map: return name_map[pref.lower()]
    candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return candidates[0]

def load_csv_from_zip(zip_path: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, 'r') as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if not csv_names: raise FileNotFoundError("No CSV in zip.")
        csv_names.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
        with z.open(csv_names[0]) as f:
            raw = f.read()
            for enc in ["utf-8","utf-8-sig","latin-1"]:
                for sep in [",",";","\t","|"]:
                    try:
                        df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep)
                        if len(df.columns)>1: return df
                    except Exception: pass
    raise ValueError("Failed reading CSV from zip.")

def load_csv(csv_path: Optional[str], zip_path: Optional[str], **kwargs) -> pd.DataFrame:
    if csv_path and os.path.isfile(csv_path):
        return pd.read_csv(csv_path, sep=kwargs.get("sep", ","), encoding=kwargs.get("encoding","utf-8"))
    if zip_path and os.path.isfile(zip_path):
        return load_csv_from_zip(zip_path)
    found = find_local_csv(os.getcwd())
    if not found: raise FileNotFoundError("No CSV found; pass --csv or add a csv in folder or /archive.")
    return pd.read_csv(found, sep=kwargs.get("sep", ","), encoding=kwargs.get("encoding","utf-8"))

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [unidecode(str(c)).strip().lower().replace(" ","_") for c in df.columns]
    return df

def guess_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    cols = df.columns.tolist()
    rating  = next((c for c in cols if re.search(r"rating|score|points", c)), None)
    price   = next((c for c in cols if re.search(r"price|cost|\$|usd|100g_usd", c)), None)
    year    = next((c for c in cols if re.search(r"year|date|review_date", c)), None)
    country = next((c for c in cols if re.search(r"country|origin|loc_country|location", c)), None)
    text    = next((c for c in cols if re.search(r"review|text|description|notes|body|comment|desc_1", c)), None)
    return rating, price, year, country, text

def coerce_year(series: pd.Series) -> pd.Series:
    def parse(v):
        s = str(v); m = re.search(r"(20\d{2}|19\d{2})", s)
        return int(m.group(1)) if m else np.nan
    return series.map(parse)

# ========= MinHash + LSH =========
class MinHash:
    def __init__(self, n_hashes=64, seed=42):
        self.n = n_hashes; random.seed(seed)
        self.a = [random.randint(1, 2**31-1) for _ in range(self.n)]
        self.b = [random.randint(0, 2**31-1) for _ in range(self.n)]
        self.p = 2**31-1
    def _hashes(self, token_id: int):
        return [(self.a[i]*token_id + self.b[i]) % self.p for i in range(self.n)]
    def signature(self, tokens: set) -> tuple:
        sig = [math.inf]*self.n
        for t in tokens:
            tid = hash(t) & 0xffffffff
            for i,h in enumerate(self._hashes(tid)):
                if h < sig[i]: sig[i] = h
        return tuple(int(x) for x in sig)

def shingle(text: str, k=3) -> set:
    text = unidecode((text or "").lower())
    text = re.sub(r"[^a-z0-9 ]+"," ", text)
    text = re.sub(r"\s+"," ", text).strip()
    if not text: return set()
    s=set(); t=f" {text} "
    for i in range(max(0, len(t)-k+1)): s.add(t[i:i+k])
    return s

class LSH:
    def __init__(self, bands=16, rows=4):
        self.bands=bands; self.rows=rows
        self.tables=[defaultdict(list) for _ in range(bands)]
    def _band_keys(self, sig: tuple):
        assert len(sig) == self.bands*self.rows
        for b in range(self.bands):
            start=b*self.rows
            yield b, hash(sig[start:start+self.rows])
    def add(self, idx:int, sig:tuple):
        for b,key in self._band_keys(sig): self.tables[b][key].append(int(idx))
    def candidates(self, sig:tuple):
        cand=set()
        for b,key in self._band_keys(sig): cand.update(self.tables[b].get(key, []))
        return cand

def jaccard(a:set,b:set)->float:
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    return len(a & b)/len(a | b)

# ========= numeric prep =========
def fit_numeric_index(df: pd.DataFrame, rating, price, year, country) -> pd.DataFrame:
    work = pd.DataFrame(index=df.index)
    if rating and not pd.api.types.is_numeric_dtype(df[rating]): df[rating]=pd.to_numeric(df[rating], errors='coerce')
    if price and not pd.api.types.is_numeric_dtype(df[price]):   df[price]=pd.to_numeric(df[price], errors='coerce')
    if year: df[year]=coerce_year(df[year])
    if rating: work["rating"]=df[rating]
    if price:  work["price"]=df[price]
    if year:   work["year"]=df[year]
    if country:
        freq = df[country].fillna("UNK").astype(str).value_counts(normalize=True)
        work["country_freq"]=df[country].fillna("UNK").astype(str).map(freq).fillna(0.0)
    sel=[c for c in ["rating","price","year","country_freq"] if c in work.columns]
    work=work[sel].copy()
    for c in sel: work[c]=pd.to_numeric(work[c], errors="coerce")
    work=work.fillna(work.median(numeric_only=True))
    return work

def range_mask(work: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    m=pd.Series(True, index=work.index)
    if "rating" in work.columns and args.rating_min is not None: m &= work["rating"]>=args.rating_min
    if "rating" in work.columns and args.rating_max is not None: m &= work["rating"]<=args.rating_max
    if "price" in work.columns and args.price_min is not None:   m &= work["price"]>=args.price_min
    if "price" in work.columns and args.price_max is not None:   m &= work["price"]<=args.price_max
    if "year" in work.columns and args.year_min is not None:     m &= work["year"]>=args.year_min
    if "year" in work.columns and args.year_max is not None:     m &= work["year"]<=args.year_max
    return m

# ====== Index factory ======
def make_index(name: Optional[str]):
    nm = (name or "kd").lower()
    if nm == "kd":
        from indexes.kd_custom import KDTreeIndex
        return KDTreeIndex()
    if nm == "quad":
        from indexes.quad import QuadIndex
        return QuadIndex()
    if nm == "range":
        from indexes.range_tree import RangeTreeIndex
        return RangeTreeIndex()
    if nm == "rtree":
        from indexes.rtree import RTreeIndex
        return RTreeIndex()
    # fallback: sklearn KD if installed
    try:
        from indexes.kd_sklearn import KDIndex
        return KDIndex()
    except Exception:
        from indexes.kd_custom import KDTreeIndex
        return KDTreeIndex()

# helpers
def _sig_cache_key(texts: pd.Series, nhashes: int, ksh: int, col_text: str, sample: Optional[int]) -> str:
    head="||".join(texts.head(100).astype(str).tolist())
    h=hashlib.sha256(f"{len(texts)}::{nhashes}::{ksh}::{col_text}::{sample}::{head}".encode()).hexdigest()
    return f".cache_sigs_{h}.pkl"

#  pipeline
def run_pipeline(csv_path=None, zip_path=None, quiet: bool=False, **kwargs) -> Dict[str, Any]:
    t0=time.perf_counter()
    df=load_csv(csv_path, zip_path, **kwargs); df=normalize_cols(df)
    if kwargs.get("sample"): df=df.iloc[:int(kwargs["sample"])].reset_index(drop=True)

    rating,price,year,country,text=guess_columns(df)
    rating  = kwargs.get("col_rating")  or rating
    price   = kwargs.get("col_price")   or price
    year    = kwargs.get("col_year")    or year
    country = kwargs.get("col_country") or country
    text    = kwargs.get("col_text")    or text
    if text is None and "title" in df.columns: text="title"
    if text is None: raise ValueError("No textual column found.")

    if kwargs.get("country"):
        cc=str(kwargs["country"]).lower(); ccol=country or "country"
        if ccol in df.columns: df=df[df[ccol].astype(str).str.lower().str.contains(cc, na=False)]

    df=df.reset_index(drop=True); t1=time.perf_counter()

    work=fit_numeric_index(df, rating, price, year, country); t2=time.perf_counter()
    args_obj=argparse.Namespace(**kwargs); mask=range_mask(work, args_obj)
    cand_idx=list(map(int, work[mask].index.tolist())); t3=time.perf_counter()

    # Optional: kNN refinement using chosen index
    if len(cand_idx) > 0 and int(kwargs.get("knn", 0)) > 0:
        idx = make_index(kwargs.get("index"))
        # αν το index δεν έχει δική του knn, παράκαμψε
        if not hasattr(idx, "knn") or idx.__class__.knn is MultiIndex.knn:
            pass
        else:
            idx.build(work.values)
            k = min(int(kwargs["knn"]), len(work))
            query_vec = work.loc[cand_idx].values[0]
            nn_ind, _ = idx.knn(query_vec, k)
            nn_idx = list(map(int, work.iloc[nn_ind].index.tolist()))
            cand_idx = list(set(cand_idx) & set(nn_idx)) or nn_idx


    bands=kwargs.get("bands",16); rows=kwargs.get("rows",4)
    nh=int(kwargs.get("nhashes", bands*rows))
    if nh != bands*rows: nh = bands*rows
    ksh=kwargs.get("kshingle",3)
    mh=MinHash(n_hashes=nh); lsh=LSH(bands=bands, rows=rows)

    texts=df[text].fillna("").astype(str)
    cache_path=_sig_cache_key(texts, nh, ksh, text, kwargs.get("sample"))
    try:
        with open(cache_path,"rb") as f: sigs=pickle.load(f)
    except Exception:
        shingles=[shingle(t, k=ksh) for t in texts]
        sigs=[mh.signature(s) for s in shingles]
        with open(cache_path,"wb") as f: pickle.dump(sigs,f)

    if not cand_idx:
        if not quiet:
            print("No rows passed the filters.")
            print("Cols:", {"rating":rating,"price":price,"year":year,"country":country,"text":text})
        return {"n_cand":0,"times":{"load_prep":t1-t0,"index_prep":t2-t1,"range_filter":t3-t2,"lsh_add":0.0},
                "params":{"bands":bands,"rows":rows,"nhashes":nh,"kshingle":ksh,"knn":kwargs.get("knn",0),"sample":kwargs.get("sample"),"index":kwargs.get("index","kd")},
                "cols":{"rating":rating,"price":price,"year":year,"country":country,"text":text},"topn":0}

    for i in cand_idx: lsh.add(int(i), sigs[int(i)])
    t4=time.perf_counter()
    if not quiet: print(f"[Info] candidates after filters: {len(cand_idx)}")

    qid=kwargs.get("query_id")
    if qid is None:
        if "rating" in work.columns and len(cand_idx)>0:
            qid=int(work.loc[cand_idx]["rating"].astype(float).idxmax())
        else:
            qid=int(cand_idx[0])
    else: qid=int(qid)

    qsig=sigs[qid]; cands=lsh.candidates(qsig)

    def shingles_for(i:int): return shingle(texts.iloc[i], k=ksh)
    pairs=[]
    for j in cands:
        j=int(j)
        if j==qid: continue
        sim=jaccard(shingles_for(qid), shingles_for(j))
        pairs.append((j, float(sim)))
    pairs.sort(key=lambda x:x[1], reverse=True)
    topn=min(int(kwargs.get("topn",10)), len(pairs))
    results=[(qid,1.0)]+pairs[:topn]

    show_cols: List[str] = [c for c in [rating,price,year,country,text] if c]
    out_rows: List[Dict[str, Any]] = []
    for idx,sim in results:
        ix=int(idx)
        base = df.loc[ix].reindex(show_cols)
        row = {**base.to_dict(), "_id": ix, "_sim": round(float(sim), 4)}
        out_rows.append(row)
    out_df=pd.DataFrame(out_rows)

    if not quiet:
        print(out_df)
        print("\nCols:", {"rating":rating,"price":price,"year":year,"country":country,"text":text})
        print(f"\n[Times] load+prep={t1-t0:.3f}s | index_prep={t2-t1:.3f}s | range_filter={t3-t2:.3f}s | lsh_add={t4-t3:.3f}s")
        out_df.to_csv("results_lsh.csv", index=False, encoding="utf-8")
        print("Saved: results_lsh.csv")

    return {
        "n_cand": len(cand_idx),
        "times": {"load_prep":t1-t0, "index_prep":t2-t1, "range_filter":t3-t2, "lsh_add":t4-t3},
        "params": {"bands":bands, "rows":rows, "nhashes":nh, "kshingle":ksh, "knn":kwargs.get("knn",0), "sample":kwargs.get("sample"), "index":kwargs.get("index","kd")},
        "cols": {"rating":rating,"price":price,"year":year,"country":country,"text":text},
        "topn": len(results)-1, "qid": qid,
    }

if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv", type=str); ap.add_argument("--zip", type=str)
    ap.add_argument("--price_min", type=float); ap.add_argument("--price_max", type=float)
    ap.add_argument("--rating_min", type=float); ap.add_argument("--rating_max", type=float)
    ap.add_argument("--year_min", type=int); ap.add_argument("--year_max", type=int)
    ap.add_argument("--country", type=str); ap.add_argument("--knn", type=int, default=0)
    ap.add_argument("--query_id", type=int); ap.add_argument("--topn", type=int, default=10)
    ap.add_argument("--bands", type=int, default=16); ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--nhashes", type=int, default=64); ap.add_argument("--kshingle", type=int, default=3)
    ap.add_argument("--index", choices=["kd","quad","range","rtree"], default="kd")
    ap.add_argument("--col_rating"); ap.add_argument("--col_price"); ap.add_argument("--col_year"); ap.add_argument("--col_country"); ap.add_argument("--col_text")
    ap.add_argument("--sep", default=","); ap.add_argument("--encoding", default="utf-8")
    ap.add_argument("--sample", type=int); ap.add_argument("--quiet", action="store_true")
    args=ap.parse_args()
    extra={k:v for k,v in vars(args).items() if k not in ["csv","zip"]}
    run_pipeline(csv_path=args.csv, zip_path=args.zip, **extra)
