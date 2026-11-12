"""Build a 5-second last-quote mid-price grid from HistData bid/ask tick zips (paper 44 needs it).
Only grid points preceded by a quote update are kept (as in the paper). Timestamps naive ET."""
import glob
import os
import zipfile
import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "input_Data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "_mid5s.parquet")


def read_zip(path):
    z = zipfile.ZipFile(path)
    n = [x for x in z.namelist() if x.endswith(".csv")][0]
    df = pd.read_csv(z.open(n), sep=";", header=None, names=["ts", "px", "v"], usecols=[0, 1])
    df["ts"] = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
    return df.groupby("ts")["px"].last()


parts = []
for bidf in sorted(glob.glob(os.path.join(SRC, "HISTDATA_COM_NT_XAUUSD_T_BID*.zip"))):
    askf = bidf.replace("_BID", "_ASK")
    b, a = read_zip(bidf), read_zip(askf)
    q = pd.concat([b.rename("bid"), a.rename("ask")], axis=1).sort_index().ffill().dropna()
    q = q[(q.ask > q.bid) & ((q.ask - q.bid) < 5)]
    mid = (q.bid + q.ask) / 2
    g = mid.groupby(mid.index.floor("5s"))
    m5s = pd.DataFrame({"mid": g.last(), "spread": (q.ask - q.bid).groupby(q.index.floor("5s")).last()})
    parts.append(m5s)
    print(os.path.basename(bidf), len(m5s), flush=True)
out = pd.concat(parts)
out = out[~out.index.duplicated(keep="last")].sort_index()
out.index.name = "ts"
out.astype("float32").to_parquet(OUT)
print("rows", len(out))
