import sys, json, urllib.request
from pathlib import Path
import pandas as pd

season=int(sys.argv[1]) if len(sys.argv)>1 else 2025
url=f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
out=Path("data/teams.json"); out.parent.mkdir(exist_ok=True)

df=pd.read_parquet(url)
df=df[(df["season_type"]=="REG") & df["posteam"].notna()]
df=df[df["play_type"].isin(["pass","run","qb_spike"])].copy()

# Exclude plays without a meaningful down from down/tendency denominators.
valid=df[df["down"].notna()].copy()
valid["is_pass"]=valid["play_type"].isin(["pass","qb_spike"])
valid["is_run"]=valid["play_type"].eq("run")
valid["rz"]=valid["yardline_100"].le(20)
valid["trailing"]=valid["score_differential"]<0
valid["leading"]=valid["score_differential"]>0
valid["two_minute"]=(valid["half_seconds_remaining"]<=120)

def rate(g):
    n=len(g)
    return float(g["is_pass"].sum()/n) if n else 0.0
def split(g):
    return {"pass":int(g["is_pass"].sum()),"runs":int(g["is_run"].sum()),
            "pass_rate":rate(g)}

teams=[]
for team,g in valid.groupby("posteam"):
    rz=g[g["rz"]]
    tr=g[g["trailing"]]
    lead=g[g["leading"]]
    tm=g[g["two_minute"]]
    downs=[]
    for down in sorted(g["down"].dropna().astype(int).unique()):
        x=g[g["down"].eq(down)]
        s=split(x); s["down"]=int(down); downs.append(s)
    weekly=[]
    for week,x in g.groupby("week"):
        s=split(x); s["week"]=int(week); weekly.append(s)
    weekly.sort(key=lambda x:x["week"])
    teams.append({
        "team":team,
        "pass_rate":rate(g),
        "rz_pass_rate":rate(rz),
        "trailing_pass_rate":rate(tr),
        "leading_pass_rate":rate(lead),
        "two_minute_pass_rate":rate(tm),
        "down_splits":downs,
        "weekly":weekly
    })
teams.sort(key=lambda x:x["team"])
out.write_text(json.dumps({"season":season,"teams":teams},indent=2))
print(f"Wrote {out} for {season}")
