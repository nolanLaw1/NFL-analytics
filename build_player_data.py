import sys
import json
from pathlib import Path
import pandas as pd

season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
print(f"Downloading nflverse play-by-play for {season}...")
df = pd.read_parquet(url)
df = df[df["season_type"].eq("REG")].copy()
SKILL = {"WR", "RB", "TE"}
players = {}

def ensure(pid, name="", position="", team=""):
    if pid not in players:
        players[pid] = {"player_id": pid, "name": name or pid, "position": position or "", "team": team or "", "targets": 0, "receptions": 0, "receiving_yards": 0, "receiving_tds": 0, "carries": 0, "rushing_yards": 0, "rushing_tds": 0, "red_zone_targets": 0, "red_zone_receptions": 0, "red_zone_carries": 0, "red_zone_receiving_tds": 0, "red_zone_rushing_tds": 0, "pass_attempts": 0, "red_zone_pass_attempts": 0, "weekly": {}}
    return players[pid]

rec = df[df["pass_attempt"].eq(1) & df["receiver_id"].notna() & df["posteam"].notna()].copy()
for pid, g in rec.groupby("receiver_id"):
    row=g.iloc[0]; pos=str(row.get("receiver_position") or "")
    if pos not in SKILL: continue
    p=ensure(pid,row.get("receiver_name") or pid,pos,row.get("posteam") or "")
    p["targets"] += len(g); p["receptions"] += int(g["complete_pass"].fillna(0).eq(1).sum()); p["receiving_yards"] += int(g["receiving_yards"].fillna(0).sum()); p["receiving_tds"] += int(g["pass_touchdown"].fillna(0).eq(1).sum())
    rz=g[g["yardline_100"].le(20)]; p["red_zone_targets"] += len(rz); p["red_zone_receptions"] += int(rz["complete_pass"].fillna(0).eq(1).sum()); p["red_zone_receiving_tds"] += int(rz["pass_touchdown"].fillna(0).eq(1).sum())
    for week,w in g.groupby("week"):
        k=int(week); e=p["weekly"].setdefault(k,{"week":k,"targets":0,"carries":0,"receptions":0,"red_zone_targets":0,"red_zone_carries":0}); e["targets"]+=len(w); e["receptions"]+=int(w["complete_pass"].fillna(0).eq(1).sum()); e["red_zone_targets"]+=int(w["yardline_100"].le(20).sum())

rush=df[df["rush_attempt"].eq(1) & df["rusher_player_id"].notna() & df["posteam"].notna()].copy()
for pid,g in rush.groupby("rusher_player_id"):
    row=g.iloc[0]; pos=str(row.get("rusher_position") or "")
    if pos not in SKILL: continue
    p=ensure(pid,row.get("rusher_player_name") or pid,pos,row.get("posteam") or "")
    p["carries"]+=len(g); p["rushing_yards"]+=int(g["rushing_yards"].fillna(0).sum()); p["rushing_tds"]+=int(g["rush_touchdown"].fillna(0).eq(1).sum())
    rz=g[g["yardline_100"].le(20)]; p["red_zone_carries"]+=len(rz); p["red_zone_rushing_tds"]+=int(rz["rush_touchdown"].fillna(0).eq(1).sum())
    for week,w in g.groupby("week"):
        k=int(week); e=p["weekly"].setdefault(k,{"week":k,"targets":0,"carries":0,"receptions":0,"red_zone_targets":0,"red_zone_carries":0}); e["carries"]+=len(w); e["red_zone_carries"]+=int(w["yardline_100"].le(20).sum())

passers=df[df["pass_attempt"].eq(1) & df["passer_player_id"].notna() & df["posteam"].notna()].copy()
for pid,g in passers.groupby("passer_player_id"):
    row=g.iloc[0]; p=ensure(pid,row.get("passer_player_name") or pid,"QB",row.get("posteam") or "")
    p["pass_attempts"]+=len(g); p["red_zone_pass_attempts"]+=int(g["yardline_100"].le(20).sum())
    for week,w in g.groupby("week"):
        k=int(week); e=p["weekly"].setdefault(k,{"week":k,"targets":0,"carries":0,"receptions":0,"red_zone_targets":0,"red_zone_carries":0}); e["pass_attempts"]=e.get("pass_attempts",0)+len(w); e["red_zone_pass_attempts"]=e.get("red_zone_pass_attempts",0)+int(w["yardline_100"].le(20).sum())

result=[]
for p in players.values():
    if p["targets"]<10 and p["carries"]<10 and p["pass_attempts"]<10: continue
    p["weekly"]=[p["weekly"][k] for k in sorted(p["weekly"])]
    result.append(p)
result.sort(key=lambda p:(-max(p["targets"],p["carries"],p["pass_attempts"]),p["name"]))
out=Path("data/players.json"); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({"season":season,"players":result},indent=2)); print(f"Wrote {out} with {len(result)} players.")
