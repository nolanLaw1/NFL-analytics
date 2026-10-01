#!/usr/bin/env python3
"""
Build league leaderboard data from nflverse play-by-play.

Usage:
    python build_league_data.py 2025

Creates:
    data/league.json
"""

import json
import sys
from pathlib import Path
import pandas as pd

DATA_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
SKILL_POSITIONS = {"WR", "RB", "TE"}

COLS = [
    "season", "week", "game_type", "posteam", "pass_attempt",
    "receiver_player_id", "receiver_player_name", "receiver_player_position",
    "rusher_player_id", "rusher_player_name", "rusher_player_position",
    "yardline_100", "complete_pass", "receiving_yards", "rushing_yards",
    "touchdown"
]

def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    pbp = pd.read_parquet(DATA_URL.format(season=season), columns=COLS)
    pbp = pbp[pbp["game_type"].eq("REG")].copy()
    pbp["red_zone"] = pd.to_numeric(pbp["yardline_100"], errors="coerce").le(20)

    # Normal receiving leaderboards
    rec_plays = pbp[
        pbp["receiver_player_id"].notna()
        & pbp["receiver_player_position"].isin(SKILL_POSITIONS)
    ].copy()
    rec_plays["target"] = rec_plays["pass_attempt"].fillna(0).eq(1).astype(int)
    rec_plays["reception"] = rec_plays["complete_pass"].fillna(0).eq(1).astype(int)
    rec_plays["receiving_yards"] = pd.to_numeric(rec_plays["receiving_yards"], errors="coerce").fillna(0)
    rec_plays["td"] = rec_plays["touchdown"].fillna(0).eq(1).astype(int)

    rec = rec_plays.groupby(
        ["receiver_player_id","receiver_player_name","receiver_player_position"],
        dropna=False
    ).agg(
        targets=("target","sum"),
        receptions=("reception","sum"),
        receiving_yards=("receiving_yards","sum"),
        receiving_tds=("td","sum")
    ).reset_index()

    rec = rec.rename(columns={
        "receiver_player_id":"player_id",
        "receiver_player_name":"name",
        "receiver_player_position":"position"
    })

    # Normal rushing leaderboards
    rush_plays = pbp[
        pbp["rusher_player_id"].notna()
        & pbp["rusher_player_position"].isin(SKILL_POSITIONS)
    ].copy()
    rush_plays["rushing_yards"] = pd.to_numeric(rush_plays["rushing_yards"], errors="coerce").fillna(0)
    rush_plays["td"] = rush_plays["touchdown"].fillna(0).eq(1).astype(int)

    rush = rush_plays.groupby(
        ["rusher_player_id","rusher_player_name","rusher_player_position"],
        dropna=False
    ).agg(
        carries=("rusher_player_id","size"),
        rushing_yards=("rushing_yards","sum"),
        rushing_tds=("td","sum")
    ).reset_index()

    rush = rush.rename(columns={
        "rusher_player_id":"player_id",
        "rusher_player_name":"name",
        "rusher_player_position":"position"
    })

    # Red-zone receiving
    rz_rec = rec_plays[rec_plays["red_zone"]].copy()
    rz_rec_group = rz_rec.groupby(
        ["receiver_player_id","receiver_player_name","receiver_player_position"],
        dropna=False
    ).agg(
        red_zone_targets=("target","sum"),
        red_zone_receptions=("reception","sum"),
        red_zone_receiving_tds=("td","sum")
    ).reset_index().rename(columns={
        "receiver_player_id":"player_id",
        "receiver_player_name":"name",
        "receiver_player_position":"position"
    })

    team_rz_targets = rz_rec.groupby("posteam")["target"].sum().to_dict()

    # Attach team to players using the most common receiving team in the season.
    team_map = (
        rec_plays.groupby(["receiver_player_id","posteam"]).size()
        .reset_index(name="plays")
        .sort_values(["receiver_player_id","plays"], ascending=[True,False])
        .drop_duplicates("receiver_player_id")
        .set_index("receiver_player_id")["posteam"]
        .to_dict()
    )
    rec["team"] = rec["player_id"].map(team_map)
    rush["team"] = rush["player_id"].map(team_map)
    rz_rec_group["team"] = rz_rec_group["player_id"].map(team_map)
    rz_rec_group["team_rz_targets"] = rz_rec_group["team"].map(team_rz_targets).fillna(0).astype(int)
    rz_rec_group["rz_target_share"] = (
        rz_rec_group["red_zone_targets"] / rz_rec_group["team_rz_targets"].replace(0, pd.NA)
    ).fillna(0)

    def rows(df, fields):
        out = []
        for _, r in df.iterrows():
            item = {"player_id": r["player_id"], "name": r["name"], "position": r["position"], "team": r.get("team")}
            for f in fields:
                v = r[f]
                item[f] = int(v) if pd.notna(v) and f != "rz_target_share" else (round(float(v),4) if pd.notna(v) else 0)
            out.append(item)
        return out

    # Red-zone rushing
    rz_rush = rush_plays[rush_plays["red_zone"]].groupby(
        ["rusher_player_id","rusher_player_name","rusher_player_position"],
        dropna=False
    ).agg(
        red_zone_carries=("rusher_player_id","size"),
        red_zone_rushing_yards=("rushing_yards","sum"),
        red_zone_rushing_tds=("td","sum")
    ).reset_index().rename(columns={
        "rusher_player_id":"player_id",
        "rusher_player_name":"name",
        "rusher_player_position":"position"
    })
    rz_rush["team"] = rz_rush["player_id"].map(team_map)

    data = {
        "season": season,
        "normal": {
            "targets": rows(rec.sort_values(["targets","name"], ascending=[False,True]), ["targets"]),
            "carries": rows(rush.sort_values(["carries","name"], ascending=[False,True]), ["carries"]),
            "receptions": rows(rec.sort_values(["receptions","name"], ascending=[False,True]), ["receptions"]),
            "receiving_yards": rows(rec.sort_values(["receiving_yards","name"], ascending=[False,True]), ["receiving_yards"]),
            "rushing_yards": rows(rush.sort_values(["rushing_yards","name"], ascending=[False,True]), ["rushing_yards"]),
            "receiving_tds": rows(rec.sort_values(["receiving_tds","name"], ascending=[False,True]), ["receiving_tds"]),
            "rushing_tds": rows(rush.sort_values(["rushing_tds","name"], ascending=[False,True]), ["rushing_tds"]),
        },
        "red_zone": {
            "targets": rows(rz_rec_group.sort_values(["red_zone_targets","name"], ascending=[False,True]), ["red_zone_targets","team_rz_targets","rz_target_share"]),
            "carries": rows(rz_rush.sort_values(["red_zone_carries","name"], ascending=[False,True]), ["red_zone_carries"]),
            "receptions": rows(rz_rec_group.sort_values(["red_zone_receptions","name"], ascending=[False,True]), ["red_zone_receptions"]),
            "receiving_tds": rows(rz_rec_group.sort_values(["red_zone_receiving_tds","name"], ascending=[False,True]), ["red_zone_receiving_tds"]),
            "rushing_tds": rows(rz_rush.sort_values(["red_zone_rushing_tds","name"], ascending=[False,True]), ["red_zone_rushing_tds"]),
        }
    }

    out = Path("data")
    out.mkdir(exist_ok=True)
    (out / "league.json").write_text(json.dumps(data, indent=2, default=str))
    print(f"Wrote {out / 'league.json'}")

if __name__ == "__main__":
    main()
