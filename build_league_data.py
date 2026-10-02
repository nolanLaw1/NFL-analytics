#!/usr/bin/env python3
"""
Build league leaderboard data from nflverse play-by-play.

Usage:
    python build_league_data.py 2026

Creates:
    data/league.json
"""

import json
import sys
from pathlib import Path

import pandas as pd


PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

# Current nflverse player database.
# This is the source of truth for player IDs and positions.
PLAYERS_URL = "https://github.com/nflverse/nflverse-data/releases/download/players/players.csv"

SKILL_POSITIONS = {"WR", "RB", "TE"}

COLS = [
    "season",
    "week",
    "season_type",
    "posteam",
    "pass_attempt",
    "receiver_player_id",
    "receiver_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "yardline_100",
    "complete_pass",
    "receiving_yards",
    "rushing_yards",
    "touchdown",
]


def main():

    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2026

    print(f"Loading {season} play-by-play data...")

    pbp = pd.read_parquet(
        PBP_URL.format(season=season),
        columns=COLS
    )

    # Regular season only.
    pbp = pbp[
        pbp["season_type"].eq("REG")
    ].copy()

    if pbp.empty:
        raise RuntimeError(
            f"No regular-season play-by-play data found for {season}."
        )

    print(
        f"Loaded {len(pbp):,} regular-season plays."
    )

    # ---------------------------------------------------------
    # Load player positions
    # ---------------------------------------------------------

    print("Loading nflverse player database...")

    players = pd.read_csv(
        PLAYERS_URL
    )

    player_id_col = None

    for col in [
        "gsis_id",
        "player_id",
        "id"
    ]:
        if col in players.columns:
            player_id_col = col
            break

    position_col = None

    for col in [
        "position",
        "position_group"
    ]:
        if col in players.columns:
            position_col = col
            break

    if player_id_col is None:
        raise RuntimeError(
            "Could not find a player ID column in nflverse players.csv."
        )

    if position_col is None:
        raise RuntimeError(
            "Could not find a position column in nflverse players.csv."
        )

    players = players[
        [player_id_col, position_col]
    ].drop_duplicates()

    players = players.rename(
        columns={
            player_id_col: "player_id",
            position_col: "position",
        }
    )

    # ---------------------------------------------------------
    # Attach receiver positions
    # ---------------------------------------------------------

    pbp = pbp.merge(
        players.rename(
            columns={
                "player_id": "receiver_player_id",
                "position": "receiver_player_position",
            }
        ),
        on="receiver_player_id",
        how="left",
    )

    # ---------------------------------------------------------
    # Attach rusher positions
    # ---------------------------------------------------------

    pbp = pbp.merge(
        players.rename(
            columns={
                "player_id": "rusher_player_id",
                "position": "rusher_player_position",
            }
        ),
        on="rusher_player_id",
        how="left",
    )

    # ---------------------------------------------------------
    # Red zone
    # ---------------------------------------------------------

    pbp["red_zone"] = (
        pd.to_numeric(
            pbp["yardline_100"],
            errors="coerce"
        ) <= 20
    )

    # ---------------------------------------------------------
    # Receiving
    # ---------------------------------------------------------

    rec_plays = pbp[
        pbp["receiver_player_id"].notna()
        & pbp["receiver_player_position"].isin(
            SKILL_POSITIONS
        )
    ].copy()

    rec_plays["target"] = (
        rec_plays["pass_attempt"]
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    rec_plays["reception"] = (
        rec_plays["complete_pass"]
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    rec_plays["receiving_yards"] = pd.to_numeric(
        rec_plays["receiving_yards"],
        errors="coerce"
    ).fillna(0)

    rec_plays["td"] = (
        rec_plays["touchdown"]
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    rec = rec_plays.groupby(
        [
            "receiver_player_id",
            "receiver_player_name",
            "receiver_player_position",
        ],
        dropna=False,
    ).agg(
        targets=("target", "sum"),
        receptions=("reception", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        receiving_tds=("td", "sum"),
    ).reset_index()

    rec = rec.rename(
        columns={
            "receiver_player_id": "player_id",
            "receiver_player_name": "name",
            "receiver_player_position": "position",
        }
    )

    # ---------------------------------------------------------
    # Rushing
    # ---------------------------------------------------------

    rush_plays = pbp[
        pbp["rusher_player_id"].notna()
        & pbp["rusher_player_position"].isin(
            SKILL_POSITIONS
        )
    ].copy()

    rush_plays["rushing_yards"] = pd.to_numeric(
        rush_plays["rushing_yards"],
        errors="coerce"
    ).fillna(0)

    rush_plays["td"] = (
        rush_plays["touchdown"]
        .fillna(0)
        .eq(1)
        .astype(int)
    )

    rush = rush_plays.groupby(
        [
            "rusher_player_id",
            "rusher_player_name",
            "rusher_player_position",
        ],
        dropna=False,
    ).agg(
        carries=("rusher_player_id", "size"),
        rushing_yards=("rushing_yards", "sum"),
        rushing_tds=("td", "sum"),
    ).reset_index()

    rush = rush.rename(
        columns={
            "rusher_player_id": "player_id",
            "rusher_player_name": "name",
            "rusher_player_position": "position",
        }
    )

    # ---------------------------------------------------------
    # Red-zone receiving
    # ---------------------------------------------------------

    rz_rec = rec_plays[
        rec_plays["red_zone"]
    ].copy()

    rz_rec_group = rz_rec.groupby(
        [
            "receiver_player_id",
            "receiver_player_name",
            "receiver_player_position",
        ],
        dropna=False,
    ).agg(
        red_zone_targets=("target", "sum"),
        red_zone_receptions=("reception", "sum"),
        red_zone_receiving_tds=("td", "sum"),
    ).reset_index().rename(
        columns={
            "receiver_player_id": "player_id",
            "receiver_player_name": "name",
            "receiver_player_position": "position",
        }
    )

    team_rz_targets = (
        rz_rec.groupby("posteam")["target"]
        .sum()
        .to_dict()
    )

    # ---------------------------------------------------------
    # Determine each player's primary team
    # ---------------------------------------------------------

    team_map = (
        rec_plays.groupby(
            [
                "receiver_player_id",
                "posteam"
            ]
        )
        .size()
        .reset_index(name="plays")
        .sort_values(
            [
                "receiver_player_id",
                "plays"
            ],
            ascending=[
                True,
                False
            ]
        )
        .drop_duplicates(
            "receiver_player_id"
        )
        .set_index(
            "receiver_player_id"
        )["posteam"]
        .to_dict()
    )

    rec["team"] = (
        rec["player_id"]
        .map(team_map)
    )

    rush["team"] = (
        rush["player_id"]
        .map(team_map)
    )

    rz_rec_group["team"] = (
        rz_rec_group["player_id"]
        .map(team_map)
    )

    rz_rec_group["team_rz_targets"] = (
        rz_rec_group["team"]
        .map(team_rz_targets)
        .fillna(0)
        .astype(int)
    )

    rz_rec_group["rz_target_share"] = (
        rz_rec_group["red_zone_targets"]
        /
        rz_rec_group["team_rz_targets"]
        .replace(0, pd.NA)
    ).fillna(0)

    # ---------------------------------------------------------
    # Helper
    # ---------------------------------------------------------

    def rows(df, fields):

        out = []

        for _, r in df.iterrows():

            item = {
                "player_id": r["player_id"],
                "name": r["name"],
                "position": r["position"],
                "team": (
                    ""
                    if pd.isna(r.get("team"))
                    else str(r.get("team"))
                ),
            }

            for f in fields:

                v = r[f]

                if f == "rz_target_share":

                    item[f] = (
                        round(float(v), 4)
                        if pd.notna(v)
                        else 0
                    )

                else:

                    item[f] = (
                        int(v)
                        if pd.notna(v)
                        else 0
                    )

            out.append(item)

        return out

    # ---------------------------------------------------------
    # Red-zone rushing
    # ---------------------------------------------------------

    rz_rush = (
        rush_plays[
            rush_plays["red_zone"]
        ]
        .groupby(
            [
                "rusher_player_id",
                "rusher_player_name",
                "rusher_player_position",
            ],
            dropna=False,
        )
        .agg(
            red_zone_carries=(
                "rusher_player_id",
                "size"
            ),
            red_zone_rushing_yards=(
                "rushing_yards",
                "sum"
            ),
            red_zone_rushing_tds=(
                "td",
                "sum"
            ),
        )
        .reset_index()
        .rename(
            columns={
                "rusher_player_id": "player_id",
                "rusher_player_name": "name",
                "rusher_player_position": "position",
            }
        )
    )

    rz_rush["team"] = (
        rz_rush["player_id"]
        .map(team_map)
    )

    # ---------------------------------------------------------
    # Validate data before writing
    # ---------------------------------------------------------

    if rec.empty and rush.empty:
        raise RuntimeError(
            "League data contains no receiving or rushing players. "
            "The player-position mapping likely failed."
        )

    print(
        f"Found {len(rec):,} receiving players."
    )

    print(
        f"Found {len(rush):,} rushing players."
    )

    # ---------------------------------------------------------
    # Build output
    # ---------------------------------------------------------

    data = {
        "season": season,

        "normal": {

            "targets": rows(
                rec.sort_values(
                    [
                        "targets",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["targets"],
            ),

            "carries": rows(
                rush.sort_values(
                    [
                        "carries",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["carries"],
            ),

            "receptions": rows(
                rec.sort_values(
                    [
                        "receptions",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["receptions"],
            ),

            "receiving_yards": rows(
                rec.sort_values(
                    [
                        "receiving_yards",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["receiving_yards"],
            ),

            "rushing_yards": rows(
                rush.sort_values(
                    [
                        "rushing_yards",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["rushing_yards"],
            ),

            "receiving_tds": rows(
                rec.sort_values(
                    [
                        "receiving_tds",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["receiving_tds"],
            ),

            "rushing_tds": rows(
                rush.sort_values(
                    [
                        "rushing_tds",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["rushing_tds"],
            ),
        },

        "red_zone": {

            "targets": rows(
                rz_rec_group.sort_values(
                    [
                        "red_zone_targets",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                [
                    "red_zone_targets",
                    "team_rz_targets",
                    "rz_target_share",
                ],
            ),

            "carries": rows(
                rz_rush.sort_values(
                    [
                        "red_zone_carries",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["red_zone_carries"],
            ),

            "receptions": rows(
                rz_rec_group.sort_values(
                    [
                        "red_zone_receptions",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["red_zone_receptions"],
            ),

            "receiving_tds": rows(
                rz_rec_group.sort_values(
                    [
                        "red_zone_receiving_tds",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["red_zone_receiving_tds"],
            ),

            "rushing_tds": rows(
                rz_rush.sort_values(
                    [
                        "red_zone_rushing_tds",
                        "name"
                    ],
                    ascending=[
                        False,
                        True
                    ]
                ),
                ["red_zone_rushing_tds"],
            ),
        },
    }

    # ---------------------------------------------------------
    # Write JSON
    # ---------------------------------------------------------

    out = Path("data")
    out.mkdir(
        exist_ok=True
    )

    output_file = (
        out / "league.json"
    )

    output_file.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False
        )
    )

    print(
        f"Wrote {output_file}"
    )


if __name__ == "__main__":
    main()
