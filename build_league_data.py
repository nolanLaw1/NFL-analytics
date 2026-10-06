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


# ============================================================
# CONFIG
# ============================================================

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026

PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "pbp/play_by_play_{season}.parquet"
)

PLAYERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "players/players.csv"
)

SKILL_POSITIONS = {"WR", "RB", "TE"}


# ============================================================
# HELPERS
# ============================================================

def clean_id(series):
    return (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )


def safe_int(value):
    if pd.isna(value):
        return 0

    try:
        return int(value)
    except Exception:
        return 0


def safe_float(value):
    if pd.isna(value):
        return 0.0

    try:
        return float(value)
    except Exception:
        return 0.0


# ============================================================
# LOAD DATA
# ============================================================

def load_pbp():

    print(f"Loading {SEASON} play-by-play data...")

    pbp_url = PBP_URL.format(season=SEASON)

    pbp = pd.read_parquet(pbp_url)

    print(
        f"Loaded {len(pbp):,} total plays."
    )

    # Regular season only
    if "season_type" in pbp.columns:

        pbp = pbp[
            pbp["season_type"].eq("REG")
        ].copy()

    if pbp.empty:

        raise RuntimeError(
            f"No regular-season play-by-play data found for {SEASON}."
        )

    print(
        f"Using {len(pbp):,} regular-season plays."
    )

    return pbp


def load_players():

    print("Loading nflverse player database...")

    players = pd.read_csv(
        PLAYERS_URL,
        low_memory=False
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
            "Could not find player ID column in players.csv."
        )

    if position_col is None:

        raise RuntimeError(
            "Could not find position column in players.csv."
        )

    players = players[
        [
            player_id_col,
            position_col
        ]
    ].copy()

    players = players.rename(
        columns={
            player_id_col: "player_id",
            position_col: "position"
        }
    )

    players["player_id"] = clean_id(
        players["player_id"]
    )

    players["position"] = (
        players["position"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    players = players.drop_duplicates(
        subset=["player_id"]
    )

    print(
        f"Loaded {len(players):,} players."
    )

    return players


# ============================================================
# MAIN
# ============================================================

def main():

    pbp = load_pbp()

    players = load_players()


    # ========================================================
    # NORMALIZE IMPORTANT PBP COLUMNS
    # ========================================================

    required_columns = [
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

    for column in required_columns:

        if column not in pbp.columns:

            pbp[column] = 0


    pbp["receiver_player_id"] = clean_id(
        pbp["receiver_player_id"]
    )

    pbp["rusher_player_id"] = clean_id(
        pbp["rusher_player_id"]
    )

    pbp["posteam"] = (
        pbp["posteam"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    pbp["yardline_100"] = pd.to_numeric(
        pbp["yardline_100"],
        errors="coerce"
    )

    pbp["pass_attempt"] = pd.to_numeric(
        pbp["pass_attempt"],
        errors="coerce"
    ).fillna(0)

    pbp["complete_pass"] = pd.to_numeric(
        pbp["complete_pass"],
        errors="coerce"
    ).fillna(0)

    pbp["touchdown"] = pd.to_numeric(
        pbp["touchdown"],
        errors="coerce"
    ).fillna(0)

    pbp["receiving_yards"] = pd.to_numeric(
        pbp["receiving_yards"],
        errors="coerce"
    ).fillna(0)

    pbp["rushing_yards"] = pd.to_numeric(
        pbp["rushing_yards"],
        errors="coerce"
    ).fillna(0)


    # ========================================================
    # ATTACH RECEIVER POSITIONS
    # ========================================================

    receiver_positions = players.rename(
        columns={
            "player_id": "receiver_player_id",
            "position": "receiver_player_position"
        }
    )

    pbp = pbp.merge(
        receiver_positions,
        on="receiver_player_id",
        how="left"
    )


    # ========================================================
    # ATTACH RUSHER POSITIONS
    # ========================================================

    rusher_positions = players.rename(
        columns={
            "player_id": "rusher_player_id",
            "position": "rusher_player_position"
        }
    )

    pbp = pbp.merge(
        rusher_positions,
        on="rusher_player_id",
        how="left"
    )


    # ========================================================
    # RED-ZONE FLAG
    # ========================================================

    pbp["red_zone"] = (
        pbp["yardline_100"].notna()
        &
        (pbp["yardline_100"] <= 20)
    )


    # ========================================================
    # RECEIVING PLAYS
    # ========================================================

    rec_plays = pbp[
        pbp["receiver_player_id"].ne("")
        &
        pbp["receiver_player_position"].isin(
            SKILL_POSITIONS
        )
    ].copy()

    rec_plays["target"] = (
        rec_plays["pass_attempt"]
        .eq(1)
        .astype(int)
    )

    rec_plays["reception"] = (
        rec_plays["complete_pass"]
        .eq(1)
        .astype(int)
    )

    rec_plays["receiving_td"] = (
        (
            rec_plays["complete_pass"].eq(1)
        )
        &
        (
            rec_plays["touchdown"].eq(1)
        )
    ).astype(int)


    # ========================================================
    # NORMAL RECEIVING TOTALS
    # ========================================================

    rec = (
        rec_plays
        .groupby(
            [
                "receiver_player_id",
                "receiver_player_name",
                "receiver_player_position"
            ],
            dropna=False
        )
        .agg(
            targets=("target", "sum"),
            receptions=("reception", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            receiving_tds=("receiving_td", "sum")
        )
        .reset_index()
    )

    rec = rec.rename(
        columns={
            "receiver_player_id": "player_id",
            "receiver_player_name": "name",
            "receiver_player_position": "position"
        }
    )


    # ========================================================
    # RUSHING PLAYS
    # ========================================================

    rush_plays = pbp[
        pbp["rusher_player_id"].ne("")
        &
        pbp["rusher_player_position"].isin(
            SKILL_POSITIONS
        )
    ].copy()

    # Only actual rush attempts
    if "rush_attempt" in rush_plays.columns:

        rush_plays["rush_attempt_flag"] = (
            pd.to_numeric(
                rush_plays["rush_attempt"],
                errors="coerce"
            )
            .fillna(0)
            .eq(1)
            .astype(int)
        )

    else:

        rush_plays["rush_attempt_flag"] = 1


    rush_plays = rush_plays[
        rush_plays["rush_attempt_flag"].eq(1)
    ].copy()

    rush_plays["rushing_td"] = (
        rush_plays["touchdown"]
        .eq(1)
        .astype(int)
    )


    # ========================================================
    # NORMAL RUSHING TOTALS
    # ========================================================

    rush = (
        rush_plays
        .groupby(
            [
                "rusher_player_id",
                "rusher_player_name",
                "rusher_player_position"
            ],
            dropna=False
        )
        .agg(
            carries=("rush_attempt_flag", "sum"),
            rushing_yards=("rushing_yards", "sum"),
            rushing_tds=("rushing_td", "sum")
        )
        .reset_index()
    )

    rush = rush.rename(
        columns={
            "rusher_player_id": "player_id",
            "rusher_player_name": "name",
            "rusher_player_position": "position"
        }
    )


    # ========================================================
    # RED-ZONE RECEIVING
    # ========================================================

    rz_rec = rec_plays[
        rec_plays["red_zone"]
    ].copy()


    rz_rec_group = (
        rz_rec
        .groupby(
            [
                "receiver_player_id",
                "receiver_player_name",
                "receiver_player_position"
            ],
            dropna=False
        )
        .agg(
            red_zone_targets=("target", "sum"),
            red_zone_receptions=("reception", "sum"),
            red_zone_receiving_yards=(
                "receiving_yards",
                "sum"
            ),
            red_zone_receiving_tds=(
                "receiving_td",
                "sum"
            )
        )
        .reset_index()
        .rename(
            columns={
                "receiver_player_id": "player_id",
                "receiver_player_name": "name",
                "receiver_player_position": "position"
            }
        )
    )


    # ========================================================
    # RED-ZONE RUSHING
    # ========================================================

    rz_rush = (
        rush_plays[
            rush_plays["red_zone"]
        ]
        .groupby(
            [
                "rusher_player_id",
                "rusher_player_name",
                "rusher_player_position"
            ],
            dropna=False
        )
        .agg(
            red_zone_carries=(
                "rush_attempt_flag",
                "sum"
            ),
            red_zone_rushing_yards=(
                "rushing_yards",
                "sum"
            ),
            red_zone_rushing_tds=(
                "rushing_td",
                "sum"
            )
        )
        .reset_index()
        .rename(
            columns={
                "rusher_player_id": "player_id",
                "rusher_player_name": "name",
                "rusher_player_position": "position"
            }
        )
    )


    # ========================================================
    # TEAM MAPPING
    # ========================================================

    # Receiver team
    receiver_team_map = (
        rec_plays[
            rec_plays["receiver_player_id"].ne("")
            &
            rec_plays["posteam"].ne("")
        ]
        .groupby(
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


    # Rusher team
    rusher_team_map = (
        rush_plays[
            rush_plays["rusher_player_id"].ne("")
            &
            rush_plays["posteam"].ne("")
        ]
        .groupby(
            [
                "rusher_player_id",
                "posteam"
            ]
        )
        .size()
        .reset_index(name="plays")
        .sort_values(
            [
                "rusher_player_id",
                "plays"
            ],
            ascending=[
                True,
                False
            ]
        )
        .drop_duplicates(
            "rusher_player_id"
        )
        .set_index(
            "rusher_player_id"
        )["posteam"]
        .to_dict()
    )


    rec["team"] = (
        rec["player_id"]
        .map(receiver_team_map)
        .fillna("")
    )

    rush["team"] = (
        rush["player_id"]
        .map(rusher_team_map)
        .fillna("")
    )

    rz_rec_group["team"] = (
        rz_rec_group["player_id"]
        .map(receiver_team_map)
        .fillna("")
    )

    rz_rush["team"] = (
        rz_rush["player_id"]
        .map(rusher_team_map)
        .fillna("")
    )


    # ========================================================
    # TEAM RED-ZONE TARGET TOTALS
    # ========================================================

    team_rz_targets = (
        rz_rec
        .groupby("posteam")["target"]
        .sum()
        .to_dict()
    )

    rz_rec_group["team_rz_targets"] = (
        rz_rec_group["team"]
        .map(team_rz_targets)
        .fillna(0)
        .astype(int)
    )


    # ========================================================
    # RED-ZONE TARGET SHARE
    # ========================================================

    rz_rec_group["rz_target_share"] = 0.0

    valid = (
        rz_rec_group["team_rz_targets"] > 0
    )

    rz_rec_group.loc[
        valid,
        "rz_target_share"
    ] = (
        rz_rec_group.loc[
            valid,
            "red_zone_targets"
        ]
        /
        rz_rec_group.loc[
            valid,
            "team_rz_targets"
        ]
    )


    # ========================================================
    # HELPER FOR JSON ROWS
    # ========================================================

    def rows(df, fields):

        output = []

        for _, r in df.iterrows():

            item = {
                "player_id": str(
                    r["player_id"]
                ),
                "name": (
                    ""
                    if pd.isna(r["name"])
                    else str(r["name"])
                ),
                "position": (
                    ""
                    if pd.isna(r["position"])
                    else str(r["position"])
                ),
                "team": (
                    ""
                    if pd.isna(r.get("team"))
                    else str(r.get("team"))
                )
            }

            for field in fields:

                value = r.get(
                    field,
                    0
                )

                if field == "rz_target_share":

                    item[field] = round(
                        safe_float(value),
                        4
                    )

                else:

                    item[field] = safe_int(
                        value
                    )

            output.append(item)

        return output


    # ========================================================
    # SORTED DATASETS
    # ========================================================

    normal_targets = rec.sort_values(
        [
            "targets",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    normal_carries = rush.sort_values(
        [
            "carries",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    normal_receptions = rec.sort_values(
        [
            "receptions",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    normal_receiving_yards = rec.sort_values(
        [
            "receiving_yards",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    normal_rushing_yards = rush.sort_values(
        [
            "rushing_yards",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    normal_receiving_tds = rec.sort_values(
        [
            "receiving_tds",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    normal_rushing_tds = rush.sort_values(
        [
            "rushing_tds",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )


    rz_targets_sorted = rz_rec_group.sort_values(
        [
            "red_zone_targets",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    rz_carries_sorted = rz_rush.sort_values(
        [
            "red_zone_carries",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    rz_receptions_sorted = rz_rec_group.sort_values(
        [
            "red_zone_receptions",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    rz_receiving_yards_sorted = rz_rec_group.sort_values(
        [
            "red_zone_receiving_yards",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    rz_receiving_tds_sorted = rz_rec_group.sort_values(
        [
            "red_zone_receiving_tds",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    rz_rushing_yards_sorted = rz_rush.sort_values(
        [
            "red_zone_rushing_yards",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )

    rz_rushing_tds_sorted = rz_rush.sort_values(
        [
            "red_zone_rushing_tds",
            "name"
        ],
        ascending=[
            False,
            True
        ]
    )


    # ========================================================
    # BUILD OUTPUT
    # ========================================================

    data = {

        "season": SEASON,

        "normal": {

            "targets": rows(
                normal_targets,
                [
                    "targets"
                ]
            ),

            "carries": rows(
                normal_carries,
                [
                    "carries"
                ]
            ),

            "receptions": rows(
                normal_receptions,
                [
                    "receptions"
                ]
            ),

            "receiving_yards": rows(
                normal_receiving_yards,
                [
                    "receiving_yards"
                ]
            ),

            "rushing_yards": rows(
                normal_rushing_yards,
                [
                    "rushing_yards"
                ]
            ),

            "receiving_tds": rows(
                normal_receiving_tds,
                [
                    "receiving_tds"
                ]
            ),

            "rushing_tds": rows(
                normal_rushing_tds,
                [
                    "rushing_tds"
                ]
            )
        },


        "red_zone": {

            # -----------------------------------------------
            # TARGETS
            # -----------------------------------------------

            "targets": rows(
                rz_targets_sorted,
                [
                    "red_zone_targets",
                    "team_rz_targets",
                    "rz_target_share"
                ]
            ),

            # -----------------------------------------------
            # CARRIES
            # -----------------------------------------------

            "carries": rows(
                rz_carries_sorted,
                [
                    "red_zone_carries"
                ]
            ),

            # -----------------------------------------------
            # RECEPTIONS
            # -----------------------------------------------

            "receptions": rows(
                rz_receptions_sorted,
                [
                    "red_zone_receptions"
                ]
            ),

            # -----------------------------------------------
            # RECEIVING YARDS
            # -----------------------------------------------

            "receiving_yards": rows(
                rz_receiving_yards_sorted,
                [
                    "red_zone_receiving_yards"
                ]
            ),

            # -----------------------------------------------
            # RECEIVING TDs
            # -----------------------------------------------

            "receiving_tds": rows(
                rz_receiving_tds_sorted,
                [
                    "red_zone_receiving_tds"
                ]
            ),

            # -----------------------------------------------
            # RUSHING YARDS
            # -----------------------------------------------

            "rushing_yards": rows(
                rz_rushing_yards_sorted,
                [
                    "red_zone_rushing_yards"
                ]
            ),

            # -----------------------------------------------
            # RUSHING TDs
            # -----------------------------------------------

            "rushing_tds": rows(
                rz_rushing_tds_sorted,
                [
                    "red_zone_rushing_tds"
                ]
            )
        }
    }


    # ========================================================
    # WRITE FILE
    # ========================================================

    output_dir = Path("data")

    output_dir.mkdir(
        exist_ok=True
    )

    output_file = (
        output_dir / "league.json"
    )

    output_file.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False
        )
    )

    print()
    print("========================================")
    print("LEAGUE DATA BUILD COMPLETE")
    print("========================================")
    print(
        f"Season: {SEASON}"
    )
    print(
        f"Receiving players: {len(rec):,}"
    )
    print(
        f"Rushing players: {len(rush):,}"
    )
    print(
        f"RZ receiving players: {len(rz_rec_group):,}"
    )
    print(
        f"RZ rushing players: {len(rz_rush):,}"
    )
    print(
        f"Wrote: {output_file}"
    )
    print("========================================")


if __name__ == "__main__":
    main()
