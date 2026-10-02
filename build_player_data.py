import sys
import json
from pathlib import Path

import pandas as pd


SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "players.json"


def first_existing(df, names, default=None):
    for name in names:
        if name in df.columns:
            return name
    return default


def clean_number(value):
    if pd.isna(value):
        return 0

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


def load_player_stats():
    url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"stats_player/stats_player_week_{SEASON}.csv"
    )

    print(f"Downloading player statistics from {url}")

    df = pd.read_csv(
        url,
        low_memory=False
    )

    print(f"Loaded {len(df)} player-stat rows.")

    return df


def load_pbp():
    url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"pbp/play_by_play_{SEASON}.parquet"
    )

    print(f"Downloading play-by-play data from {url}")

    return pd.read_parquet(url)


def build_player_totals(stats):
    player_id_col = first_existing(
        stats,
        [
            "player_id",
            "player_player_id",
            "gsis_id"
        ]
    )

    name_col = first_existing(
        stats,
        [
            "player_name",
            "name"
        ]
    )

    position_col = first_existing(
        stats,
        [
            "position"
        ]
    )

    team_col = first_existing(
        stats,
        [
            "recent_team",
            "team",
            "posteam"
        ]
    )

    if not player_id_col:
        raise RuntimeError(
            f"Could not find player ID column. Available columns: {list(stats.columns)}"
        )

    if not name_col:
        raise RuntimeError(
            f"Could not find player name column. Available columns: {list(stats.columns)}"
        )

    if not position_col:
        raise RuntimeError(
            f"Could not find position column. Available columns: {list(stats.columns)}"
        )

    stats["player_id"] = stats[player_id_col].astype(str)
    stats["player_name"] = stats[name_col].fillna("").astype(str)
    stats["position"] = stats[position_col].fillna("").astype(str)

    if team_col:
        stats["team"] = stats[team_col].fillna("").astype(str)
    else:
        stats["team"] = ""

    numeric_columns = [
        "completions",
        "attempts",
        "passing_yards",
        "passing_tds",
        "interceptions",
        "sacks",
        "sack_yards",
        "passing_air_yards",
        "passing_first_downs",
        "passing_epa",
        "carries",
        "rushing_yards",
        "rushing_tds",
        "rushing_first_downs",
        "rushing_epa",
        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds",
        "receiving_air_yards",
        "receiving_first_downs",
        "receiving_epa",
        "fantasy_points",
        "fantasy_points_ppr",
        "special_teams_tds",
    ]

    for column in numeric_columns:
        if column not in stats.columns:
            stats[column] = 0

    players = (
        stats.groupby(
            [
                "player_id",
                "player_name",
                "position",
            ],
            dropna=False
        )
        .agg(
            team=(
                "team",
                lambda x: next(
                    (v for v in reversed(x.tolist()) if v),
                    ""
                )
            ),
            **{
                column: (column, "sum")
                for column in numeric_columns
            }
        )
        .reset_index()
    )

    return players


def calculate_red_zone_rushes(pbp):
    required = {
        "rush_attempt",
        "rusher_player_id",
        "posteam",
        "yardline_100",
    }

    missing = required - set(pbp.columns)

    if missing:
        raise RuntimeError(
            f"Play-by-play data is missing columns: {sorted(missing)}"
        )

    rz = pbp[
        (pbp["rush_attempt"] == 1)
        & (pbp["yardline_100"].notna())
        & (pbp["yardline_100"] <= 20)
        & (pbp["rusher_player_id"].notna())
        & (pbp["posteam"].notna())
    ].copy()

    rz["player_id"] = rz["rusher_player_id"].astype(str)
    rz["team"] = rz["posteam"].astype(str)

    player_rushes = (
        rz.groupby(
            ["player_id", "team"],
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "red_zone_rushes"
            }
        )
    )

    team_rushes = (
        rz.groupby(
            "team",
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "team_red_zone_rushes"
            }
        )
    )

    return player_rushes.merge(
        team_rushes,
        on="team",
        how="left"
    )


def calculate_red_zone_targets(pbp):
    required = {
        "pass_attempt",
        "receiver_player_id",
        "posteam",
        "yardline_100",
    }

    if not required.issubset(pbp.columns):
        return pd.DataFrame(
            columns=[
                "player_id",
                "red_zone_targets",
                "team_red_zone_targets",
            ]
        )

    rz = pbp[
        (pbp["pass_attempt"] == 1)
        & (pbp["yardline_100"].notna())
        & (pbp["yardline_100"] <= 20)
        & (pbp["receiver_player_id"].notna())
        & (pbp["posteam"].notna())
    ].copy()

    rz["player_id"] = rz["receiver_player_id"].astype(str)
    rz["team"] = rz["posteam"].astype(str)

    player_targets = (
        rz.groupby(
            ["player_id", "team"],
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "red_zone_targets"
            }
        )
    )

    team_targets = (
        rz.groupby(
            "team",
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "team_red_zone_targets"
            }
        )
    )

    return player_targets.merge(
        team_targets,
        on="team",
        how="left"
    )


def add_red_zone_data(players, pbp):

    # RED-ZONE RUSHES

    red_zone_rushes = calculate_red_zone_rushes(pbp)

    if not red_zone_rushes.empty:

        rush_totals = (
            red_zone_rushes
            .groupby(
                "player_id",
                as_index=False
            )
            .agg(
                red_zone_rushes=(
                    "red_zone_rushes",
                    "sum"
                ),
                team_red_zone_rushes=(
                    "team_red_zone_rushes",
                    "sum"
                ),
            )
        )

        rush_totals["red_zone_rush_share"] = (
            rush_totals["red_zone_rushes"]
            / rush_totals["team_red_zone_rushes"]
            * 100
        )

        players = players.merge(
            rush_totals[
                [
                    "player_id",
                    "red_zone_rushes",
                    "team_red_zone_rushes",
                    "red_zone_rush_share",
                ]
            ],
            on="player_id",
            how="left"
        )

    # RED-ZONE TARGETS

    red_zone_targets = calculate_red_zone_targets(pbp)

    if not red_zone_targets.empty:

        target_totals = (
            red_zone_targets
            .groupby(
                "player_id",
                as_index=False
            )
            .agg(
                red_zone_targets=(
                    "red_zone_targets",
                    "sum"
                ),
                team_red_zone_targets=(
                    "team_red_zone_targets",
                    "sum"
                ),
            )
        )

        target_totals["red_zone_target_share"] = (
            target_totals["red_zone_targets"]
            / target_totals["team_red_zone_targets"]
            * 100
        )

        players = players.merge(
            target_totals[
                [
                    "player_id",
                    "red_zone_targets",
                    "team_red_zone_targets",
                    "red_zone_target_share",
                ]
            ],
            on="player_id",
            how="left"
        )

    # DEFAULT VALUES

    defaults = {
        "red_zone_rushes": 0,
        "team_red_zone_rushes": 0,
        "red_zone_rush_share": 0,
        "red_zone_targets": 0,
        "team_red_zone_targets": 0,
        "red_zone_target_share": 0,
    }

    for column, default in defaults.items():

        if column not in players.columns:
            players[column] = default

        players[column] = (
            pd.to_numeric(
                players[column],
                errors="coerce"
            )
            .fillna(default)
        )

    return players


def convert_records(players):

    records = []

    for _, row in players.iterrows():

        record = {}

        for column in players.columns:

            value = row[column]

            if pd.isna(value):
                value = 0

            record[column] = clean_number(value)

        records.append(record)

    return records


def main():

    print(
        f"Building player data for {SEASON}..."
    )

    print(
        "Downloading player statistics..."
    )

    player_stats = load_player_stats()

    print(
        "Building player totals..."
    )

    players = build_player_totals(
        player_stats
    )

    print(
        f"Built {len(players)} players."
    )

    print(
        "Downloading play-by-play data..."
    )

    pbp = load_pbp()

    print(
        "Calculating red-zone data..."
    )

    players = add_red_zone_data(
        players,
        pbp
    )

    records = convert_records(
        players
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Done. Wrote {len(records)} players to {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
