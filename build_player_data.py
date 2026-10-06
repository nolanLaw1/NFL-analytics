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
    if isinstance(value, (list, dict)):
        return value

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
            "player_display_name",
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

    stats["player_id"] = (
        stats[player_id_col]
        .fillna("")
        .astype(str)
    )

    stats["player_name"] = (
        stats[name_col]
        .fillna("")
        .astype(str)
    )

    stats["name"] = stats["player_name"]

    stats["position"] = (
        stats[position_col]
        .fillna("")
        .astype(str)
    )

    if team_col:

        stats["team"] = (
            stats[team_col]
            .fillna("")
            .astype(str)
        )

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

        stats[column] = (
            pd.to_numeric(
                stats[column],
                errors="coerce"
            )
            .fillna(0)
        )

    players = (
        stats.groupby(
            [
                "player_id",
                "player_name",
                "name",
                "position",
            ],
            dropna=False
        )
        .agg(
            team=(
                "team",
                lambda x: next(
                    (
                        v for v in reversed(x.tolist())
                        if v and v != "nan"
                    ),
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

    # =========================================================
    # NORMAL TARGET SHARE
    # =========================================================

    team_targets = (
        players.groupby(
            "team",
            as_index=False
        )["targets"]
        .sum()
        .rename(
            columns={
                "targets": "team_targets"
            }
        )
    )

    players = players.merge(
        team_targets,
        on="team",
        how="left"
    )

    players["team_targets"] = (
        pd.to_numeric(
            players["team_targets"],
            errors="coerce"
        )
        .fillna(0)
    )

    players["target_share"] = 0.0

    valid_targets = players["team_targets"] > 0

    players.loc[
        valid_targets,
        "target_share"
    ] = (
        players.loc[
            valid_targets,
            "targets"
        ]
        /
        players.loc[
            valid_targets,
            "team_targets"
        ]
        * 100
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
        &
        (pbp["yardline_100"].notna())
        &
        (pbp["yardline_100"] <= 20)
        &
        (pbp["rusher_player_id"].notna())
        &
        (pbp["posteam"].notna())
    ].copy()

    if "season_type" in rz.columns:
        rz = rz[
            rz["season_type"] == "REG"
        ]

    rz["player_id"] = (
        rz["rusher_player_id"]
        .astype(str)
    )

    rz["team"] = (
        rz["posteam"]
        .astype(str)
    )

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
        &
        (pbp["yardline_100"].notna())
        &
        (pbp["yardline_100"] <= 20)
        &
        (pbp["receiver_player_id"].notna())
        &
        (pbp["posteam"].notna())
    ].copy()

    if "season_type" in rz.columns:
        rz = rz[
            rz["season_type"] == "REG"
        ]

    rz["player_id"] = (
        rz["receiver_player_id"]
        .astype(str)
    )

    rz["team"] = (
        rz["posteam"]
        .astype(str)
    )

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


# =============================================================
# RED-ZONE RECEIVING
# =============================================================

def calculate_red_zone_receiving(pbp):

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
                "red_zone_receptions",
                "red_zone_receiving_yards",
                "red_zone_receiving_touchdowns",
            ]
        )

    rz = pbp[
        (pbp["pass_attempt"] == 1)
        &
        (pbp["yardline_100"].notna())
        &
        (pbp["yardline_100"] <= 20)
        &
        (pbp["receiver_player_id"].notna())
        &
        (pbp["posteam"].notna())
    ].copy()

    if "season_type" in rz.columns:
        rz = rz[
            rz["season_type"] == "REG"
        ]

    rz["player_id"] = (
        rz["receiver_player_id"]
        .astype(str)
    )

    # ---------------------------------------------------------
    # RECEPTIONS
    # ---------------------------------------------------------

    if "complete_pass" in rz.columns:

        rz["rz_reception"] = (
            pd.to_numeric(
                rz["complete_pass"],
                errors="coerce"
            )
            .fillna(0)
        )

    elif "pass_completed" in rz.columns:

        rz["rz_re
