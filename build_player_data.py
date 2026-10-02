import sys
import json
from pathlib import Path

import pandas as pd


SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2025

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "players.json"

PLAYER_STATS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "player_stats/player_stats.csv"
)

PBP_URL = (
    f"https://github.com/nflverse/nflverse-data/releases/download/"
    f"pbp/play_by_play_{SEASON}.parquet"
)


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
    df = pd.read_csv(PLAYER_STATS_URL, low_memory=False)

    if "season" in df.columns:
        df = df[df["season"] == SEASON].copy()

    return df


def load_pbp():
    return pd.read_parquet(PBP_URL)


def build_player_totals(stats):
    player_id_col = first_existing(
        stats,
        ["player_id", "player_player_id"]
    )

    name_col = first_existing(
        stats,
        ["player_name", "name"]
    )

    position_col = first_existing(
        stats,
        ["position"]
    )

    team_col = first_existing(
        stats,
        ["recent_team", "team", "posteam"]
    )

    if not player_id_col or not name_col or not position_col:
        raise RuntimeError(
            "Could not find required player columns in nflverse player_stats.csv"
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

    group_columns = [
        "player_id",
        "player_name",
        "position",
    ]

    players = (
        stats.groupby(group_columns, dropna=False)
        .agg(
            team=("team", lambda x: next(
                (v for v in reversed(x.tolist()) if v),
                ""
            )),
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

    # Individual player red-zone rush attempts
    player_rushes = (
        rz.groupby(["player_id", "team"])
        .size()
        .reset_index(name="red_zone_rushes")
    )

    # Total team red-zone rush attempts
    team_rushes = (
        rz.groupby("team")
        .size()
        .reset_index(name="team_red_zone_rushes")
    )

    result = player_rushes.merge(
        team_rushes,
        on="team",
        how="left"
    )

    # Player's percentage of his team's red-zone rush attempts
    result["red_zone_rush_share"] = (
        result["red_zone_rushes"]
        / result["team_red_zone_rushes"].replace(0, pd.NA)
        * 100
    )

    return result


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
                "red_zone_target_share",
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
        rz.groupby(["player_id", "team"])
        .size()
        .reset_index(name="red_zone_targets")
    )

    team_targets = (
        rz.groupby("team")
        .size()
        .reset_index(name="team_red_zone_targets")
    )

    result = player_targets.merge(
        team_targets,
        on="team",
        how="left"
    )

    result["red_zone_target_share"] = (
        result["red_zone_targets"]
        / result["team_red_zone_targets"].replace(0, pd.NA)
        * 100
    )

    return result


def add_red_zone_data(players, pbp):
    red_zone_rushes = calculate_red_zone_rushes(pbp)

    if not red_zone_rushes.empty:

        # Aggregate across teams for players who changed teams.
        rush_totals = (
            red_zone_rushes
            .groupby("player_id", as_index=False)
            .agg(
                red_zone_rushes=("red_zone_rushes", "sum"),
                team_red_zone_rushes=("team_red_zone_rushes", "sum"),
            )
        )

        rush_totals["red_zone_rush_share"] = (
            rush_totals["red_zone_rushes"]
            / rush_totals["team_red_zone_rushes"].replace(0, pd.NA)
            * 100
        )

        players = players.merge(
            rush_totals,
            on="player_id",
            how="left"
        )

    red_zone_targets = calculate_red_zone_targets(pbp)

    if not red_zone_targets.empty:

        target_totals = (
            red_zone_targets
            .groupby("player_id", as_index=False)
            .agg(
                red_zone_targets=("red_zone_targets", "sum"),
                team_red_zone_targets=("team_red_zone_targets", "sum"),
            )
        )

        target_totals["red_zone_target_share"] = (
            target_totals["red_zone_targets"]
            / target_totals["team_red_zone_targets"].replace(0, pd.NA)
            * 100
        )

        players = players.merge(
            target_totals,
            on="player_id",
            how="left"
        )

    # Make sure every player has the fields expected by the site.
    default_columns = {
        "red_zone_rushes": 0,
        "team_red_zone_rushes": 0,
        "red_zone_rush_share": 0,
        "red_zone_targets": 0,
        "team_red_zone_targets": 0,
        "red_zone_target_share": 0,
    }

    for column, default in default_columns.items():
        if column not in players.columns:
            players[column] = default

        players[column] = players[column].fillna(default)

    return players


def convert_records(players):
    records = []

    for _, row in players.iterrows():
        record = {}

        for column in players.columns:
            value = row[column]

            if pd.isna(value):
                value = 0

            value = clean_number(value)

            record[column] = value

        records.append(record)

    return records


def main():
    print(f"Building player data for {SEASON}...")

    print("Downloading player statistics...")
    player_stats = load_player_stats()

    print("Building player totals...")
    players = build_player_totals(player_stats)

    print("Downloading play-by-play data...")
    pbp = load_pbp()

    print("Calculating red-zone data...")
    players = add_red_zone_data(players, pbp)

    records = convert_records(players)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
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
