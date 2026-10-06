import sys
import json
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SEASON = int(sys.argv[1]) if len(sys.argv) > 1 else 2026

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = DATA_DIR / "players.json"


# ============================================================
# HELPERS
# ============================================================

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


# ============================================================
# LOAD PLAYER STATS
# ============================================================

def load_player_stats():

    url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"stats_player/stats_player_week_{SEASON}.csv"
    )

    print(
        f"Downloading player statistics from {url}"
    )

    df = pd.read_csv(
        url,
        low_memory=False
    )

    print(
        f"Loaded {len(df):,} player-stat rows."
    )

    return df


# ============================================================
# LOAD PBP
# ============================================================

def load_pbp():

    url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"pbp/play_by_play_{SEASON}.parquet"
    )

    print(
        f"Downloading play-by-play data from {url}"
    )

    pbp = pd.read_parquet(url)

    print(
        f"Loaded {len(pbp):,} play-by-play rows."
    )

    return pbp


# ============================================================
# PLAYER TOTALS
# ============================================================

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
            "Could not find player ID column."
        )

    if not name_col:

        raise RuntimeError(
            "Could not find player name column."
        )

    if not position_col:

        raise RuntimeError(
            "Could not find position column."
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
                        v
                        for v in reversed(x.tolist())
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


    # ========================================================
    # NORMAL TARGET SHARE
    # ========================================================

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


    valid_targets = (
        players["team_targets"] > 0
    )


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


# ============================================================
# PREP PBP
# ============================================================

def prepare_pbp(pbp):

    # Regular season only
    if "season_type" in pbp.columns:

        pbp = pbp[
            pbp["season_type"] == "REG"
        ].copy()

    else:

        pbp = pbp.copy()


    # Make sure required columns exist
    required_defaults = {

        "pass_attempt": 0,
        "rush_attempt": 0,

        "receiver_player_id": "",
        "rusher_player_id": "",

        "posteam": "",

        "yardline_100": None,

        "complete_pass": 0,
        "touchdown": 0,

        "receiving_yards": 0,
        "yards_gained": 0,

        "passer_player_id": "",
    }


    for column, default in required_defaults.items():

        if column not in pbp.columns:

            pbp[column] = default


    # Numeric columns

    for column in [

        "pass_attempt",
        "rush_attempt",
        "complete_pass",
        "touchdown",
        "yardline_100",
        "receiving_yards",
        "yards_gained",

    ]:

        pbp[column] = pd.to_numeric(
            pbp[column],
            errors="coerce"
        ).fillna(0)


    # IDs

    for column in [

        "receiver_player_id",
        "rusher_player_id",
        "passer_player_id",

    ]:

        pbp[column] = (
            pbp[column]
            .fillna("")
            .astype(str)
        )


    pbp["posteam"] = (
        pbp["posteam"]
        .fillna("")
        .astype(str)
    )


    # Red zone
    pbp["red_zone"] = (
        pbp["yardline_100"] <= 20
    )


    return pbp


# ============================================================
# RED-ZONE RUSHES
# ============================================================

def calculate_red_zone_rushes(pbp):

    rz = pbp[
        (pbp["rush_attempt"] == 1)
        &
        (pbp["red_zone"])
        &
        (pbp["rusher_player_id"] != "")
        &
        (pbp["posteam"] != "")
    ].copy()


    if rz.empty:

        return pd.DataFrame(
            columns=[
                "player_id",
                "team",
                "red_zone_rushes",
                "team_red_zone_rushes"
            ]
        )


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
            [
                "player_id",
                "team"
            ],
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


# ============================================================
# RED-ZONE TARGETS
# ============================================================

def calculate_red_zone_targets(pbp):

    rz = pbp[
        (pbp["pass_attempt"] == 1)
        &
        (pbp["red_zone"])
        &
        (pbp["receiver_player_id"] != "")
        &
        (pbp["posteam"] != "")
    ].copy()


    if rz.empty:

        return pd.DataFrame(
            columns=[
                "player_id",
                "team",
                "red_zone_targets",
                "team_red_zone_targets"
            ]
        )


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
            [
                "player_id",
                "team"
            ],
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


# ============================================================
# RED-ZONE RECEIVING
# ============================================================

def calculate_red_zone_receiving(pbp):

    rz = pbp[
        (pbp["pass_attempt"] == 1)
        &
        (pbp["red_zone"])
        &
        (pbp["receiver_player_id"] != "")
        &
        (pbp["posteam"] != "")
    ].copy()


    if rz.empty:

        return pd.DataFrame(
            columns=[
                "player_id",
                "red_zone_receptions",
                "red_zone_receiving_yards",
                "red_zone_receiving_touchdowns"
            ]
        )


    rz["player_id"] = (
        rz["receiver_player_id"]
        .astype(str)
    )


    # --------------------------------------------------------
    # RECEPTIONS
    # --------------------------------------------------------

    rz["rz_reception"] = (
        rz["complete_pass"]
        .eq(1)
        .astype(int)
    )


    # --------------------------------------------------------
    # RECEIVING YARDS
    # --------------------------------------------------------

    if "receiving_yards" in rz.columns:

        rz["rz_receiving_yards"] = (
            pd.to_numeric(
                rz["receiving_yards"],
                errors="coerce"
            )
            .fillna(0)
        )

    else:

        rz["rz_receiving_yards"] = (
            pd.to_numeric(
                rz["yards_gained"],
                errors="coerce"
            )
            .fillna(0)
        )


    # Only completed passes count as receiving yards
    rz["rz_receiving_yards"] = (
        rz["rz_receiving_yards"]
        * rz["rz_reception"]
    )


    # --------------------------------------------------------
    # RECEIVING TOUCHDOWNS
    # --------------------------------------------------------

    rz["rz_receiving_td"] = (
        (
            rz["complete_pass"] == 1
        )
        &
        (
            rz["touchdown"] == 1
        )
    ).astype(int)


    result = (
        rz.groupby(
            "player_id",
            as_index=False
        )
        .agg(
            red_zone_receptions=(
                "rz_reception",
                "sum"
            ),
            red_zone_receiving_yards=(
                "rz_receiving_yards",
                "sum"
            ),
            red_zone_receiving_touchdowns=(
                "rz_receiving_td",
                "sum"
            )
        )
    )


    return result


# ============================================================
# RED-ZONE PASSING
# ============================================================

def calculate_red_zone_passing(pbp):

    rz = pbp[
        (pbp["pass_attempt"] == 1)
        &
        (pbp["red_zone"])
        &
        (pbp["passer_player_id"] != "")
    ].copy()


    if rz.empty:

        return pd.DataFrame(
            columns=[
                "player_id",
                "red_zone_pass_attempts",
                "red_zone_pass_touchdowns"
            ]
        )


    rz["player_id"] = (
        rz["passer_player_id"]
        .astype(str)
    )


    rz["rz_pass_td"] = (
        rz["touchdown"] == 1
    ).astype(int)


    result = (
        rz.groupby(
            "player_id",
            as_index=False
        )
        .agg(
            red_zone_pass_attempts=(
                "pass_attempt",
                "sum"
            ),
            red_zone_pass_touchdowns=(
                "rz_pass_td",
                "sum"
            )
        )
    )


    return result


# ============================================================
# ADD RED-ZONE DATA
# ============================================================

def add_red_zone_data(players, pbp):

    # --------------------------------------------------------
    # RUSHES
    # --------------------------------------------------------

    rushes = calculate_red_zone_rushes(
        pbp
    )


    players = players.merge(
        rushes[
            [
                "player_id",
                "red_zone_rushes",
                "team_red_zone_rushes"
            ]
        ],
        on="player_id",
        how="left"
    )


    # --------------------------------------------------------
    # TARGETS
    # --------------------------------------------------------

    targets = calculate_red_zone_targets(
        pbp
    )


    players = players.merge(
        targets[
            [
                "player_id",
                "red_zone_targets",
                "team_red_zone_targets"
            ]
        ],
        on="player_id",
        how="left"
    )


    # --------------------------------------------------------
    # RECEIVING
    # --------------------------------------------------------

    receiving = calculate_red_zone_receiving(
        pbp
    )


    players = players.merge(
        receiving,
        on="player_id",
        how="left"
    )


    # --------------------------------------------------------
    # PASSING
    # --------------------------------------------------------

    passing = calculate_red_zone_passing(
        pbp
    )


    players = players.merge(
        passing,
        on="player_id",
        how="left"
    )


    # --------------------------------------------------------
    # DEFAULTS
    # --------------------------------------------------------

    new_columns = [

        "red_zone_rushes",
        "team_red_zone_rushes",

        "red_zone_targets",
        "team_red_zone_targets",

        "red_zone_receptions",
        "red_zone_receiving_yards",
        "red_zone_receiving_touchdowns",

        "red_zone_pass_attempts",
        "red_zone_pass_touchdowns",

    ]


    for column in new_columns:

        if column not in players.columns:

            players[column] = 0

        players[column] = (
            pd.to_numeric(
                players[column],
                errors="coerce"
            )
            .fillna(0)
        )


    # --------------------------------------------------------
    # RZ CARRIES
    #
    # Keep red_zone_rushes for compatibility,
    # but expose red_zone_carries for the website.
    # --------------------------------------------------------

    players["red_zone_carries"] = (
        players["red_zone_rushes"]
    )


    # --------------------------------------------------------
    # TEAM RZ TARGET SHARE
    # --------------------------------------------------------

    players["red_zone_target_share"] = 0.0


    valid_targets = (
        players["team_red_zone_targets"] > 0
    )


    players.loc[
        valid_targets,
        "red_zone_target_share"
    ] = (
        players.loc[
            valid_targets,
            "red_zone_targets"
        ]
        /
        players.loc[
            valid_targets,
            "team_red_zone_targets"
        ]
        * 100
    )


    # --------------------------------------------------------
    # TEAM RZ RUSH SHARE
    # --------------------------------------------------------

    players["red_zone_rush_share"] = 0.0


    valid_rushes = (
        players["team_red_zone_rushes"] > 0
    )


    players.loc[
        valid_rushes,
        "red_zone_rush_share"
    ] = (
        players.loc[
            valid_rushes,
            "red_zone_rushes"
        ]
        /
        players.loc[
            valid_rushes,
            "team_red_zone_rushes"
        ]
        * 100
    )


    return players


# ============================================================
# WEEKLY USAGE
# ============================================================

def build_weekly_usage(stats, pbp):

    weekly = []


    # Regular season only
    if "season_type" in stats.columns:

        weekly_stats = stats[
            stats["season_type"] == "REG"
        ].copy()

    else:

        weekly_stats = stats.copy()


    if "week" not in weekly_stats.columns:

        return weekly


    # --------------------------------------------------------
    # NORMALIZE IDS
    # --------------------------------------------------------

    id_col = first_existing(
        weekly_stats,
        [
            "player_id",
            "player_player_id",
            "gsis_id"
        ]
    )


    if not id_col:

        return weekly


    weekly_stats["player_id"] = (
        weekly_stats[id_col]
        .fillna("")
        .astype(str)
    )


    # --------------------------------------------------------
    # NUMERIC FIELDS
    # --------------------------------------------------------

    for column in [

        "targets",
        "carries",
        "receptions"

    ]:

        if column not in weekly_stats.columns:

            weekly_stats[column] = 0

        weekly_stats[column] = (
            pd.to_numeric(
                weekly_stats[column],
                errors="coerce"
            )
            .fillna(0)
        )


    # --------------------------------------------------------
    # TEAM TARGETS BY WEEK
    # --------------------------------------------------------

    team_week_targets = (
        weekly_stats
        .groupby(
            [
                "recent_team",
                "week"
            ],
            as_index=False
        )["targets"]
        .sum()
        .rename(
            columns={
                "targets": "team_targets"
            }
        )
    )


    weekly_stats = weekly_stats.merge(
        team_week_targets,
        on=[
            "recent_team",
            "week"
        ],
        how="left"
    )


    weekly_stats["target_share"] = 0.0


    valid = (
        weekly_stats["team_targets"] > 0
    )


    weekly_stats.loc[
        valid,
        "target_share"
    ] = (
        weekly_stats.loc[
            valid,
            "targets"
        ]
        /
        weekly_stats.loc[
            valid,
            "team_targets"
        ]
        * 100
    )


    # --------------------------------------------------------
    # RZ PBP WEEKLY
    # --------------------------------------------------------

    rz_pbp = pbp[
        pbp["red_zone"]
    ].copy()


    # RZ targets
    rz_targets = (
        rz_pbp[
            (rz_pbp["pass_attempt"] == 1)
            &
            (rz_pbp["receiver_player_id"] != "")
        ]
        .groupby(
            [
                "receiver_player_id",
                "week"
            ],
            as_index=False
        )
        .size()
        .rename(
            columns={
                "receiver_player_id": "player_id",
                "size": "red_zone_targets"
            }
        )
    )


    # RZ carries
    rz_carries = (
        rz_pbp[
            (rz_pbp["rush_attempt"] == 1)
            &
            (rz_pbp["rusher_player_id"] != "")
        ]
        .groupby(
            [
                "rusher_player_id",
                "week"
            ],
            as_index=False
        )
        .size()
        .rename(
            columns={
                "rusher_player_id": "player_id",
                "size": "red_zone_carries"
            }
        )
    )


    # --------------------------------------------------------
    # BUILD PLAYER/WEEK ROWS
    # --------------------------------------------------------

    for _, row in weekly_stats.iterrows():

        player_id = str(
            row["player_id"]
        )

        week = int(
            row["week"]
        )


        target_row = rz_targets[
            (
                rz_targets["player_id"]
                == player_id
            )
            &
            (
                rz_targets["week"]
                == week
            )
        ]


        carry_row = rz_carries[
            (
                rz_carries["player_id"]
                == player_id
            )
            &
            (
                rz_carries["week"]
                == week
            )
        ]


        red_zone_targets = (
            int(
                target_row.iloc[0]["red_zone_targets"]
            )
            if not target_row.empty
            else 0
        )


        red_zone_carries = (
            int(
                carry_row.iloc[0]["red_zone_carries"]
            )
            if not carry_row.empty
            else 0
        )


        # Team RZ targets
        team = str(
            row.get(
                "recent_team",
                ""
            )
        )


        team_rz_targets = 0


        if team:

            team_rz_targets = int(
                rz_pbp[
                    (
                        rz_pbp["posteam"]
                        == team
                    )
                    &
                    (
                        rz_pbp["week"]
                        == week
                    )
                    &
                    (
                        rz_pbp["pass_attempt"]
                        == 1
                    )
                ].shape[0]
            )


        red_zone_target_share = 0.0


        if team_rz_targets > 0:

            red_zone_target_share = (
                red_zone_targets
                /
                team_rz_targets
                * 100
            )


        weekly.append(
            {
                "player_id": player_id,

                "week": week,

                "targets": int(
                    row["targets"]
                ),

                "target_share": round(
                    float(
                        row["target_share"]
                    ),
                    1
                ),

                "carries": int(
                    row["carries"]
                ),

                "receptions": int(
                    row["receptions"]
                ),

                "red_zone_targets": (
                    red_zone_targets
                ),

                "red_zone_target_share": round(
                    float(
                        red_zone_target_share
                    ),
                    1
                ),

                "red_zone_carries": (
                    red_zone_carries
                )
            }
        )


    return weekly


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        f"Building player data for {SEASON}..."
    )


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    stats = load_player_stats()

    pbp = load_pbp()


    # --------------------------------------------------------
    # PREP PBP
    # --------------------------------------------------------

    pbp = prepare_pbp(
        pbp
    )


    # --------------------------------------------------------
    # BUILD TOTALS
    # --------------------------------------------------------

    players = build_player_totals(
        stats
    )


    # --------------------------------------------------------
    # RED-ZONE DATA
    # --------------------------------------------------------

    players = add_red_zone_data(
        players,
        pbp
    )


    # --------------------------------------------------------
    # WEEKLY DATA
    # --------------------------------------------------------

    weekly = build_weekly_usage(
        stats,
        pbp
    )


    # Convert weekly data into player lookup

    weekly_by_player = {}


    for row in weekly:

        player_id = row["player_id"]

        if player_id not in weekly_by_player:

            weekly_by_player[player_id] = []

        weekly_by_player[player_id].append(
            {
                "week": row["week"],
                "targets": row["targets"],
                "target_share": row["target_share"],
                "carries": row["carries"],
                "receptions": row["receptions"],
                "red_zone_targets": row[
                    "red_zone_targets"
                ],
                "red_zone_target_share": row[
                    "red_zone_target_share"
                ],
                "red_zone_carries": row[
                    "red_zone_carries"
                ]
            }
        )


    # --------------------------------------------------------
    # ATTACH WEEKLY
    # --------------------------------------------------------

    players["weekly"] = players[
        "player_id"
    ].map(
        lambda x: weekly_by_player.get(
            str(x),
            []
        )
    )


    # --------------------------------------------------------
    # REMOVE INTERNAL COLUMNS
    # --------------------------------------------------------

    if "team_targets" in players.columns:

        players = players.drop(
            columns=[
                "team_targets"
            ]
        )


    # --------------------------------------------------------
    # CLEAN NUMBERS
    # --------------------------------------------------------

    for column in players.columns:

        if column == "weekly":
            continue

        if column in [
            "player_id",
            "player_name",
            "name",
            "position",
            "team"
        ]:
            continue

        players[column] = (
            pd.to_numeric(
                players[column],
                errors="coerce"
            )
            .fillna(0)
        )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output = []


    for _, row in players.iterrows():

        item = {}


        for column in players.columns:

            value = row[column]


            if column == "weekly":

                item[column] = value

                continue


            if pd.isna(value):

                item[column] = 0

                continue


            item[column] = clean_number(
                value
            )


        output.append(
            item
        )


    OUTPUT_FILE.write_text(
        json.dumps(
            {
                "season": SEASON,
                "players": output
            },
            indent=2,
            allow_nan=False
        )
    )


    print()
    print("========================================")
    print("PLAYER DATA BUILD COMPLETE")
    print("========================================")
    print(
        f"Season: {SEASON}"
    )
    print(
        f"Players: {len(output):,}"
    )
    print(
        "Added RZ carries: YES"
    )
    print(
        "Added RZ receptions: YES"
    )
    print(
        "Added RZ receiving yards: YES"
    )
    print(
        "Added RZ receiving TDs: YES"
    )
    print(
        "Added RZ pass attempts: YES"
    )
    print(
        "Added RZ pass TDs: YES"
    )
    print(
        f"Wrote: {OUTPUT_FILE}"
    )
    print("========================================")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()
