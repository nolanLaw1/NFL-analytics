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

def get_column(df, possible_columns, required=True):
    for column in possible_columns:
        if column in df.columns:
            return column

    if required:
        raise RuntimeError(
            f"Could not find any of these columns: {possible_columns}"
        )

    return None


def safe_number(value):
    if pd.isna(value):
        return 0

    if isinstance(value, float) and value.is_integer():
        return int(value)

    return value


# ============================================================
# LOAD PLAYER DATA
# ============================================================

def load_player_stats():

    url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"stats_player/stats_player_week_{SEASON}.csv"
    )

    print(f"Downloading player statistics from {url}")

    stats = pd.read_csv(
        url,
        low_memory=False
    )

    print(f"Loaded {len(stats):,} player-stat rows.")

    return stats


# ============================================================
# LOAD PBP
# ============================================================

def load_pbp():

    url = (
        "https://github.com/nflverse/nflverse-data/releases/download/"
        f"pbp/play_by_play_{SEASON}.parquet"
    )

    print(f"Downloading play-by-play data from {url}")

    pbp = pd.read_parquet(url)

    print(f"Loaded {len(pbp):,} play-by-play rows.")

    return pbp


# ============================================================
# PREP PLAYER STATS
# ============================================================

def prepare_player_stats(stats):

    player_id_col = get_column(
        stats,
        [
            "player_id",
            "gsis_id"
        ]
    )

    name_col = get_column(
        stats,
        [
            "player_display_name",
            "player_name",
            "name"
        ]
    )

    position_col = get_column(
        stats,
        [
            "position"
        ]
    )

    team_col = get_column(
        stats,
        [
            "team",
            "recent_team"
        ],
        required=False
    )

    stats = stats.copy()

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


    # Only the statistics we actually use
    numeric_columns = [
        "attempts",
        "completions",
        "passing_yards",
        "passing_tds",
        "interceptions",

        "carries",
        "rushing_yards",
        "rushing_tds",

        "targets",
        "receptions",
        "receiving_yards",
        "receiving_tds"
    ]

    for column in numeric_columns:

        if column not in stats.columns:
            stats[column] = 0

        stats[column] = pd.to_numeric(
            stats[column],
            errors="coerce"
        ).fillna(0)


    return stats


# ============================================================
# BUILD PLAYER TOTALS
# ============================================================

def build_player_totals(stats):

    players = (
        stats
        .groupby(
            [
                "player_id",
                "player_name",
                "name",
                "position"
            ],
            dropna=False
        )
        .agg(
            team=(
                "team",
                lambda values: next(
                    (
                        str(value)
                        for value in reversed(values.tolist())
                        if str(value)
                        not in ["", "nan", "None"]
                    ),
                    ""
                )
            ),

            attempts=("attempts", "sum"),
            completions=("completions", "sum"),
            passing_yards=("passing_yards", "sum"),
            passing_tds=("passing_tds", "sum"),
            interceptions=("interceptions", "sum"),

            carries=("carries", "sum"),
            rushing_yards=("rushing_yards", "sum"),
            rushing_tds=("rushing_tds", "sum"),

            targets=("targets", "sum"),
            receptions=("receptions", "sum"),
            receiving_yards=("receiving_yards", "sum"),
            receiving_tds=("receiving_tds", "sum")
        )
        .reset_index()
    )


    # Completion percentage
    players["completion_pct"] = 0.0

    valid = players["attempts"] > 0

    players.loc[valid, "completion_pct"] = (
        players.loc[valid, "completions"]
        /
        players.loc[valid, "attempts"]
        * 100
    )


    # Yards per attempt
    players["yards_per_attempt"] = 0.0

    players.loc[valid, "yards_per_attempt"] = (
        players.loc[valid, "passing_yards"]
        /
        players.loc[valid, "attempts"]
    )


    # Team targets
    team_targets = (
        players
        .groupby(
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


    players["target_share"] = 0.0

    valid = players["team_targets"] > 0

    players.loc[valid, "target_share"] = (
        players.loc[valid, "targets"]
        /
        players.loc[valid, "team_targets"]
        * 100
    )


    players.drop(
        columns=["team_targets"],
        inplace=True
    )


    return players


# ============================================================
# PREP PBP
# ============================================================

def prepare_pbp(pbp):

    pbp = pbp.copy()


    # Regular season only
    if "season_type" in pbp.columns:

        pbp = pbp[
            pbp["season_type"] == "REG"
        ].copy()


    # Make sure needed columns exist
    defaults = {
        "pass_attempt": 0,
        "rush_attempt": 0,
        "receiver_player_id": "",
        "rusher_player_id": "",
        "passer_player_id": "",
        "posteam": "",
        "yardline_100": 999,
        "complete_pass": 0,
        "touchdown": 0,
        "receiving_yards": 0,
        "yards_gained": 0
    }


    for column, default in defaults.items():

        if column not in pbp.columns:
            pbp[column] = default


    # Numeric
    for column in [
        "pass_attempt",
        "rush_attempt",
        "yardline_100",
        "complete_pass",
        "touchdown",
        "receiving_yards",
        "yards_gained"
    ]:

        pbp[column] = pd.to_numeric(
            pbp[column],
            errors="coerce"
        ).fillna(0)


    # IDs / team
    for column in [
        "receiver_player_id",
        "rusher_player_id",
        "passer_player_id",
        "posteam"
    ]:

        pbp[column] = (
            pbp[column]
            .fillna("")
            .astype(str)
        )


    # Red zone = inside the 20
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
                "red_zone_rushes",
                "team_red_zone_rushes"
            ]
        )


    rz["player_id"] = (
        rz["rusher_player_id"]
    )

    rz["team"] = (
        rz["posteam"]
    )


    player = (
        rz
        .groupby(
            "player_id",
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "red_zone_rushes"
            }
        )
    )


    team = (
        rz
        .groupby(
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


    player_team = (
        rz[
            [
                "player_id",
                "team"
            ]
        ]
        .drop_duplicates(
            "player_id"
        )
    )


    result = player.merge(
        player_team,
        on="player_id",
        how="left"
    )


    result = result.merge(
        team,
        on="team",
        how="left"
    )


    return result[
        [
            "player_id",
            "red_zone_rushes",
            "team_red_zone_rushes"
        ]
    ]


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
                "red_zone_targets",
                "team_red_zone_targets"
            ]
        )


    rz["player_id"] = (
        rz["receiver_player_id"]
    )

    rz["team"] = (
        rz["posteam"]
    )


    player = (
        rz
        .groupby(
            "player_id",
            as_index=False
        )
        .size()
        .rename(
            columns={
                "size": "red_zone_targets"
            }
        )
    )


    team = (
        rz
        .groupby(
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


    player_team = (
        rz[
            [
                "player_id",
                "team"
            ]
        ]
        .drop_duplicates(
            "player_id"
        )
    )


    result = player.merge(
        player_team,
        on="player_id",
        how="left"
    )


    result = result.merge(
        team,
        on="team",
        how="left"
    )


    return result[
        [
            "player_id",
            "red_zone_targets",
            "team_red_zone_targets"
        ]
    ]


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
    )


    # Reception
    rz["rz_reception"] = (
        rz["complete_pass"] == 1
    ).astype(int)


    # Receiving yards
    rz["rz_receiving_yards"] = (
        rz["receiving_yards"]
    )


    # Receiving TD
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
        rz
        .groupby(
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
    )


    rz["rz_pass_td"] = (
        rz["touchdown"] == 1
    ).astype(int)


    result = (
        rz
        .groupby(
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

    rushes = calculate_red_zone_rushes(
        pbp
    )

    players = players.merge(
        rushes,
        on="player_id",
        how="left"
    )


    targets = calculate_red_zone_targets(
        pbp
    )

    players = players.merge(
        targets,
        on="player_id",
        how="left"
    )


    receiving = calculate_red_zone_receiving(
        pbp
    )

    players = players.merge(
        receiving,
        on="player_id",
        how="left"
    )


    passing = calculate_red_zone_passing(
        pbp
    )

    players = players.merge(
        passing,
        on="player_id",
        how="left"
    )


    # Defaults
    rz_columns = [
        "red_zone_rushes",
        "team_red_zone_rushes",

        "red_zone_targets",
        "team_red_zone_targets",

        "red_zone_receptions",
        "red_zone_receiving_yards",
        "red_zone_receiving_touchdowns",

        "red_zone_pass_attempts",
        "red_zone_pass_touchdowns"
    ]


    for column in rz_columns:

        if column not in players.columns:
            players[column] = 0

        players[column] = pd.to_numeric(
            players[column],
            errors="coerce"
        ).fillna(0)


    # Website compatibility
    players["red_zone_carries"] = (
        players["red_zone_rushes"]
    )


    # RZ target share
    players["red_zone_target_share"] = 0.0

    valid = (
        players["team_red_zone_targets"] > 0
    )

    players.loc[valid, "red_zone_target_share"] = (
        players.loc[valid, "red_zone_targets"]
        /
        players.loc[valid, "team_red_zone_targets"]
        * 100
    )


    # RZ rush share
    players["red_zone_rush_share"] = 0.0

    valid = (
        players["team_red_zone_rushes"] > 0
    )

    players.loc[valid, "red_zone_rush_share"] = (
        players.loc[valid, "red_zone_rushes"]
        /
        players.loc[valid, "team_red_zone_rushes"]
        * 100
    )


    return players


# ============================================================
# WEEKLY USAGE
# ============================================================

def build_weekly_usage(stats, pbp):

    weekly = []


    # --------------------------------------------------------
    # REGULAR SEASON
    # --------------------------------------------------------

    if "season_type" in stats.columns:

        weekly_stats = stats[
            stats["season_type"] == "REG"
        ].copy()

    else:

        weekly_stats = stats.copy()


    if "week" not in weekly_stats.columns:

        return weekly


    # --------------------------------------------------------
    # IDENTIFY PLAYER ID
    # --------------------------------------------------------

    player_id_col = get_column(
        weekly_stats,
        [
            "player_id",
            "gsis_id"
        ]
    )


    # --------------------------------------------------------
    # IDENTIFY TEAM
    #
    # This is the important fix.
    # 2026 uses "team" in the file we're downloading.
    # --------------------------------------------------------

    team_col = get_column(
        weekly_stats,
        [
            "team",
            "recent_team"
        ]
    )


    weekly_stats["player_id"] = (
        weekly_stats[player_id_col]
        .fillna("")
        .astype(str)
    )


    weekly_stats["team"] = (
        weekly_stats[team_col]
        .fillna("")
        .astype(str)
    )


    # --------------------------------------------------------
    # REQUIRED WEEKLY STATS
    # --------------------------------------------------------

    for column in [
        "targets",
        "carries",
        "receptions"
    ]:

        if column not in weekly_stats.columns:
            weekly_stats[column] = 0

        weekly_stats[column] = pd.to_numeric(
            weekly_stats[column],
            errors="coerce"
        ).fillna(0)


    # --------------------------------------------------------
    # TEAM TARGETS PER WEEK
    # --------------------------------------------------------

    team_week_targets = (
        weekly_stats
        .groupby(
            [
                "team",
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
            "team",
            "week"
        ],
        how="left"
    )


    # --------------------------------------------------------
    # TARGET SHARE
    # --------------------------------------------------------

    weekly_stats["target_share"] = 0.0

    valid = (
        weekly_stats["team_targets"] > 0
    )


    weekly_stats.loc[valid, "target_share"] = (
        weekly_stats.loc[valid, "targets"]
        /
        weekly_stats.loc[valid, "team_targets"]
        * 100
    )


    # --------------------------------------------------------
    # RZ TARGETS BY PLAYER/WEEK
    # --------------------------------------------------------

    rz_targets = (
        pbp[
            (pbp["red_zone"])
            &
            (pbp["pass_attempt"] == 1)
            &
            (pbp["receiver_player_id"] != "")
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


    # --------------------------------------------------------
    # RZ CARRIES BY PLAYER/WEEK
    # --------------------------------------------------------

    rz_carries = (
        pbp[
            (pbp["red_zone"])
            &
            (pbp["rush_attempt"] == 1)
            &
            (pbp["rusher_player_id"] != "")
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
    # BUILD WEEKLY ROWS
    # --------------------------------------------------------

    for _, row in weekly_stats.iterrows():

        player_id = str(
            row["player_id"]
        )

        week = int(
            row["week"]
        )


        target_match = rz_targets[
            (rz_targets["player_id"] == player_id)
            &
            (rz_targets["week"] == week)
        ]


        carry_match = rz_carries[
            (rz_carries["player_id"] == player_id)
            &
            (rz_carries["week"] == week)
        ]


        red_zone_targets = 0

        if not target_match.empty:

            red_zone_targets = int(
                target_match.iloc[0]["red_zone_targets"]
            )


        red_zone_carries = 0

        if not carry_match.empty:

            red_zone_carries = int(
                carry_match.iloc[0]["red_zone_carries"]
            )


        team = str(
            row["team"]
        )


        # Team RZ targets for that week
        team_rz_targets = int(
            pbp[
                (pbp["red_zone"])
                &
                (pbp["pass_attempt"] == 1)
                &
                (pbp["posteam"] == team)
                &
                (pbp["week"] == week)
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


    # Load data
    stats = load_player_stats()

    pbp = load_pbp()


    # Prepare
    stats = prepare_player_stats(
        stats
    )

    pbp = prepare_pbp(
        pbp
    )


    # Player totals
    players = build_player_totals(
        stats
    )


    # Red-zone data
    players = add_red_zone_data(
        players,
        pbp
    )


    # Weekly data
    weekly = build_weekly_usage(
        stats,
        pbp
    )


    # --------------------------------------------------------
    # ATTACH WEEKLY DATA
    # --------------------------------------------------------

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


    players["weekly"] = players[
        "player_id"
    ].map(
        lambda player_id:
            weekly_by_player.get(
                str(player_id),
                []
            )
    )


    # --------------------------------------------------------
    # REMOVE INTERNAL FIELDS
    # --------------------------------------------------------

    for column in [
        "team_red_zone_rushes",
        "team_red_zone_targets"
    ]:

        if column in players.columns:

            players.drop(
                columns=[column],
                inplace=True
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


        players[column] = pd.to_numeric(
            players[column],
            errors="coerce"
        ).fillna(0)


        players[column] = players[column].apply(
            safe_number
        )


    # --------------------------------------------------------
    # BUILD JSON
    # --------------------------------------------------------

    output = []


    for _, row in players.iterrows():

        player = {}


        for column in players.columns:

            value = row[column]


            if column == "weekly":

                player[column] = value
                continue


            player[column] = safe_number(
                value
            )


        output.append(
            player
        )


    result = {
        "season": SEASON,
        "players": output
    }


    OUTPUT_FILE.write_text(
        json.dumps(
            result,
            indent=2,
            allow_nan=False
        )
    )


    print()
    print("========================================")
    print("PLAYER DATA BUILD COMPLETE")
    print("========================================")
    print(f"Season: {SEASON}")
    print(f"Players: {len(output):,}")
    print("RZ carries: YES")
    print("RZ receptions: YES")
    print("RZ receiving yards: YES")
    print("RZ receiving TDs: YES")
    print("RZ pass attempts: YES")
    print("RZ pass TDs: YES")
    print(f"Output: {OUTPUT_FILE}")
    print("========================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
