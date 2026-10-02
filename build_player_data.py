import sys
import json
from pathlib import Path
import pandas as pd

season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025

PBP_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    f"pbp/play_by_play_{season}.parquet"
)

PLAYERS_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/"
    "players/players.csv"
)

SKILL = {"WR", "RB", "TE"}

print(f"Loading nflverse play-by-play for {season}...")
df = pd.read_parquet(PBP_URL)

df = df[df["season_type"].eq("REG")].copy()

print("Loading nflverse player positions...")
players_df = pd.read_csv(PLAYERS_URL)

# nflverse uses gsis_id as the primary player ID.
players_df = players_df[
    ["gsis_id", "display_name", "position"]
].copy()

players_df["gsis_id"] = players_df["gsis_id"].astype(str)

position_map = dict(
    zip(players_df["gsis_id"], players_df["position"])
)

name_map = dict(
    zip(players_df["gsis_id"], players_df["display_name"])
)

players = {}


def ensure(pid, name="", position="", team=""):
    pid = str(pid)

    if pid not in players:
        players[pid] = {
            "player_id": pid,
            "name": name or name_map.get(pid) or pid,
            "position": position or position_map.get(pid) or "",
            "team": team or "",

            "targets": 0,
            "receptions": 0,
            "receiving_yards": 0,
            "receiving_tds": 0,

            "carries": 0,
            "rushing_yards": 0,
            "rushing_tds": 0,

            "red_zone_targets": 0,
            "red_zone_receptions": 0,
            "red_zone_carries": 0,
            "red_zone_receiving_tds": 0,
            "red_zone_rushing_tds": 0,

            "pass_attempts": 0,
            "red_zone_pass_attempts": 0,

            # Used internally to calculate target share.
            "_share_teams": set(),

            "weekly": {},
        }

    return players[pid]


# =========================================================
# TEAM PASSING TOTALS
# =========================================================

print("Calculating team passing totals...")

pass_attempts = df[
    df["pass_attempt"].eq(1)
    & df["posteam"].notna()
].copy()

team_pass_attempts = (
    pass_attempts
    .groupby("posteam")
    .size()
    .to_dict()
)


# =========================================================
# RECEIVING
# =========================================================

rec = df[
    df["pass_attempt"].eq(1)
    & df["receiver_player_id"].notna()
    & df["posteam"].notna()
].copy()

for pid, g in rec.groupby("receiver_player_id"):

    pid = str(pid)

    position = position_map.get(pid, "")

    if position not in SKILL:
        continue

    row = g.iloc[0]

    player_team = (
        g["posteam"]
        .value_counts()
        .index[0]
    )

    p = ensure(
        pid,
        name_map.get(pid)
        or row.get("receiver_player_name")
        or pid,
        position,
        player_team,
    )

    # Track every team the player received targets for.
    for team in g["posteam"].dropna().astype(str).unique():
        p["_share_teams"].add(team)

    # -------------------------
    # Overall receiving
    # -------------------------

    p["targets"] += len(g)

    p["receptions"] += int(
        g["complete_pass"]
        .fillna(0)
        .eq(1)
        .sum()
    )

    p["receiving_yards"] += int(
        g["receiving_yards"]
        .fillna(0)
        .sum()
    )

    p["receiving_tds"] += int(
        g["pass_touchdown"]
        .fillna(0)
        .eq(1)
        .sum()
    )

    # -------------------------
    # Red zone receiving
    # -------------------------

    rz = g[
        pd.to_numeric(
            g["yardline_100"],
            errors="coerce"
        ).le(20)
    ]

    p["red_zone_targets"] += len(rz)

    p["red_zone_receptions"] += int(
        rz["complete_pass"]
        .fillna(0)
        .eq(1)
        .sum()
    )

    p["red_zone_receiving_tds"] += int(
        rz["pass_touchdown"]
        .fillna(0)
        .eq(1)
        .sum()
    )

    # -------------------------
    # Weekly receiving
    # -------------------------

    for week, w in g.groupby("week"):

        week = int(week)

        e = p["weekly"].setdefault(
            week,
            {
                "week": week,
                "targets": 0,
                "carries": 0,
                "receptions": 0,
                "red_zone_targets": 0,
                "red_zone_carries": 0,
            },
        )

        e["targets"] += len(w)

        e["receptions"] += int(
            w["complete_pass"]
            .fillna(0)
            .eq(1)
            .sum()
        )

        e["red_zone_targets"] += int(
            pd.to_numeric(
                w["yardline_100"],
                errors="coerce"
            ).le(20).sum()
        )


# =========================================================
# RUSHING
# =========================================================

rush = df[
    df["rush_attempt"].eq(1)
    & df["rusher_player_id"].notna()
    & df["posteam"].notna()
].copy()

for pid, g in rush.groupby("rusher_player_id"):

    pid = str(pid)

    position = position_map.get(pid, "")

    if position not in SKILL:
        continue

    row = g.iloc[0]

    player_team = (
        g["posteam"]
        .value_counts()
        .index[0]
    )

    p = ensure(
        pid,
        name_map.get(pid)
        or row.get("rusher_player_name")
        or pid,
        position,
        player_team,
    )

    # Track teams for consistency.
    for team in g["posteam"].dropna().astype(str).unique():
        p["_share_teams"].add(team)

    # -------------------------
    # Overall rushing
    # -------------------------

    p["carries"] += len(g)

    p["rushing_yards"] += int(
        g["rushing_yards"]
        .fillna(0)
        .sum()
    )

    p["rushing_tds"] += int(
        g["rush_touchdown"]
        .fillna(0)
        .eq(1)
        .sum()
    )

    # -------------------------
    # Red zone rushing
    # -------------------------

    rz = g[
        pd.to_numeric(
            g["yardline_100"],
            errors="coerce"
        ).le(20)
    ]

    p["red_zone_carries"] += len(rz)

    p["red_zone_rushing_tds"] += int(
        rz["rush_touchdown"]
        .fillna(0)
        .eq(1)
        .sum()
    )

    # -------------------------
    # Weekly rushing
    # -------------------------

    for week, w in g.groupby("week"):

        week = int(week)

        e = p["weekly"].setdefault(
            week,
            {
                "week": week,
                "targets": 0,
                "carries": 0,
                "receptions": 0,
                "red_zone_targets": 0,
                "red_zone_carries": 0,
            },
        )

        e["carries"] += len(w)

        e["red_zone_carries"] += int(
            pd.to_numeric(
                w["yardline_100"],
                errors="coerce"
            ).le(20).sum()
        )


# =========================================================
# PASSING
# =========================================================

passers = df[
    df["pass_attempt"].eq(1)
    & df["passer_player_id"].notna()
    & df["posteam"].notna()
].copy()

for pid, g in passers.groupby("passer_player_id"):

    pid = str(pid)

    row = g.iloc[0]

    player_team = (
        g["posteam"]
        .value_counts()
        .index[0]
    )

    p = ensure(
        pid,
        name_map.get(pid)
        or row.get("passer_player_name")
        or pid,
        "QB",
        player_team,
    )

    # Track teams for consistency.
    for team in g["posteam"].dropna().astype(str).unique():
        p["_share_teams"].add(team)

    p["pass_attempts"] += len(g)

    p["red_zone_pass_attempts"] += int(
        pd.to_numeric(
            g["yardline_100"],
            errors="coerce"
        ).le(20).sum()
    )

    # -------------------------
    # Weekly passing
    # -------------------------

    for week, w in g.groupby("week"):

        week = int(week)

        e = p["weekly"].setdefault(
            week,
            {
                "week": week,
                "targets": 0,
                "carries": 0,
                "receptions": 0,
                "red_zone_targets": 0,
                "red_zone_carries": 0,
            },
        )

        e["pass_attempts"] = (
            e.get("pass_attempts", 0)
            + len(w)
        )

        e["red_zone_pass_attempts"] = (
            e.get("red_zone_pass_attempts", 0)
            + int(
                pd.to_numeric(
                    w["yardline_100"],
                    errors="coerce"
                ).le(20).sum()
            )
        )


# =========================================================
# TARGET SHARE
# =========================================================

print("Calculating target share...")

for p in players.values():

    teams = p["_share_teams"]

    # For traded players, combine the pass attempts
    # from every team they played for.
    denominator = sum(
        team_pass_attempts.get(team, 0)
        for team in teams
    )

    if denominator > 0:
        p["target_share"] = round(
            (p["targets"] / denominator) * 100,
            1
        )
    else:
        p["target_share"] = 0.0

    # RZ target share:
    # player red-zone targets /
    # team red-zone targets.
    rz_denominator = 0

    for team in teams:

        team_rz_targets = len(
            rec[
                (rec["posteam"] == team)
                & pd.to_numeric(
                    rec["yardline_100"],
                    errors="coerce"
                ).le(20)
            ]
        )

        rz_denominator += team_rz_targets

    if rz_denominator > 0:
        p["red_zone_target_share"] = round(
            (
                p["red_zone_targets"]
                / rz_denominator
            ) * 100,
            1
        )
    else:
        p["red_zone_target_share"] = 0.0


# =========================================================
# FINALIZE
# =========================================================

result = []

for p in players.values():

    # Keep meaningful players.
    if (
        p["targets"] < 10
        and p["carries"] < 10
        and p["pass_attempts"] < 10
    ):
        continue

    p["weekly"] = [
        p["weekly"][week]
        for week in sorted(p["weekly"])
    ]

    # Remove internal calculation field.
    p.pop("_share_teams", None)

    result.append(p)


# Default ordering.
result.sort(
    key=lambda p: (
        -max(
            p["targets"],
            p["carries"],
            p["pass_attempts"],
        ),
        p["name"],
    )
)


# =========================================================
# WRITE FILE
# =========================================================

out = Path("data/players.json")
out.parent.mkdir(parents=True, exist_ok=True)

out.write_text(
    json.dumps(
        {
            "season": season,
            "players": result,
        },
        indent=2,
    )
)

print(
    f"Wrote {out} with {len(result)} players."
)
