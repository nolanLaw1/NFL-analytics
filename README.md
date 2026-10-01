# NFL Analytics — Unified Site

This package connects LEAGUE, PLAYERS, and TEAMS into one consistent site.

## Structure
- `league.html` — Normal + Red Zone league leaderboards
- `players.html` — searchable player directory
- `player.html` — player detail page
- `teams.html` — team directory
- `team.html` — team tendency dashboard
- `styles.css` — shared site styling
- `build_league_data.py` — builds `data/league.json`
- `build_team_data.py` — builds `data/teams.json`

## Data
Place `league.json`, `players.json`, and `teams.json` in `data/`.
The league and team builders download regular-season nflverse play-by-play.

Example:
    python build_league_data.py 2025
    python build_team_data.py 2025

The existing player data builder from the Players package can be used to create `data/players.json`.

## Run locally
Because browsers may block local JSON fetches from `file://`, serve the folder:
    python -m http.server 8000

Then open:
    http://localhost:8000/league.html

## Deployment
Upload the contents of this folder to GitHub Pages, Netlify, or another static host. Keep the three JSON files inside `data/`.
