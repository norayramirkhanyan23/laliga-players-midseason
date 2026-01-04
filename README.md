# LaLiga Players — Midseason (Understat CSV)

Mid-season LaLiga player analysis using **Understat** team CSV exports.  
Focus: **chance creation** (xA/90 vs A/90) and **expected vs actual goal involvement** (xG/90 + xA/90 vs Goals/90 + Assists/90).

## What’s inside

### Visualizations (saved in `/outputs`)
- **Top 10 Expected vs Actual Goal Involvement**
  - Bars: `xG/90` + `xA/90` (expected)
  - White dot: `Goals/90 + Assists/90` (actual)
- **Top 3 teams × Top 3 players**
  - Same idea as above, but split by team panels
- **Chance Creation Maps (team-specific)**
  - Scatter: `xA/90` (expected assists per 90) vs `A/90` (assists per 90)
  - Diagonal line = “actual equals expected”
  - Above line = overperforming expected assists
  - Below line = underperforming expected assists

> Filters used in scripts: typically `min >= 450` (season to date), plus optional `goals + assists >= 5` for cleaner plots.

## Data source

- **Understat** team pages → export player stats as CSV  
  CSV files are placed in the project root (e.g., `barca.csv`, `real.csv`, `atleti.csv`, etc.)

## Project structure

