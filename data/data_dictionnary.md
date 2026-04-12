# Data Dictionary — `fpl_modeling_data.csv`

**74,477 rows × 51 columns** (44 features + 1 target + 6 identifiers)
Derived from 3 FPL seasons (2021-22 through 2023-24), gameweeks 2–38.

---

## Identifiers (not used as model features)

| Column | Type | Description |
|--------|------|-------------|
| `name` | str | Player name as listed in FPL |
| `season` | str | Season identifier, e.g. `2022-23` |
| `GW` | int | Gameweek number (2–38; GW 1 dropped due to no lag history) |
| `player_season` | str | Compound key: `{name}_{season}`. Used for grouping timelines. Not stable across seasons — same player has different keys each year |
| `team` | str | Player's FPL team name at the time of the gameweek |
| `position` | str | Playing position: `GK`, `DEF`, `MID`, `FWD` |

---

## Target Variable

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `total_points` | int | [-4, 26] | FPL points awarded for this gameweek. Composite of minutes played, goals, assists, clean sheets, bonus, cards, etc. In DGWs, this is the sum across both fixtures |

---

## Pre-Kickoff Features (directly usable)

These are known before the gameweek deadline and require no lagging.

| Column | Type | Description |
|--------|------|-------------|
| `was_home` | float | 1.0 = home, 0.0 = away, 0.5 = one home + one away fixture (DGW only) |
| `opponent_team` | int | Integer ID (1–20) of the opposing team. In DGWs, this is the first opponent only — a known limitation |
| `value` | int | Player price in tenths of £M (e.g. 120 = £12.0M). Set before the GW deadline |
| `selected` | int | Number of FPL managers who own this player at the GW deadline |
| `transfers_in` | int | Managers who transferred this player in before the GW |
| `transfers_out` | int | Managers who transferred this player out before the GW |
| `transfers_balance` | int | `transfers_in - transfers_out`. Positive = gaining popularity |
| `xP` | float | FPL's own pre-match expected points prediction. Also serves as a natural baseline |
| `pos_DEF` | int | 1 if defender, 0 otherwise |
| `pos_FWD` | int | 1 if forward, 0 otherwise |
| `pos_GK` | int | 1 if goalkeeper, 0 otherwise |
| `pos_MID` | int | 1 if midfielder, 0 otherwise |
| `is_dgw` | int | 1 if this gameweek had two fixtures for this player, 0 otherwise |

---

## Engineered Features

| Column | Type | Description |
|--------|------|-------------|
| `gws_played` | int | Cumulative count of prior gameweeks in which this player played >0 minutes (this season). A proxy for match fitness and rotation risk |

---

## Lag Features (derived from post-kickoff stats)

For each stat below, two features exist:

- `{stat}_lag1` — the value from the immediately preceding gameweek
- `{stat}_roll3` — the mean of the preceding 1–3 gameweeks (expanding for GWs 2–3, then a true 3-GW window)

Both are computed within each player-season to prevent cross-season or cross-player bleed.

| Base Stat | Same-GW corr with target | What it measures |
|-----------|--------------------------|------------------|
| `total_points` | — (it IS the target) | Overall FPL output. Lagged version captures recent form |
| `minutes` | 0.640 | Minutes played. Primary indicator of whether a player is in the starting XI |
| `goals_scored` | 0.682 | Goals scored in the match |
| `assists` | 0.462 | Assists provided |
| `clean_sheets` | 0.577 | 1 if the team conceded 0 goals and the player played ≥60 min, else 0 |
| `goals_conceded` | 0.202 | Goals conceded by the player's team while the player was on the pitch |
| `bonus` | 0.752 | Bonus points (0–3) awarded to the top BPS performers per fixture |
| `bps` | 0.910 | Raw Bonus Points System score. Composite of match actions |
| `saves` | 0.165 | Saves made (goalkeepers only; 0 for outfield players) |
| `influence` | 0.828 | ICT component measuring impact on the match result |
| `creativity` | 0.482 | ICT component measuring chance creation |
| `threat` | 0.578 | ICT component measuring goal threat |
| `ict_index` | 0.772 | Composite of influence + creativity + threat |
| `yellow_cards` | 0.085 | Yellow cards received |
| `red_cards` | -0.043 | Red cards received |

This produces **30 lag/rolling columns** (15 stats × 2 variants).