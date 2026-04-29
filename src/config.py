"""
Project configuration: paths, splits, feature groups, K-means parameters.

This module is intentionally thin and dependency-light so model code can
import constants and split functions without pulling in the full data
prep pipeline.
"""
from pathlib import Path

import pandas as pd

# === Paths ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "fpl_modeling_data.csv"

# === Seasons ===
SEASONS = ["2021-22", "2022-23", "2023-24"]
TRAIN_SEASONS = ["2021-22", "2022-23"]
HOLDOUT_SEASON = "2023-24"

# === Splits ===
# Train: all of TRAIN_SEASONS
# Val:   HOLDOUT_SEASON, GW 2 through TEST_GW_START - 1  (= GW 2-33)
# Test:  HOLDOUT_SEASON, GW TEST_GW_START through 38     (= GW 34-38, final 5 GWs)
# Caveat: end-of-season GWs have known distribution shift (rotation in
# secured top-4 sides, intensity in relegation battles). Per-GW evaluation
# on val should be reported alongside aggregate metrics so the test-set
# performance can be contextualized.
TEST_GW_START = 34

# === K-means ===
K_CLUSTERS = 4
KMEANS_FIT_MIN_GWS = 5     # min GWs played for a player-season to enter the K-means fit
KMEANS_ASSIGN_MIN_GWS = 3  # min GWs played for a player-season to be eligible for centroid lookup
KMEANS_RANDOM_STATE = 42
KMEANS_FEATURES = [
    "mean_minutes_per_gw",
    "mean_influence_per_app",
    "mean_creativity_per_app",
    "mean_threat_per_app",
    "mean_value",
]
# Position one-hots are deliberately NOT in KMEANS_FEATURES.
# Including them caused position rediscovery (silhouette 0.45 but clusters
# were ~99% single-position). Without them, the algorithm finds within-position
# style structure (e.g. attacking fullbacks clustering with creative MIDs).

CLUSTER_NAMES = {
    0: "Defensive starters",      # 16% GK + 72% DEF, full minutes, low attacking output
    1: "Rotational / fringe",     # mostly attackers (62% MID + 20% FWD), low minutes
    2: "Premium attackers",       # high threat (~30), high price (~£8.3m), captaincy candidates
    3: "Creative regulars",       # cross-positional starters with attacking output
    K_CLUSTERS: "Unclassified",   # insufficient data: new-to-PL players + low-minute fringe
}

# === Feature groups ===
# Every non-ID/non-target column in the processed CSV belongs to exactly one group.
# Downstream model code: `from src.config import get_feature_group`
FEATURE_GROUPS = {
    "IDS": [
        "name", "name_key", "season", "GW",
        "player_season", "team", "position",
    ],
    "TARGET": ["total_points"],
    "STAGE1_LABELS": ["played_60min", "played_any"],
    "BASE_PREKICKOFF": [
        "was_home", "opponent_team", "value", "selected",
        "transfers_in", "transfers_out", "transfers_balance",
        "is_dgw", "n_fixtures", "gws_played",
    ],
    "POSITION": ["pos_DEF", "pos_FWD", "pos_GK", "pos_MID"],
    "XP": ["xP"],
    "ICT_BLOCK": [
        "influence_lag1", "influence_roll3",
        "creativity_lag1", "creativity_roll3",
        "threat_lag1", "threat_roll3",
        "ict_index_lag1", "ict_index_roll3",
        "bps_lag1", "bps_roll3",
        "minutes_lag1", "minutes_roll3",
    ],
    "OTHER_LAGS": [
        "total_points_lag1", "total_points_roll3",
        "goals_scored_lag1", "goals_scored_roll3",
        "assists_lag1", "assists_roll3",
        "clean_sheets_lag1", "clean_sheets_roll3",
        "goals_conceded_lag1", "goals_conceded_roll3",
        "bonus_lag1", "bonus_roll3",
        "saves_lag1", "saves_roll3",
        "yellow_cards_lag1", "yellow_cards_roll3",
        "red_cards_lag1", "red_cards_roll3",
    ],
    "CLUSTER": ["cluster_id"],
}


def get_feature_group(name: str) -> list[str]:
    """Return the column list for a named feature group."""
    if name not in FEATURE_GROUPS:
        raise KeyError(
            f"Unknown feature group: {name!r}. Valid: {sorted(FEATURE_GROUPS)}"
        )
    return list(FEATURE_GROUPS[name])


def get_splits(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Return (train_mask, val_mask, test_mask) — boolean Series aligned to df.index.

    Asserts disjoint and exhaustive coverage of df.
    """
    train_mask = df["season"].isin(TRAIN_SEASONS)
    holdout = df["season"] == HOLDOUT_SEASON
    val_mask = holdout & (df["GW"] < TEST_GW_START)
    test_mask = holdout & (df["GW"] >= TEST_GW_START)

    assert (train_mask & val_mask).sum() == 0, "train/val overlap"
    assert (train_mask & test_mask).sum() == 0, "train/test overlap"
    assert (val_mask & test_mask).sum() == 0, "val/test overlap"
    assert (train_mask | val_mask | test_mask).all(), "rows fall outside train/val/test"

    return train_mask, val_mask, test_mask