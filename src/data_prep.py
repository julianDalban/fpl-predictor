"""
Prepping the data after having collected it. This is our preprocessing step
"""

import unicodedata

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.config import (
    HOLDOUT_SEASON,
    K_CLUSTERS,
    KMEANS_ASSIGN_MIN_GWS,
    KMEANS_FEATURES,
    KMEANS_FIT_MIN_GWS,
    KMEANS_RANDOM_STATE,
    PROCESSED_DIR,
    PROCESSED_PATH,
    RAW_DIR,
    SEASONS,
    TRAIN_SEASONS,
)

LAG_STATS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "bonus", "bps", "saves", "influence", "creativity",
    "threat", "ict_index", "yellow_cards", "red_cards",
]

DGW_FIRST_COLS = [
    "name", "team", "position", "was_home", "opponent_team", "value",
    "selected", "transfers_in", "transfers_out", "transfers_balance", "xP",
]
DGW_SUM_COLS = LAG_STATS

def normalize_name(s) -> str:
    """Lowercase + strip diacritics. Stable cross-season join key."""
    if pd.isna(s):
        return s
    stripped = "".join(
        c for c in unicodedata.normalize("NFKD", str(s))
        if not unicodedata.combining(c)
    )
    return stripped.lower().strip()


def load_raw_season(season: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / season / "merged_gw.csv", low_memory=False)
    df["season"] = season
    df["name_key"] = df["name"].map(normalize_name)
    df["position"] = df["position"].replace({"GKP": "GK"})
    return df

# Stage-1 labels (computed BEFORE DGW collapse)

def compute_stage1_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Per-fixture max rule for played_60min. Must be computed at the fixture
    level — a 45+45 DGW does NOT cross the threshold (no 2-pt bonus or CS
    eligibility in either match), but a summed-minutes rule would falsely
    flag it as 1.
    """
    raw = raw.copy()
    raw["crossed_60"] = (raw["minutes"] >= 60).astype(int)
    raw["any_mins"] = (raw["minutes"] > 0).astype(int)
    return (
        raw.groupby(["name_key", "season", "GW"], as_index=False)
           .agg(
               played_60min=("crossed_60", "max"),
               played_any=("any_mins", "max"),
               n_fixtures=("minutes", "size"),
           )
    )


# ---------------------------------------------------------------------------
# Modeling-table assembly
# ---------------------------------------------------------------------------

def collapse_dgw(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse fixture rows -> one row per (name_key, season, GW)."""
    agg_dict: dict[str, str] = {}
    for col in DGW_FIRST_COLS:
        if col in raw.columns:
            agg_dict[col] = "first"
    for col in DGW_SUM_COLS:
        if col in raw.columns:
            agg_dict[col] = "sum"
    return raw.groupby(["name_key", "season", "GW"], as_index=False).agg(agg_dict)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag1 and roll3 versions of LAG_STATS within player_season groups.
    Roll3 uses min_periods=1 (expanding for the first 1-2 GWs of each season).
    """
    df = df.copy()
    df["player_season"] = df["name_key"] + "_" + df["season"]
    df = df.sort_values(["player_season", "GW"]).reset_index(drop=True)

    for stat in LAG_STATS:
        g = df.groupby("player_season")[stat]
        df[f"{stat}_lag1"] = g.shift(1)
        df[f"{stat}_roll3"] = g.transform(
            lambda s: s.shift(1).rolling(window=3, min_periods=1).mean()
        )
    return df


def add_gws_played(df: pd.DataFrame) -> pd.DataFrame:
    """
    gws_played at GW n = count of prior GWs (within player_season) where
    the player got >0 minutes. Excludes the current GW.
    """
    df = df.copy()
    df["gws_played"] = df.groupby("player_season")["minutes"].transform(
        lambda s: (s > 0).cumsum().shift(1).fillna(0)
    ).astype(int)
    return df

def drop_raw_postkickoff(df: pd.DataFrame) -> pd.DataFrame:
    """
    After lags + gws_played are computed, the raw same-GW post-kickoff
    stats serve no further purpose and become a leakage hazard. Drop them.
    `total_points` is preserved (it's the target).
    """
    raw_postkickoff = [
        "minutes", "goals_scored", "assists", "clean_sheets",
        "goals_conceded", "bonus", "bps", "saves", "influence",
        "creativity", "threat", "ict_index", "yellow_cards", "red_cards",
    ]
    return df.drop(columns=[c for c in raw_postkickoff if c in df.columns])


def drop_gw1_and_finalize_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop GW 1 (no lag history available) and zero-fill any remaining NaN
    lag values (mid-season transfers' first appearance).
    """
    df = df[df["GW"] > 1].copy()
    lag_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_roll3")]
    df[lag_cols] = df[lag_cols].fillna(0)
    return df


def add_position_onehots(df: pd.DataFrame) -> pd.DataFrame:
    """Exactly four columns: pos_DEF, pos_FWD, pos_GK, pos_MID. Sum to 1 per row."""
    df = df.copy()
    onehots = pd.get_dummies(df["position"], prefix="pos").astype(int)
    for col in ["pos_DEF", "pos_FWD", "pos_GK", "pos_MID"]:
        if col not in onehots.columns:
            onehots[col] = 0
    onehots = onehots[["pos_DEF", "pos_FWD", "pos_GK", "pos_MID"]]
    return pd.concat([df, onehots], axis=1)


# ---------------------------------------------------------------------------
# K-means: fit on train, build cluster lookup
# ---------------------------------------------------------------------------

def compute_player_season_aggregates(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Per (name_key, season) playing-style aggregates for K-means.

    Computed from raw fixture rows so DGW fixtures count as separate
    appearances (a player with 5 DGWs across 30 GWs has gws_played=35
    here, which is the right denominator for per-appearance rates).
    """
    df = raw.copy()
    df["played"] = (df["minutes"] > 0).astype(int)
    agg = (
        df.groupby(["name_key", "season"], as_index=False)
          .agg(
              name=("name", "first"),
              position=("position", "first"),
              gws_played=("played", "sum"),
              total_minutes=("minutes", "sum"),
              total_influence=("influence", "sum"),
              total_creativity=("creativity", "sum"),
              total_threat=("threat", "sum"),
              mean_value=("value", "mean"),
          )
    )
    safe = agg["gws_played"].clip(lower=1)
    agg["mean_minutes_per_gw"] = agg["total_minutes"] / safe
    agg["mean_influence_per_app"] = agg["total_influence"] / safe
    agg["mean_creativity_per_app"] = agg["total_creativity"] / safe
    agg["mean_threat_per_app"] = agg["total_threat"] / safe
    return agg


def fit_kmeans(train_aggs: pd.DataFrame) -> tuple[KMeans, StandardScaler]:
    """
    Fit K-means on TRAIN player-seasons with gws_played >= KMEANS_FIT_MIN_GWS.
    Sorted by player_season for determinism; random_state pinned.
    """
    fit_set = (
        train_aggs[train_aggs["gws_played"] >= KMEANS_FIT_MIN_GWS]
        .assign(player_season=lambda d: d["name_key"] + "_" + d["season"])
        .sort_values("player_season")
        .reset_index(drop=True)
    )
    scaler = StandardScaler().fit(fit_set[KMEANS_FEATURES].values)
    X = scaler.transform(fit_set[KMEANS_FEATURES].values)
    km = KMeans(n_clusters=K_CLUSTERS, random_state=KMEANS_RANDOM_STATE, n_init=10).fit(X)
    return km, scaler


def build_cluster_lookup(
    aggs: pd.DataFrame, km: KMeans, scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Returns DataFrame[name_key, season, cluster_id].

    Train rows: cluster_id = km.predict on the player-season's OWN aggregates,
        provided gws_played >= KMEANS_ASSIGN_MIN_GWS. Player-seasons below the
        threshold are absent from the lookup -> caller maps to K (unclassified).

    Holdout rows: cluster_id = km.predict on the player's MOST RECENT prior-season
        aggregates from train (no in-season info, no leakage). Players with no
        eligible prior-season history are absent -> unclassified.
    """
    # Train: assign via own season's aggregates
    train_eligible = aggs[
        aggs["season"].isin(TRAIN_SEASONS)
        & (aggs["gws_played"] >= KMEANS_ASSIGN_MIN_GWS)
    ].copy()
    X_train = scaler.transform(train_eligible[KMEANS_FEATURES].values)
    train_eligible["cluster_id"] = km.predict(X_train)
    train_lookup = train_eligible[["name_key", "season", "cluster_id"]]

    # Holdout: assign via most recent eligible prior-season aggregates
    most_recent_prior = (
        train_eligible.sort_values("season", ascending=False)
                      .drop_duplicates("name_key")
                      [["name_key", "cluster_id"]]
    )
    holdout_keys = aggs[aggs["season"] == HOLDOUT_SEASON][["name_key", "season"]]
    holdout_lookup = holdout_keys.merge(most_recent_prior, on="name_key", how="left")

    return pd.concat([train_lookup, holdout_lookup], ignore_index=True)


# ---------------------------------------------------------------------------
# Merge + acceptance checks
# ---------------------------------------------------------------------------

def merge_extensions(
    df: pd.DataFrame,
    stage1: pd.DataFrame,
    cluster_lookup: pd.DataFrame,
    K: int,
) -> pd.DataFrame:
    """Merge stage-1 labels and cluster_id; derive is_dgw from n_fixtures."""
    df = df.merge(
        stage1[["name_key", "season", "GW", "played_60min", "played_any", "n_fixtures"]],
        on=["name_key", "season", "GW"],
        how="left",
        validate="one_to_one",
    )
    df["is_dgw"] = (df["n_fixtures"] >= 2).astype(int)

    df = df.merge(cluster_lookup, on=["name_key", "season"], how="left")
    df["cluster_id"] = df["cluster_id"].fillna(K).astype(int)
    return df


def assert_no_leakage(df: pd.DataFrame, n_samples: int = 50) -> None:
    """
    Spot-check leakage: total_points_lag1.iloc[i] must equal total_points.iloc[i-1]
    within each player_season group, for a random sample.
    """
    rng = np.random.default_rng(0)
    samples = rng.choice(df["player_season"].unique(), size=n_samples, replace=False)
    failures = 0
    for ps in samples:
        sub = df[df["player_season"] == ps].sort_values("GW").reset_index(drop=True)
        for i in range(1, len(sub)):
            expected = sub.loc[i - 1, "total_points"]
            actual = sub.loc[i, "total_points_lag1"]
            if not np.isclose(actual, expected):
                failures += 1
    if failures:
        raise AssertionError(f"Lag leakage check failed on {failures} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Raw load
    print("Loading raw seasons...")
    raw_all = pd.concat([load_raw_season(s) for s in SEASONS], ignore_index=True)

    # 2. Stage-1 labels (per-fixture max rule, computed BEFORE DGW collapse)
    print("Computing stage-1 labels...")
    stage1_labels = compute_stage1_labels(raw_all)

    # 3. K-means on playing-style aggregates (train-only fit)
    print("Fitting K-means...")
    aggs = compute_player_season_aggregates(raw_all)
    km, scaler = fit_kmeans(aggs[aggs["season"].isin(TRAIN_SEASONS)])
    cluster_lookup = build_cluster_lookup(aggs, km, scaler)

    # 4. Build modeling table
    print("Building modeling table...")
    df = collapse_dgw(raw_all)
    df = add_lag_features(df)
    df = add_gws_played(df)
    df = drop_raw_postkickoff(df)
    df = drop_gw1_and_finalize_lags(df)
    df = add_position_onehots(df)

    # 5. Merge extensions / partition assertion
    df = merge_extensions(df, stage1_labels, cluster_lookup, K=K_CLUSTERS)
    from src.config import FEATURE_GROUPS
    all_grouped = set().union(*FEATURE_GROUPS.values())
    extras = set(df.columns) - all_grouped
    assert not extras, f"columns not in any feature group: {extras}"

    # 6. Acceptance checks
    print("Running acceptance checks...")
    pos_cols = ["pos_DEF", "pos_FWD", "pos_GK", "pos_MID"]
    assert (df[pos_cols].sum(axis=1) == 1).all(), "position one-hots do not sum to 1"
    assert df["played_60min"].isna().sum() == 0
    assert df["played_any"].isna().sum() == 0
    assert df["cluster_id"].isna().sum() == 0
    assert set(df["played_60min"].unique()) <= {0, 1}
    assert set(df["played_any"].unique()) <= {0, 1}
    assert set(df["cluster_id"].unique()) <= set(range(K_CLUSTERS + 1))
    assert ((df["played_60min"] == 1) & (df["played_any"] == 0)).sum() == 0
    assert_no_leakage(df)

    # 7. Save
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"\nWrote {PROCESSED_PATH}")
    print(f"Shape: {df.shape}")
    print(f"\nCluster distribution:")
    print(df["cluster_id"].value_counts().sort_index().to_string())
    print(f"\nLabel rates:")
    print(f"  played_any:    {df['played_any'].mean():.3f}")
    print(f"  played_60min:  {df['played_60min'].mean():.3f}")


if __name__ == "__main__":
    main()