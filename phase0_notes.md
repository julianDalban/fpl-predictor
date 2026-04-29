# Phase 0 Takeaways: Data Layer Lockdown

## What this phase was for

Phase 0 establishes the data pipeline and feature scaffolding that every later modeling phase builds on. No models are tuned here — the deliverable is a deterministic raw-to-processed pipeline, frozen train/val/test splits, and a set of design decisions documented well enough that downstream experiments can toggle features rather than re-engineer the data layer.

**Train:** 21/22 + 22/23. **Val:** 23/24 GW 2-33. **Test:** 23/24 GW 34-38, locked until Phase 3.

## Decisions locked

### Cross-season name matching

A diacritic-stripped, lowercased name field is the canonical join key across seasons. Matching on raw player names silently misidentified roughly 15 established players (Coufal, Souček, Højlund, etc.) as new arrivals because of accent inconsistencies in the source data — a class of error that propagates invisibly into any feature that depends on prior-season information.

### Position normalization

The 21/22 source data contained 80 rows encoded as `"GKP"` while later seasons used `"GK"`. Normalizing to a single canonical label before one-hot encoding produces exactly four position columns that sum to 1 per row.

### Stage-1 hurdle labels

Two binary labels were added to support a later hurdle architecture: did the player play any minutes, and did the player cross 60 minutes. Both are computed at the **fixture level** (before double-gameweek collapse) using a per-fixture max rule rather than summed minutes across DGW fixtures. Justification: the rule maps to actual FPL scoring — the 2-pt appearance bonus and clean-sheet eligibility trigger at 60 minutes in a single fixture, not cumulatively. 33 DGW rows differ between the two rules (e.g. a 45+45 DGW falsely flags as "crossed 60" under summed minutes but correctly does not under per-fixture max).

### K-means archetypes

K=4 fit on train-only player-season aggregates, restricted to player-seasons with at least 5 gameweeks of data. Features are five numeric playing-style stats: mean minutes per gameweek, mean influence/creativity/threat per appearance, and mean value. **Position one-hots are deliberately excluded from the feature set** — including them caused position rediscovery (silhouette 0.45 but clusters were ~99% single-position). Removing them dropped silhouette to 0.325 but produced cross-positional archetypes.

The four clusters: **Defensive starters** (16% GK + 72% DEF, full minutes, low attacking output), **Rotational/fringe** (low-minute attackers across MID and FWD), **Premium attackers** (high threat, high price, captaincy candidates — and notably containing 3% DEF, the attacking fullbacks like Trent and Cancelo whose output groups them with elite MIDs), **Creative regulars** (cross-positional starters with attacking output).

Holdout assignment uses each player's most recent prior-season aggregates rather than within-23/24 information, preserving leakage safety. Player-seasons with no eligible prior history are routed to a distinct "unclassified" bucket. That bucket holds roughly 25% of 23/24 minutes — a structural feature of English football (three promoted clubs plus summer transfers from foreign leagues every year), not a 23/24 anomaly. Forcing those players into a position-and-price-tier fallback was rejected as making up data we don't have; the unclassified bucket is the epistemically honest choice.

### Splits

Train, validation, and test masks are frozen and treated as constants for the rest of the project. The test set is the final 5 gameweeks of 23/24 and stays untouched until Phase 3. End-of-season gameweeks have known distribution shift — secured top-4 sides rotate, relegation-threatened sides pile on intensity, mid-table teams have nothing to play for. Per-gameweek validation performance should be reported alongside aggregate metrics so test-set numbers can be contextualized.

### Feature groups

Features are organized into named groups (pre-kickoff context, position one-hots, FPL's expected points, an ICT/form block flagged as collinear and earmarked for PCA, other lag features, and cluster ID). Downstream model code declares feature sets compositionally — by group name — rather than enumerating column names. This is what makes Phase 1's many feature-set ablations cheap to run.

## Cross-cutting findings

- **The cluster ID feature has different statistical meaning in train vs validation by design.** Train rows get their cluster from their own season's aggregates; 23/24 rows get it from the most recent prior season — a leakage-safety requirement, not a bug. The column effectively means "current archetype" in train but "last year's archetype" in validation. A Phase 0 sanity-check confirmed the cost (small train R² gain, slightly larger validation R² loss). K-means surfaced interpretable archetypes, but cross-validating their predictive contribution is fundamentally complicated by the necessarily different feature semantics in holdout. This was confirmed in Phase 1, where cluster ID was dropped from the model feature set on direct ablation but retained as a standalone unsupervised analysis section.

- **"Exactly N rows" is the wrong acceptance criterion.** The original spec asked for an exact row count. The rebuilt pipeline produces a slightly different one because mid-season transfer first-appearances are handled marginally differently when deriving from raw than when extending an older processed file. The shift is documented and accepted.

## Phase 0 → Phase 1

The data layer is locked. Every Phase 1 experiment — flat Ridge/Lasso baselines, the hurdle architecture, Tweedie regression, PCA on the ICT block — consumed this pipeline's output without modification, and every feature-set ablation resolved through the named groups defined here. The cluster ID ablation is the cleanest evidence the scaffolding worked as intended: a deliberately leakage-safe feature was tested, found not to generalize, and dropped — without re-engineering the data layer to do so.