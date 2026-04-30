# Phase 1 Takeaways: xP-Included Modeling Track

## What this phase was for

Phase 1 establishes the best achievable validation performance when FPL's own pre-match expected points (`xP`) is available as a feature. This is the **upper-bound track** of the project: the headline finding is the gap between this number and Phase 2's best, which is run without `xP` and reflects what we can predict from raw stats alone.

The bar to clear is `xP` itself used as a passthrough prediction, beating FPL's own predictor by a meaningful margin is the only result that justifies the modeling effort.

**Train:** 21/22 + 22/23 (46,991 rows). **Val:** 23/24 GW 2-33 (23,825 rows). **Test:** 23/24 GW 34-38 (untouched in Phase 1).

## Headline result

| Model | Val R² | Val MAE | Notes |
|---|---|---|---|
| Global mean | -0.0102 | 1.5528 | Floor: any worse and the model is broken |
| Trailing 3-GW avg | 0.1139 | 1.0405 | Simplest non-trivial baseline |
| xP passthrough | 0.5169 | 0.8129 | FPL's own predictor: the bar to beat |
| Ridge flat (α=1) | 0.5989 | 0.8275 | +0.082 R² over xP, but slightly *worse* MAE |
| **Hurdle (LR + Ridge α=1)** | **0.6386** | **0.7116** | **+0.122 R², −0.101 MAE over xP (Phase 1 winner)** |
| Tweedie (power=1.1, α=0.01) | 0.5113 | 0.8619 | Did not beat xP, single-link compression hurt fit |
| Hurdle + PCA (n=5 on ICT) | 0.6383 | 0.7117 | Performance neutral; PCA serves an interpretive role |

The hurdle architecture is the Phase 1 winner. Phase 2's xP-excluded MLP will be measured against R² = 0.6386 / MAE = 0.7116.

## Models tried, in order of contribution

### Flat linear baseline (Ridge / Lasso)

A standard regression baseline on the full feature set: `BASE_PREKICKOFF + POSITION + xP + ICT_BLOCK + OTHER_LAGS`. Both Ridge and Lasso landed at essentially identical performance (R² ≈ 0.599, MAE ≈ 0.83), with regularization having no meaningful effect until α reached 10⁴+ for Ridge, confirming the feature set is not overfit-prone in any practical L2 regime.

**The interesting finding is the R² / MAE divergence.** Ridge improves R² over xP by 0.082 (16% relative gain on a hard-signal target) but is *worse* than xP on per-position MAE in three of four positions (GK, DEF, FWD). The model captures more variance, specifically it leans harder on likely-big-output rows using lag features, but at the cost of typical-case noise. Per-position MAE breakdowns showed the R² gain comes mostly from MID/DEF, where `total_points_roll3` and `bps_roll3` carry recent-form signal that `xP` doesn't fully encode; on FWDs and GKs, `xP` already captures their narrower scoring distributions well, and adding lag features mostly adds noise.

**The `cluster_id` feature was dropped from the feature set.** A direct ablation showed it cost val R² in both Ridge and Lasso. K-means clustering remains as a standalone unsupervised section in the writeup, but is not used as a downstream model feature.

**The flat linear model is the wrong tool for this target.** It treats DNPs (about 60% of rows scoring zero) and active-player scoring as one continuous distribution, when structurally they are two regimes. The structured-zero models in 1.2 and 1.3 address this directly.

### Hurdle model (Phase 1 winner)

Two-stage architecture decoupling the structural-zero generation from the conditional scoring distribution:

- **Stage 1:** Logistic regression predicting `P(played_any = 1)`. Selected over SVM-RBF after a controlled comparison: SVM (trained on a 10,000-row stratified subsample due to compute cost with Platt scaling) and LR (trained on full data) achieved indistinguishable performance: AUC 0.9443 vs 0.9466, Brier 0.0849 vs 0.0863. The play/no-play decision is well-approximated by a linear boundary in the pre-kickoff feature space, so the SVM's non-linearity contributed nothing. LR was selected because it used the full training data and required no subsample caveat. The SVM run satisfies the course's SVM requirement and produced a clean negative finding.

- **Stage 2:** Ridge (α=1) trained only on rows where `played_any = 1`. Predicts expected points conditional on the player being on the pitch.

- **Combined prediction:** `P(played) × E[points | played]` applied to all val rows.

**Threshold choice rationale.** `played_any` was used as the stage-1 cut rather than `played_60min`. The structural zero in this target is "no minutes played", getting any minutes guarantees at least one appearance point, so `y > 0`. `played_60min` cuts inside the positive distribution, which would force stage 2 to never see cameo appearances and bias the conditional estimate.

**Why it wins.** R² = 0.6386 is a +0.040 gain over flat Ridge. The MAE story is more striking: 0.7116 vs flat Ridge's 0.8275: a 14% reduction in typical error. The biggest single MAE improvement is on goalkeepers (0.679 → 0.445), where the rotation pattern is most structural and most cleanly captured by an explicit "is this player starting" model.

The residuals-by-decile diagnostic confirmed no threshold on `P(play)` is needed at inference: the lowest deciles showed tightly-clustered near-zero residuals, meaning the multiplicative product handles DNPs naturally.

### Tweedie regression (did not pan out)

A single-stage alternative that handles zero-inflation natively through the compound Poisson-Gamma family (power between 1 and 2). Negative target values were clipped to zero before training (220 train rows, 0.47%); val targets were left unclipped for fair evaluation. A 2D grid sweep over `power ∈ {1.1, 1.3, 1.5, 1.7, 1.9}` × `alpha ∈ {0, 0.01, 1.0}` was run.

**Best Tweedie was R² = 0.5113 — worse than flat Ridge (0.5989) and far worse than the hurdle (0.6386).** Several configurations produced numerically unstable fits (R² as low as -42 at high power with low regularization), reflecting log-link explosion under un-regularized GLMs rather than a finding about the data. Among stable configurations, the achievable R² capped near `xP` passthrough.

**Why Tweedie underperformed.** Two structural reasons. First, the log link compresses high-end variation: for a target where most values are 0–10, fitting `log(points)` blunts the points-2 vs points-12 distinction that matters for FPL. Second, Tweedie's single coefficient vector and single link function must balance the zero-mass and the positive tail simultaneously; the hurdle sidesteps this entirely with separate models and links (logit for play, identity for points-given-play).

**The finding is informative.** For zero-inflated continuous targets where the play decision and the conditional scoring decision have meaningfully different functional forms, a unified single-stage model cannot match a decoupled two-stage one. Tweedie's parsimony advantage doesn't recover what it loses to architectural rigidity here.

### PCA on the ICT block (neutral, but interpretively valuable)

The progress report's correlation heatmap identified the ICT block (12 columns: lag1 and roll3 versions of influence, creativity, threat, ict_index, bps, minutes) as heavily collinear. PCA was applied to that block only, leaving `BASE_PREKICKOFF`, `POSITION`, `xP`, and `OTHER_LAGS` raw and standardized normally. A single PCA was fit on the full-train ICT block and applied identically to both hurdle stages.

**Component selection.** PC1 captures 70% of variance, PC2 adds 10%, PC3-PC4 add ~6% each, then a clear elbow before PC5 (3%) and a flat tail. Five components retain 96% of variance and were selected on the elbow rule.

**The components are interpretable.**

- **PC1 — player involvement axis.** All twelve ICT variables load positively in a tight 0.25–0.32 range, reading as "everything goes up together when the player is on the pitch and contributing."

- **PC2 — within-position attacking-style axis.** Positive loadings on threat (0.42–0.45) and creativity (0.25–0.27); negative loadings on bps and minutes (−0.27 to −0.31). This captures attacking specialists who post threat and creativity *despite* lower minutes — an axis that distinguishes player styles within positions, not redundant with the `position` one-hots.

**Performance impact: negligible.** Hurdle val R² 0.6386 → 0.6383 (Δ = -0.0003). Stage 1 metrics matched to 4 decimals. Per-position MAEs all within 0.001.

**The honest framing.** PCA does not earn its keep as a model improvement — Ridge's L2 was already handling the collinearity. Where PCA earns its place in the writeup is as **structural characterization of the feature space**: the ICT block can be summarized by an involvement axis and a style axis, paired with the K-means clustering as the unsupervised analysis component of the project. Treating PCA as a "successful dimensionality reduction with preserved performance" is the accurate framing.

## Cross-cutting findings

- **The play/no-play decision is the single biggest signal in the data.** The hurdle's stage 1 achieves 0.95 AUC with only linear features, and the largest performance gain in the entire phase came from explicitly modeling that decision rather than letting it leak into a continuous regression. This is intuitive in retrospect — a player who doesn't play scores zero with certainty, and most pre-kickoff features (recent minutes, fixture status, gws_played) carry strong rotation signal.

- **`xP` is the dominant linear feature.** Lasso coefficient analysis at α=10⁻⁴ showed `xP` with by far the largest magnitude (~2.4 in standardized space, vs `n_fixtures` next at ~0.6). This is expected — `xP` is FPL's full-stack pre-match prediction, and any linear model should lean on it heavily. The interesting Lasso finding was that `total_points_roll3` had a *negative* coefficient: with `xP` already encoding the matchup, recent over-performance reads as luck and predicts mean reversion. Without `xP`, this coefficient would flip strongly positive — a clean illustration of how `xP` reshapes the role of every other feature.

- **GK predictions improve dramatically with explicit play modeling.** Per-position MAE on goalkeepers dropped from 0.68 (flat Ridge) to 0.44 (hurdle), a 35% reduction. Goalkeeper rotation in the FPL data is sharper than outfielder rotation (clear backup-keeper patterns), making this the regime where explicit `P(play)` modeling pays off most.

- **Modest absolute R² is the correct expectation for FPL prediction.** Even our best model leaves 36% of variance unexplained, and football's per-game noise (red cards, bonus point allocation, defensive own goals, set-piece luck) is genuinely irreducible from pre-kickoff features. The right comparison is not to "high R²" but to whether we beat `xP`, the strongest available pre-kickoff signal — which we do, by 0.122 R² and 0.101 MAE.

## Decisions locked for Phase 2
 
- Feature set: `BASE_PREKICKOFF + POSITION + xP + ICT_BLOCK + OTHER_LAGS`. No CLUSTER (cost val R² in 1.1 ablation).
- Hurdle threshold: `played_any` (structural-zero cut, not a within-positive cut).
- Stage 1: logistic regression. SVM-RBF documented as a comparison run, not selected.
- Test set (23/24 GW 34-38) untouched.
## Phase 1 → Phase 2
 
Phase 2 removes `xP` from the feature set and rebuilds the modeling stack. The architectural lessons carry forward: the hurdle structure is preserved, the `played_any` threshold stays, and the same lag features remain available. The MLP component required by the syllabus enters in Phase 2, where its non-linearity has the best chance of contributing — without `xP` to anchor the linear model, the gap between linear and non-linear predictors should widen.
 
The headline Phase 1 → Phase 2 comparison will be: how much performance is lost when we strip out FPL's own predictor and rely only on raw stats?
 
---
 
## Plots to save for the writeup
 
Save these from your notebooks into a `figures/` folder. I have kept the list small — five plots, each pulling its weight in a different way.
 
1. **`phase1_lasso_coefficients.png`** — from Notebook 03 (1.1), the horizontal bar chart of Lasso coefficients at α=10⁻⁴. This is the single best visual for the `xP`-dominates story and the negative-coefficient-on-`total_points_roll3` finding. One plot, two narrative points.
2. **`phase1_lr_calibration.png`** — from Notebook 04 (1.2), the LR-only reliability diagram (the first calibration plot, before the SVM overlay). Shows that stage 1's `predict_proba` outputs are usable as actual probabilities, which is what makes the multiplicative hurdle product meaningful. Skip the SVM overlay version — the LR-only version is cleaner for the writeup.
3. **`phase1_hurdle_residuals_by_decile.png`** — from Notebook 04 (1.2), the residuals-by-`P(play)`-decile boxplot. This is the evidence for "no inference threshold needed" and visually demonstrates the multiplicative product handling DNPs. If you can only keep four plots, this is the one I'd cut, but it does carry weight for the methodological-decision section.
4. **`phase1_pca_scree.png`** — from Notebook 06 (1.4), the scree plot (left panel of the two-panel figure if you can crop it; otherwise keep both panels). Justifies the n=5 choice from the elbow.
5. **`phase1_pca_loadings.png`** — from Notebook 06 (1.4), the PCA loadings heatmap. The visual evidence for the two interpretive axes (involvement, style). Pairs with the scree plot to make the PCA section of the writeup self-contained.
**Plots I deliberately did *not* recommend keeping:**
 
- The Ridge alpha sweep curve (1.1): the finding is "flat across all reasonable α" — a sentence does that better than a flat line on a plot.
- The Tweedie heatmap (1.3): instability dominates the visual; explain in prose that some configurations diverged numerically and report the best stable one in the comparison table.
- Residuals-by-position boxplots (multiple notebooks): per-position MAE in the comparison table conveys the same information more compactly. Save one only if you specifically write a per-position discussion section.
- The actual-vs-predicted scatter plots (multiple notebooks): they all look roughly the same and don't carry a sharp finding for the writeup. Skip.
If your final report is heavily figure-constrained, the priority order is: 1 > 4 > 5 > 2 > 3.

