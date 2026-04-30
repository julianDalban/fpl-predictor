# Phase 2 Takeaways: xP-Excluded Modeling Track

## What this phase was for

Phase 2 is the counter-experiment to Phase 1: rebuild the best model from raw historical stats alone, with FPL's `xP` removed. The intended deliverable was a quantification of how much pre-kickoff information `xP` encodes that isn't recoverable from raw player statistics — i.e. the gap between Phase 1's xP-included winner and Phase 2's best xP-excluded model.

Two experiments: a linear hurdle baseline (2.0) reusing Phase 1's winning architecture without xP, and an MLP hurdle (2.1) testing whether nonlinear modeling could recover any of the lost performance.

**Train:** 21/22 + 22/23. **Val:** 23/24 GW 2-33. **Test:** untouched.

## Headline result

| Model | Val R² | Val MAE | Notes |
|---|---|---|---|
| xP passthrough (reference) | 0.5169 | 0.8129 | FPL's own predictor |
| Phase 1 hurdle WITH xP (ceiling) | 0.6386 | 0.7116 | Phase 1 winner — upper bound |
| Hurdle linear WITHOUT xP (Exp 2.0) | 0.2999 | 0.9762 | Same architecture as Phase 1 winner, xP removed |
| Hurdle MLP WITHOUT xP (Exp 2.1) | 0.2887 | 0.9625 | Best of 3×3 grid: (128, 64, 32) hidden, α=1e-4 |

**The headline of Phase 2 is the 0.34 R² gap between Phase 1's ceiling and Phase 2's best.** The MLP did not close this gap; combined R² is essentially equal to the linear baseline. The MAE improvement (0.9762 → 0.9625) reflects shrinkage toward the conditional mean rather than recovered signal.

The central finding is a structural negative result: **`xP` encodes pre-kickoff information that is not present in raw historical features alone.** Stripping it cuts hurdle R² roughly in half, and no architecture explored — Ridge, Lasso, MLP — closes the gap.

## What we did

### Experiment 2.0: linear hurdle baseline without xP

Re-ran Phase 1's winning hurdle architecture (LR Stage 1 + Ridge Stage 2) with `xP` removed from both stages. Stage 1 LR val AUC dropped from 0.9466 (Phase 1) to 0.9300 — small loss, justifying retention of the hurdle structure for 2.1. Stage 2 RidgeCV selected α=316 from `np.logspace(-3, 3, 13)`.

Three diagnostics reframed how to read this result:

1. **Stage 2 alone R² on played-only val = 0.085.** The combined hurdle R² of 0.30 was almost entirely Stage 1 doing the work — correctly assigning ~zero to the 60% of rows that are DNPs. The conditional points-given-play task contains very little linear signal once xP is gone.

2. **Ridge α=1 and α=316 give identical predictions to three decimals on the conditional task.** Two-and-a-half orders of magnitude of regularization do nothing because the loss surface is flat — no real signal to fit, so coefficients shrink to near-mean predictions either way. RidgeCV's high α was not over-correction; it was differentiating between near-equivalent points on a flat surface.

3. **Phase 1's Lasso sign-flip prediction failed.** Phase 1 anticipated that `total_points_roll3` would flip from negative to strongly positive once `xP` was removed. At fixed α=1e-4 (Phase 1's reference regime), it stayed negative (-0.11). The form signal did get redistributed, but to `minutes_roll3` (+0.38) and `minutes_lag1` (+0.13) — without `xP`, the points-form story landed on minutes-form, leaving points-form with a residual mean-reversion sign.

The 2.0 Lasso top-10 at α=1e-4 is dominated by structural and market features: `is_dgw` (0.56), `value` (0.54), `pos_GK` (0.31), `transfers_in` (0.17), `selected` (0.12). This was the first sign that `xP`'s information content was structural — fixture difficulty, team strength, set-piece status — not just informational about player form.

### Experiment 2.1: MLP hurdle without xP

Tested whether nonlinear modeling could recover any of the conditional-task signal lost in 2.0. sklearn `MLPRegressor` substituted for torch/keras — L2 (`alpha`) regularization in place of dropout, consistent with the Phase 1 sklearn ecosystem and adequate given the regularization-flat loss surface.

3×3 grid search over `hidden_layer_sizes ∈ {(64,), (64, 32), (128, 64, 32)}` × `alpha ∈ {1e-4, 1e-3, 1e-2}`. Adam, lr=1e-3, ReLU, sklearn early stopping with random 10% val fraction. Hyperparameter selection: MAE on a temporal early-stop set (final 5 GWs of 22/23). Best config: (128, 64, 32), α=1e-4.

**Results:**

| Metric | Linear (2.0) | MLP (2.1) | Δ |
|---|---|---|---|
| Combined R² (val) | 0.2999 | 0.2887 | -0.011 |
| Combined MAE (val) | 0.9762 | 0.9625 | -0.014 |
| Stage 2 alone R² (val) | 0.0852 | 0.0681 | -0.017 |
| Stage 2 alone R² (ES set, 22/23 GW 34-38) | — | 0.1506 | — |

The R²-down / MAE-down pattern is the signature of an over-shrunk model: MLP predictions cluster near the conditional mean, helping typical-case error but hurting variance explained on extremes. The decile residual table confirmed this — predicted-points deciles 6-8 systematically under-predict (mean residuals +0.47, +0.25, +0.23) while the top decile is well-calibrated.

**The MLP's ES vs val R² gap (0.151 vs 0.068) is the more interesting finding.** Same model, similar feature distributions, but val performance roughly halved compared to the temporally-adjacent ES set. Permutation importance showed why: `value` dominated at 0.052 — three times the next feature (`pos_GK` at 0.016). FPL recalibrates player prices each summer using its own pricing model, so a feature whose distribution drifts cross-season concentrates the MLP's failure modes there. The linear baseline distributed weight across `is_dgw`, `value`, and minutes-form features more evenly, partially insulating it from this drift.

**Per-position MAE comparison:**

| Position | n | 2.0 MAE | 2.1 MAE | Δ |
|---|---|---|---|---|
| GK | 2,736 | 0.6188 | 0.6135 | -0.005 |
| DEF | 7,710 | 1.0641 | 1.0620 | -0.002 |
| MID | 10,300 | 0.9880 | 0.9598 | -0.028 |
| FWD | 3,079 | 1.0340 | 1.0325 | -0.001 |

The MLP's MAE gain concentrates on midfielders. Phase 1 had GK as the position most helped by explicit play modeling (rotation patterns); Phase 2 has MID as the position most helped by nonlinear modeling on the conditional task. This is consistent with MIDs having the broadest scoring distribution (goals + assists + clean sheets all contribute) and therefore the most room for interaction structure between e.g. minutes, fixture, and position one-hots.

## Cross-cutting findings

- **`xP` is not just FPL's pre-match prediction; it's an information channel for features this dataset does not contain.** The 0.34 R² gap between Phase 1 and Phase 2 is not a model-class failure. Linear and nonlinear models converge to within 0.011 R² without `xP` on a regularization-flat loss surface. What `xP` encodes — team strength, fixture difficulty (FDR), set-piece taker status, penalty taker designation, bookmaker-implied scoring rates — is not recoverable from lag/roll features of player statistics, regardless of architecture. Phase 2's contribution is to put a number on this: ~0.34 R² of pre-kickoff information lives in `xP` that does not live in raw historical stats.

- **The conditional points-given-play task is genuinely low-signal without xP.** Phase 1 reported `xP` as the dominant Lasso coefficient (~2.4 standardized) without distinguishing between "xP is informative" and "the rest of the feature set is uninformative." Phase 2 disambiguates: removing `xP` collapses Stage 2 R² to 0.085, and an MLP cannot recover what the linear model didn't already extract. The play/no-play decision retained almost all of its signal (AUC 0.9466 → 0.9300), confirming that `xP`'s value lives in the conditional scoring distribution, not the play decision.

- **Cross-season generalization risk concentrates in `value`.** The MLP's reliance on `value` (3× the next feature in permutation importance) made it more vulnerable to cross-season distribution shift than the linear baseline. FPL re-prices players each summer based on its own pricing model; a model that leans heavily on `value` inherits this drift. Worth flagging for Phase 3.

- **Methodology note on temporal validation.** The Phase 2 ES set (final 5 GWs of 22/23) prevented val leakage but did not surface the cross-season generalization gap that ultimately appeared between ES (R² 0.151) and val (R² 0.068). For future projects with cross-season validation, a stronger model selection approach would hold out an entire prior season as ES rather than an in-season tail.

## Phase 2 → Phase 3

Phase 3 evaluates the Phase 1 hurdle (with `xP`) on the locked test set (23/24 GW 34-38). The Phase 2 finding doesn't change Phase 3's protocol but does sharpen expectations:

- The reportable test result is Phase 1's hurdle, not Phase 2's MLP. Phase 2 produced a structural negative result, not a deployable model.
- Two known sources of distribution shift on the test set: end-of-season rotation/intensity (documented in Phase 0) and `value` drift (surfaced in Phase 2). Per-GW test performance should be reported alongside aggregates so the headline test number can be contextualized against both effects.
- The Phase 1 vs Phase 2 comparison is the central writeup contribution. The structural negative result — quantifying that `xP` carries ≈ 0.34 R² of pre-kickoff information not present in raw historical features — is a stronger and more honest finding than "our MLP outperforms linear by X%" would have been.