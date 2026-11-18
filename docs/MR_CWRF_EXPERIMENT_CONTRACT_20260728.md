# Frozen MR-CWRF Extension Experiment Contract

## Purpose

Evaluate whether route-level robust selection across coherent detector
hypotheses provides a meaningful decision-algorithm improvement over the
already frozen CWRF-A* method. This extension may replace the paper's primary
method only after passing the validation, external-evidence, and integrity
gates below.

## Protected Prior Work

- The July 27 anonymous `v2` package is immutable.
- The July 28 common-weight CWRF paper and evidence remain a complete fallback.
- No MR-CWRF result may retroactively alter the original CWRF selection or be
  described as part of its untouched primary test.
- The existing CWRF parameters remain fixed: cluster IoU `0.5`, singleton
  amplitude `0.5`, disagreement gain `0`, risk weight `15`, grid width `96`,
  maximum challenge routes per frame `3`, and route seed `20260726`.

## Research Question

Can route-level minimax-regret selection over coherent ensemble-perception
hypotheses improve ground-truth safety-envelope violation and exposure over
static CWRF-A* without an excessive path-length penalty?

## Frozen Candidate Set

For every frame and start-goal pair, generate exactly these candidates at the
inherited risk weight `15`:

1. one fixed-risk A* path for each of the three frozen detector inputs;
2. an ensemble consensus-only A* path;
3. an unweighted ensemble-union A* path;
4. the frozen CWRF-A* path (`alpha=0.5`, `g=0`).

All candidates use the existing Gaussian field, A* implementation, grid, and
deterministic tie behavior. A failed candidate remains in the audit output and
is assigned infinite decision score; it is not silently replaced.

## Frozen Decision Rule

Let `P` be the candidate paths and let `R_m` be the fixed-risk field from
detector hypothesis `m` for `m in {0,1,2}`. For candidate path `p`, compute

`C_m(p) = L(p) + w * E_m(p)`,

where `L` is grid path length, `w=15`, and `E_m` is the step-length-weighted
integral of `R_m` along `p`. Ground truth is never used in `C_m`.

For each detector hypothesis, define normalized regret

`r_m(p) = (C_m(p) - min_q C_m(q)) / (min_q C_m(q) + 1e-9)`.

The MR-CWRF score is

`S_lambda(p) = mean_m r_m(p) + lambda * max_m r_m(p)`.

Choose the candidate with minimum score. Exact score ties are resolved by:

1. lower maximum regret;
2. lower mean regret;
3. lower mean predicted scenario cost;
4. lower geometric path length;
5. fixed candidate identifier order.

## Validation-Only Development

The only selectable MR parameter is

`lambda in {0, 0.25, 0.5, 1, 2}`.

The inherited CWRF and A* parameters may not be changed. Selection uses only
the existing held-out validation-selection partition. Ground truth may be used
for validation evaluation, never for route selection.

Before implementing parameter selection, run an oracle-headroom diagnostic on
validation only. The oracle chooses among the same candidates by the
lexicographic ground-truth order: violation, exposure, length ratio, candidate
identifier. This is an unattainable upper-bound diagnostic and may not be
reported as a deployable method.

## Validation Eligibility Gate

MR-CWRF becomes test-eligible only if one frozen lambda satisfies all of:

- lower aggregate challenge-route violation than frozen CWRF;
- lower aggregate challenge-route exposure than frozen CWRF;
- mean length-ratio increase no greater than `0.01` over frozen CWRF;
- at least one of violation or exposure has a frame-bootstrap 95% interval
  wholly below zero versus CWRF;
- no common-weight union point simultaneously dominates its violation,
  exposure, and length ratio.

If multiple lambdas are eligible, select lexicographically by violation,
exposure, length ratio, lambda. If none are eligible, MR-CWRF stops and no test
parameter is selected.

## Evaluation Roles

- Existing sequence-disjoint clean test: post-primary extension because prior
  CWRF outcomes are already known.
- Existing visual stress suite: prior diagnostic evidence only; it cannot tune
  or confirm MR-CWRF.
- PCAM/TCAM images: external evaluation only if provenance, publication rights,
  label semantics, train/test independence, and valid terrain ROI are verified
  before detector extraction.
- If PCAM/TCAM qualification fails, obtain or generate a documented independent
  lunar-image test source before making external-generalization claims.

The statistical unit is the image/frame; route differences are averaged within
image before inference. Use 5,000 frame-bootstrap resamples and retain every
predeclared camera stratum, route seed, corruption, and ablation.

## External-Data Gate

Before inference, create a signed provenance record containing:

- source URL or mission/archive identifier and required citation;
- license or permission basis for analysis and figure reproduction;
- meanings of the green, blue, and red annotations;
- image and annotation SHA-256 hashes;
- evidence that no external image was used in detector training, validation,
  hyperparameter selection, or prior CAC analysis;
- a frozen rule defining valid terrain pixels, starts, goals, and routes.

PCAM and TCAM must be reported separately as well as jointly. Ground-truth rock
boxes/masks may construct evaluation fields and safety masks only. A sky or
terrain-validity mask may restrict the planning domain, but obstacle labels may
not construct predicted decision fields.

## Frozen Unseen Stress Suite

Only after validation selection, apply all conditions below using
`default_rng(20260729 + frame_id)` and clip output to `[0,1]`:

- Poisson shot noise with peak `30`;
- multiplicative speckle noise with Gaussian sigma `0.03`;
- salt-and-pepper impulse noise with total probability `0.01`, split equally;
- JPEG encode/decode at quality `50`;
- brightness factor `0.60` followed by Poisson shot noise with peak `20`.

Every condition and failed detector run must be retained. These same-source
corruptions are robustness evidence, not an independent dataset.

## Comparators and Ablations

Report geometric A*, all three fixed-risk anchors, consensus-only, union,
frozen CWRF, and selected MR-CWRF. Required MR ablations are:

- mean regret only (`lambda=0`);
- selected mean-plus-worst regret;
- worst regret as a diagnostic ranking;
- candidate set without CWRF;
- candidate set containing only the three detector-specific paths;
- validation-only oracle candidate selector.

Run common risk weights `{5,10,15,20}` as a post-selection sensitivity. The
selected MR-CWRF point remains `w=15`; other weights cannot replace it.

## Test and Paper Promotion Gate

MR-CWRF may replace CWRF as the paper's central algorithm only if:

- the validation eligibility gate passes;
- clean post-primary test violation and exposure both improve in point estimate
  versus CWRF within a `0.01` length-ratio increase;
- the selected point is Pareto non-dominated in the common-weight comparison;
- qualified external evidence is directionally positive jointly and does not
  show a severe reversal in either PCAM or TCAM;
- runtime and every negative condition are reported;
- all numerical claims reproduce from archived raw trials.

If the external sample is too small for decisive inference, report effect sizes
and intervals without claiming statistical confirmation or generalization.

## Stop and Integrity Rules

- Never tune lambda, candidate membership, cost normalization, or tie-breaking
  after viewing clean-test, external, or unseen-stress outcomes.
- Never drop a detector, route, frame, camera stratum, corruption, or seed after
  observing its result unless a predeclared mechanical validity rule applies.
- If oracle headroom is negligible, stop MR-CWRF before implementation and
  document the negative result.
- If validation has no eligible lambda, stop MR-CWRF without test execution.
- If MR-CWRF fails its promotion gate, retain it as a negative extension and
  revert the paper to the protected CWRF version.
- FA-CWRF is a separate fallback mechanism and requires a new frozen contract;
  it may not be introduced to repair an observed MR-CWRF test result.

## Frozen-at-Creation Checksums

- July 27 anonymous `v2` package SHA-256:
  `E632E44C38987F07D7575635810752F70B60B0946D7784978D4E524732FD7854`
- July 28 pre-MR `paper.pdf` SHA-256:
  `7FDB1FBEFA7C1385231E5F171BB8E14410FE661589FFF6C64B2FAC0A9CB2CD8B`
- July 28 pre-MR research snapshot SHA-256:
  `DCE4D25DEA3AC289FBC14AF4DA69258364B4E312B41EA6716042BC37506C51D2`

Contract created before MR-CWRF oracle, validation, clean-test, external, or
unseen-stress results were computed.
