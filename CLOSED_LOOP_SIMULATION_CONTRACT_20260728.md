# Frozen Closed-Loop Procedural Simulation Contract

## Purpose and scope

This experiment tests the decision layer under repeated perception, finite
robot footprint, execution error, and replanning. It is a procedural 2-D
simulation, not a physics-accurate lunar rover or flight qualification test.

## Episode generation

Generate 300 validation episodes with seed `20260731` and 1,000 independent
test episodes with seed `20260801`. Each episode uses a `48 x 36` grid with
8--14 circular obstacles. Obstacle radii are sampled uniformly from 1.5--3.5
cells. Start and goal lie in opposite boundary bands, are collision free after
footprint inflation, and have a connected ground-truth free-space path.

The robot is a disk of radius 1.0 cell. A safety violation occurs when its
center enters an obstacle radius plus 1.5 cells; collision occurs at obstacle
radius plus the robot radius. Executed positions receive paired zero-mean
Gaussian perturbations with standard deviation 0.20 cell per commanded step.

## Repeated perception

At each replan, three frozen synthetic detectors independently observe the
same terrain. Their obstacle detection probabilities are `(0.82, 0.78, 0.74)`.
Detected centers receive Gaussian error with standard deviation 0.55 cell and
radii receive multiplicative log-normal error with standard deviation 0.12.
Each detector adds Poisson false positives with mean 0.8. A shared per-obstacle
visibility term introduces correlated misses without exposing ground truth to
the decision rule.

Detector fields use the same unit-amplitude Gaussian construction as the image
study. Distance-based clustering produces consensus, union, and CWRF fields;
support one receives CWRF amplitude 0.5 and support at least two receives 1.0.

## Closed-loop execution

At every replan, each method receives detector and fusion fields generated from
the same episode/replan random tape. A* plans from the current cell to the goal
with risk weight 15. The robot commands at most three path steps before new
perception and replanning. Collision terminates the episode. Success requires
reaching within one cell of the goal within 30 replans.

Compared methods are detector 1, consensus, union, CWRF, mean scenario cost,
worst relative regret, SMR-CWRF, and the frozen RCMR-CWRF configuration. Direct
field methods run one A* search; decision methods generate the same six frozen
candidates before selection. Ground truth only generates the environment and
evaluates executed trajectories.

## Metrics and paired inference

Report collision rate, safety-violation rate, task-completion rate, cumulative
ground-truth exposure, executed path length normalized by start-goal distance,
and replanning count. Episode is the independent statistical unit. Use 10,000
paired bootstrap resamples with seed `20260802` for method-minus-CWRF
differences. Report point estimates and 95% intervals; do not convert a trend
whose interval crosses zero into a confirmed improvement.

## Validation and test boundary

The 300 validation episodes may select only the already frozen RCMR parameter
family. Simulator distributions, detector parameters, motion noise, metrics,
and method list may not change after validation results are seen. The 1,000
test episodes run once after the final selection artifact is written.

## Stopping and reporting

If the simulator fails feasibility checks, produces fewer than 1,000 valid test
episodes, or reveals implementation asymmetry between methods, repair the bug
and regenerate both splits with versioned provenance. Scientific results may
not motivate simulator changes. Negative RCMR or SMR results remain visible in
the audit record and manuscript if the closed-loop experiment is reported.

