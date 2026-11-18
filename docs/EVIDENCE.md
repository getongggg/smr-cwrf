# What this archive supports

The selector preserves disagreement through a fixed candidate set and selects
without ground-truth labels. Its success depends on the candidate generator and
on the relevance of the perception scenarios.

Static evaluation: 1,839 primary challenge routes from rendered scenes plus
five corruptions. Shared frames and matched routes must not be mistaken for
independent observations across all conditions.

Closed loop: 1,000 independent scenario seeds, seven paired methods. The
`episode_trials.csv` file therefore has 7,000 method-episode rows. The positive
execution findings concern path length and replanning, not an established gain
in collision rate, safety-envelope violations, or completion rate.

External-domain diagnosis: the real-lunar subset comes from the same source
archive and is not an independently acquired dataset. Its aggregate effect reverses.

Excluded claims: calibrated collision probability, metric localization,
real lunar dynamics, hardware validation, flight readiness, cross-domain safety,
and recovery from a candidate set that contains no adequate route.
