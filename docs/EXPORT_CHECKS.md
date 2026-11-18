# Export validation — 2026-11-18

Completed locally using the existing Python 3.13 analysis environment:

- 13 original focused unit tests passed.
- Recomputed all reported closed-loop summary means from 7,000 method-episode rows
  (1,000 paired scenarios, seven methods); all matched the exported summary.
- Ran the documented two-episode procedural smoke command successfully.
- Checked local links in both README files: no missing targets.
- Scanned exported text for common access-token formats, private-key headers,
  credential-bearing URLs, personal email addresses and Windows user-home paths:
  no matches in this scan. This is a bounded scan, not a universal security guarantee.

Smoke outputs are kept under ignored `scratch/`, separate from frozen paper results.
This check did not retrain detectors, rerun the full benchmark or establish new results.
A clean dependency installation on a second machine was not tested.

Some auxiliary module names refer to earlier research branches. They are included
only because preserved analysis scripts import their helpers; they do not replace
the frozen SMR-CWRF method.
