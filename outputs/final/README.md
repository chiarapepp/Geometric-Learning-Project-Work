# Curated Final Outputs

This directory contains the small, presentation-ready artifacts intended for version control. Raw experiment runs, checkpoints, logs, caches, and intermediate outputs remain ignored by Git.

Contents:

- `final_report_notes.md`: concise protocol, coverage, and main findings.
- `tables/`: CSV and Markdown summaries for reproducible inspection.
- `plots/`: benchmark, convergence, robustness, model-comparison, and temporal-weight figures.
- `qualitative/`: representative event visualizations, point-cloud corruptions, compact reconstruction comparisons, and normalized convergence curves.

The current snapshot was selected from `outputs/final_report_time_weights_complete`, generated on 2026-07-25. Its coverage matrix reports 23 complete experiment combinations out of 30; consult `tables/coverage_matrix.md` before drawing aggregate conclusions.

Large model checkpoints are intentionally excluded and should be shared through an artifact store or release if required.
