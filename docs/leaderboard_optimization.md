# Phase 29 — Leaderboard Optimization Playbook

A decision framework to follow once real training results exist
(macro-AUC from `train.py`, per-finding breakdowns from
`analysis/error_analysis.py`). This is not a script — it's a
checklist of diagnostic questions and corresponding actions, since
the right move always depends on what the actual numbers show.

---

## Step 1: Read the per-finding AUC table

Run `error_analysis.py` and look at `error_by_finding.csv` /
the per-finding AUC report from `train.py`. Sort by weakest first.

Ask for each weak finding:

- **Is it rare?** Check positive rate from
  `generate_derived_labels.py`'s logged output (e.g. Fracture 6.2%,
  Lateral OA 6.2%). Rare findings are inherently harder to learn and
  more sensitive to small validation sets (see `competition_notes.yaml`
  -> `metric.implications`).
- **Is it mostly derived-label data?** Check
  `error_by_label_source.csv`. If a weak finding's errors concentrate
  in `label_source == "derived"` rows, the bottleneck may be **label
  noise from NLP extraction**, not the model. Compare against
  `validate_extractor.py`'s per-finding precision/recall from Phase 8
  -- findings we already know score lower there (PF OA, Synovitis,
  Effusion, meniscus) are the most likely candidates.
- **Is it disagreeing with pseudo-labels?** Check
  `pseudo_vs_derived_comparison.csv`. High disagreement on a finding
  suggests the derived labels for that finding are unreliable.

## Step 2: Match the diagnosis to an action

| Diagnosis | Action |
|---|---|
| Rare finding, otherwise clean labels | Increase `pos_weight` for that finding in `FocalLoss` (see `losses/focal_loss.py`), or oversample positive studies for that finding specifically. |
| High derived-label noise for this finding | Revisit `extract_labels.py` patterns for that specific finding (see Phase 8's validation results as a starting point), or down-weight derived-label loss contribution for that finding only. |
| High pseudo-label disagreement | Treat disagreeing studies as lower-confidence; consider excluding them from training or manually spot-checking a sample of the original report text. |
| Weak across ALL findings, ground-truth and derived alike | Likely a genuine model capacity/training issue, not a label issue -- consider more epochs, a stronger backbone (`efficientnet_b0`, `convnext_tiny`), or more training data (larger local subset or full Kaggle dataset). |
| Strong on ground-truth, weak on derived, gap is large | Confirms the derived-label pipeline is a real bottleneck -- prioritize improving `extract_labels.py` over model architecture changes. |

## Step 3: Check worst individual studies

From `worst_studies.csv`: are the worst studies concentrated in
specific languages (cross-reference `detected_language` from
`train_with_english_reports.csv`)? Translation quality issues would
show up here as a pattern, not randomly distributed.

## Step 4: Ensemble and TTA as final levers

Once individual-model issues are addressed as much as practical:

- Confirm TTA is helping, not just adding noise: compare
  `validate_tta.py` output with/without TTA on the same checkpoint.
- Train a second, architecturally different backbone
  (`efficientnet_b0` or `convnext_tiny` via `train.py
  --backbone_name`) and combine via `ensemble_infer.py`. Ensembling
  gives the most reliable score bump when the two models make
  *different* kinds of errors -- check whether their per-finding AUC
  weaknesses (Step 1) differ from each other before assuming the
  ensemble will help.

## Step 5: Re-validate with proper cross-validation

Before trusting any conclusion above too strongly: single-split
validation (`manifest_val.csv`) is noisy, especially for rare
findings (see Phase 13/14 notes). Re-check conclusions against
`cross_validation.py`'s k-fold results before making large
architecture or loss-function changes based on a single val split.

---

**Remember:** the competition's own documentation
(`competition_notes.yaml`) states class prevalence may not match
between train/public leaderboard/final eval, and that AUC ranking
quality matters more than calibration. Don't over-optimize for one
validation split's exact numbers -- optimize for robustness across
splits.