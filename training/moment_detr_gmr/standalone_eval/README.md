# Standalone Evaluation Helpers

This directory contains lightweight Moment-DETR evaluation utilities used by the Moment-DETR-GMR training loop. The project-level GMR benchmark metrics are provided in `eval/`; these helpers are kept here for validation during model training.

Prediction files are JSONL files. Each line should contain:

```json
{
  qid: 2579,
  query: Locate the throw-in action.,
  vid: match_clip_0001,
  pred_relevant_windows: [[50.0, 56.0, 0.9974]],
  pred_exist_score: 0.93
}
```

`pred_exist_score` is optional for plain Moment-DETR checkpoints and present for Moment-DETR-GMR checkpoints.
