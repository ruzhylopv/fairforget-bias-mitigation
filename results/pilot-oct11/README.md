# Pilot runs, 2026-10-11

RMU on `lapa-12b-instruct` (JUPITER, 1 GH200), StereoSet gender sentences.
Always: hooked layer 7, trained `mlp.down_proj.weight` in layers 5–7, batch size 4, bf16,
`--verbose`. File names: `<run>_<split>_<loss positions>_steer<‖c‖>_lr<lr>_a<α>`.

"Nothing changed" level of the unlearn loss: (‖a‖² + ‖c‖²) / H, with word-token norm ‖a‖ ≈ 2,400
(measured) and H = 3840.

| # | Job | Split | Loss positions | Steering | lr | α | Steps | Result |
|---|---|---|---|---|---|---|---|---|
| 1 | 2274503 | no | all | 20 | 5e-5 | 100 | 257 | Flat. Loss ∝ 1/(batch length): the BOS token dominates the MSE |
| 2 | | no | words | 20 | 5e-5 | 100 | 257 | Flat at ≈1,500 = "nothing changed". Measured norms: words ≈2,400, BOS ≈65,000 |
| 3 | | yes | words | 7000 | 5e-5 | 100 | 129 | Flat at ≈14,100 = "nothing changed" (≈14,260). Cosines 0.999 |
| 4 | | yes | words | 2500 | 5e-5 | 100 | 129 | Flat at ≈3,000 = "nothing changed" (≈3,130). Same retain/gradient curves as run 3 |
| 5 | | no | words | 2500 | 5e-2 | 100 | 257 | Jolt at steps 1–3 (cosines 0.92), then back to 0.997. Forget and retain identical |
| 6 | 2280022 | yes | words | 2500 | 5e-2 | 100 | 129 | Same as run 5: cosines back to 0.994–0.995, lines overlap, unlearn 3,262 |
| 7 | 2280377 | yes | words | 2500 | 5e-2 | 10 | 129 | Jolt, then cosines 0.991–0.992, still identical for forget and retain. Unlearn 2,877 (≈8% below "nothing changed") |

Only run 1 has its `losses.csv` here; the others are in `runs/<JOB_ID>/` on JUPITER scratch
(deleted after 90 days).

## Summary

- Masking BOS and padding (`--loss_positions words`) removed the length artefact of run 1.
- With lr 5e-5 the activations do not move at all (runs 2–4). Adam steps are ≈ lr in size,
  independent of the steering coefficient, so a larger ‖c‖ does not help.
- With lr 5e-2 the model moves, but the edit is never selective: forget (stereotype) and retain
  (anti-stereotype) change identically, and the retain term pulls the model back to the original.
  Splitting the items (run 6) and lowering α to 10 (run 7) do not separate them.
- Interpretation: at layer 7, stereotype and anti-stereotype StereoSet sentences are too similar
  for topic-level RMU (WMDP forgets a whole domain against Wikipedia). Candidate next steps: loss
  only from the token where stereotype and anti-stereotype differ, or neutral retain text.
