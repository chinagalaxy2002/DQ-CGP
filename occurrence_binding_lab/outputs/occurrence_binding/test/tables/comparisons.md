# Occurrence-binding paired bootstrap

All differences are second minus first, sampled by qid, and restricted to multi-occurrence records.

| Comparison | Metric | Mean difference | 95% CI | N qids |
|---|---|---:|---:|---:|
| dq_active - baseline | coverage@5_05 | 0.2560 | [0.1912, 0.3213] | 160 |
| dq_active - baseline | coverage@3_05 | 0.2319 | [0.1740, 0.2919] | 160 |
| dq_active - baseline | duplicate_rate@5_05 | -0.0040 | [-0.1032, 0.0913] | 42 |
| dq_active - baseline | aec_d1 | 0.2952 | [0.2482, 0.3422] | 160 |
| dq_active - baseline | aec_d2 | 0.3828 | [0.3356, 0.4307] | 160 |
| dq_active - baseline | binding_margin_d1 | 0.2451 | [0.2158, 0.2743] | 160 |
| dq_active - baseline | binding_margin_d2 | 0.2353 | [0.2070, 0.2640] | 160 |
| dq_active - baseline | ecr_d1 | -0.4604 | [-0.5415, -0.3792] | 160 |
| dq_active - baseline | ecr_d2 | -0.5363 | [-0.6165, -0.4535] | 160 |
| dq_active - baseline | residual_update_l2_mean | n/a | n/a | 0 |
| dq_active - dq_beta_zero | coverage@5_05 | 0.0021 | [-0.0130, 0.0167] | 160 |
| dq_active - dq_beta_zero | coverage@3_05 | 0.0130 | [-0.0031, 0.0302] | 160 |
| dq_active - dq_beta_zero | duplicate_rate@5_05 | 0.0146 | [-0.0175, 0.0497] | 114 |
| dq_active - dq_beta_zero | aec_d1 | 0.0031 | [0.0000, 0.0094] | 160 |
| dq_active - dq_beta_zero | aec_d2 | 0.0016 | [0.0000, 0.0047] | 160 |
| dq_active - dq_beta_zero | binding_margin_d1 | 0.0011 | [0.0000, 0.0028] | 160 |
| dq_active - dq_beta_zero | binding_margin_d2 | 0.0028 | [-0.0001, 0.0070] | 160 |
| dq_active - dq_beta_zero | ecr_d1 | -0.0063 | [-0.0187, 0.0000] | 160 |
| dq_active - dq_beta_zero | ecr_d2 | -0.0010 | [-0.0031, 0.0000] | 160 |
| dq_active - dq_beta_zero | residual_update_l2_mean | n/a | n/a | 0 |
| dq_active - dq_context_roll | coverage@5_05 | 0.0031 | [-0.0109, 0.0172] | 160 |
| dq_active - dq_context_roll | coverage@3_05 | 0.0109 | [-0.0016, 0.0266] | 160 |
| dq_active - dq_context_roll | duplicate_rate@5_05 | -0.0088 | [-0.0310, 0.0088] | 113 |
| dq_active - dq_context_roll | aec_d1 | 0.0000 | [0.0000, 0.0000] | 160 |
| dq_active - dq_context_roll | aec_d2 | -0.0016 | [-0.0094, 0.0047] | 160 |
| dq_active - dq_context_roll | binding_margin_d1 | 0.0002 | [-0.0045, 0.0040] | 160 |
| dq_active - dq_context_roll | binding_margin_d2 | 0.0019 | [-0.0010, 0.0047] | 160 |
| dq_active - dq_context_roll | ecr_d1 | 0.0000 | [0.0000, 0.0000] | 160 |
| dq_active - dq_context_roll | ecr_d2 | 0.0052 | [-0.0031, 0.0187] | 160 |
| dq_active - dq_context_roll | residual_update_l2_mean | 0.0000 | [0.0000, 0.0000] | 160 |

Residual roll check for dq_active - dq_context_roll: max absolute error 0.0; qids=1036.

