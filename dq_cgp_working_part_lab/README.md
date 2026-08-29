# DQ-CGP working-part experiment

This directory isolates the contribution of residual injection, matched
binding supervision, and route regularization with one frozen seed (2023).
Production model, training, dataset, matcher, and evaluator files are imported
without modification.

| Variant | Binding | Route | Injection |
|---|---:|---:|---:|
| baseline | 0 | 0 | no DQ |
| full | 0.2 | 0.01 | on |
| no_inject | 0.2 | 0.01 | off |
| no_binding | 0 | 0.01 | on |
| no_route | 0.2 | 0 | on |
| injection_only | 0 | 0 | on |
| binding_only | 0.2 | 0 | off |
| route_only | 0 | 0.01 | off |

`no_inject` intentionally computes temporal attention and routing and keeps
their losses in the graph. It only returns the native D1 state to D2; this is
different from the production beta-zero fast path.

Together the eight variants form a complete 2x2x2 factorial design. This
allows component main effects to be averaged over the other two factors and
also exposes component interactions for seed 2023.
