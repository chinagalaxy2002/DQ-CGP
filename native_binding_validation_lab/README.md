# Native-binding validation

This isolated experiment asks whether DQ-CGP can be reduced to a training-only
matched binding regularizer.

1. `train_native_binding.py` trains plain Moment-DETR with zero added trainable
   parameters. A forward hook captures native D1 cross-attention, restricts and
   renormalizes it over valid video tokens, and applies the same final-Hungarian
   matched GT-mass loss used by DQ binding.
2. `verify_stripped.py` removes all `query_cgp.*` tensors from a binding-only
   checkpoint and checks plain Moment-DETR predictions against the no-injection
   DQ model.

No production model, trainer, matcher, dataset, or evaluator file is edited.
