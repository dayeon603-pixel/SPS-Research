# Failure: Output-Only Consistency Metric

**Date:** 2026-02-12
**Status:** resolved

---

Measuring $d_Y(f(x), f(Tx))$ without constraining $T$ seemed more principled. It wasn't.

The problem: a model that memorizes training data can appear perfectly stable because it just looks up the answer regardless of what $Tx$ is. It'll return the same prediction for the original and any paraphrase of a training point. But ask it about a novel paraphrase and it falls apart — which the metric never detects.

Output-space metrics collapse information in a way that hides exactly the failure mode I care about. The metric needs to be in representation space, not just output space.

→ This is what pushed me toward the Jacobian reformulation. Sensitivity needs to be measured at the representation level, not just at the prediction level.
