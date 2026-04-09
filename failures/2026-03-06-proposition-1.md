# Failure: Proposition 1 — Missing Integrability Assumption

**Date:** 2026-03-06
**Status:** resolved

---

Claimed $\mathrm{SPS}_\varepsilon > 0$ as a strict inequality. The proof was: $\exp(-\mathbb{E}[\mathrm{Sens}]) > 0$ because exp is always positive.

But $\exp(-\infty) = 0$. If the expectation diverges to $+\infty$, the claim fails. I needed to assume the sensitivity expectation is finite (Assumption A4, integrability) before the strict positivity claim is valid.

This was a gap in the axioms. Added A4 explicitly. Without it you can only claim $\mathrm{SPS}_\varepsilon \geq 0$.
