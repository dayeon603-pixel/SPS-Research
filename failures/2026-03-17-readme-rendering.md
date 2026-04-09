# Failure: README Rendering — Blockquotes + Math on GitHub

**Date:** 2026-03-17
**Status:** resolved

---

Pushed the README with A3–A5 assumptions inside blockquote syntax (`>`). On GitHub the CommonMark renderer processes italic markers before handing off to the math renderer. So `_{\mathcal{T},\varepsilon}` inside a blockquote got partially treated as italic markup and the output was garbage — broken symbols mixed with partial formatting.

Fixed by removing the blockquotes entirely and putting the math on standalone lines. Obvious in retrospect. Always preview math-heavy markdown on GitHub before finalizing.
