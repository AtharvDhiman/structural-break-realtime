# Catching Change — real-time structural break detection

A real-time detector that watches a univariate stream one point at a time and, after
each new value, reports how confident it is that the process has **permanently
changed** — a *structural break*. Built for the ADIA Lab Structural Break Challenge
(Real-Time Edition) on CrunchDAO.

**Live site:** _add your Vercel URL here after deploying_

## The pages

- **`index.html`** — landing page.
- **`playground_page.html`** — interactive lab. Build your own stream, drop a break
  wherever you click, and watch a detector score it live. The detection math runs in
  your browser as JavaScript.
- **`explainer_page.html`** — a plain-language explainer of what a structural break is
  and why catching one early matters. No jargon.
- **`demo_page.html`** — replay of the actual competition model on real series.
- **`writeup_page.html`** — the technical write-up: the metric, the calibration, the
  residual monitoring, and the ideas that didn't work.

The site is 100% static HTML/CSS/JavaScript — no backend, no build step. Any static
host (Vercel, Netlify, GitHub Pages) serves it as-is.

## How the detector works (short version)

1. **Learn "normal" from history.** Each series comes with a long calm stretch; the
   model measures how that particular stream usually behaves.
2. **Judge each stream by its own past.** Every new point is scored relative to that
   stream's own history, so a volatile stream isn't unfairly flagged.
3. **Let evidence accumulate.** One odd point proves nothing; persistent drift pushes
   confidence up and keeps it there, because a real break is permanent.

The full model (in `model/`) fits an AR + GARCH model to the history, monitors the
standardized residuals for four kinds of break (mean, variance, shape, dependence),
turns each into a calibrated score, and combines them with a gradient-boosted model.

## Result

Held-out Time-Stratified AUC: **0.518 → 0.533 → 0.557 → 0.575** across baseline,
per-series calibration, residual monitoring, and the stacked model.

## `model/`

- `main.py` — competition entry points (`train` / `infer`), streaming interface.
- `sbrt_core.py` — the engine: residual model, calibrated statistics, incremental scorer.
- `requirements.txt` — Python dependencies.

The trained model file and the competition dataset are intentionally **not** included.

## Deploy

Static site — deploy the repo root to Vercel (framework preset: *Other*, no build
command) or enable GitHub Pages on the default branch.
