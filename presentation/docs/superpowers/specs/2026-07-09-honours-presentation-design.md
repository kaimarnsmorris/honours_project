# Honours Thesis Presentation — Design Spec

**Date:** 2026-07-09
**Author:** Kai Marns-Morris (with Claude)
**Thesis:** *Variance Reduction for Training Neural Bayes Estimators*
**Deliverable:** A Beamer slide deck (+ speaker script + backup slides + compiled PDF) for a **strictly-under-15-minute** honours presentation.

---

## 1. Goal

Produce a talk that leaves the audience **understanding the key points of the thesis as well as possible** in under 15 minutes. This is an assessed honours presentation — the mark rewards clear communication of the contribution, not technical exhaustiveness. Understanding beats rigour throughout.

## 2. Constraints (fixed by the user)

| Constraint | Decision |
|---|---|
| **Audience** | 3 maths-department professors, ~1 statistician. Mathematically able, but **not assumed to know** simulation-based inference, neural Bayes estimators, MCMC/Gibbs, or the Rao–Blackwell theorem. |
| **Time** | 15 min is **talk only** (Q&A separate). Target **~13:30** of content; hard cap **15:00**. Fallback **~11:00** if IS is dropped. |
| **Tool** | **LaTeX Beamer** (matches thesis notation; reuse figure PNGs). |
| **Emphasis** | **Rao–Blackwellisation is the star; importance sampling is the contrast.** |
| **IS is liftable** | Importance sampling must be a self-contained module droppable if time is tight, leaving no dangling threads. |

## 3. The thesis in brief (what the talk must convey)

- **The pivot observation:** Training a neural Bayes estimator (NBE) is *itself* a Monte Carlo computation. The training loss is an MC estimate of the Bayes risk, so the training gradient is a random quantity with variance — a property of *how the loss is sampled*, not of the inference problem. Therefore Monte-Carlo variance reduction applies to the training loop.
- **Contribution 1 — Rao–Blackwellisation (the win):** When a block of the parameter vector has a tractable conditional posterior (conditional conjugacy), integrate it out of the loss analytically instead of sampling it. Under squared-error loss this substitutes the conditional posterior mean for the sampled target; the residual conditional variance enters only as a constant that drops from the gradient. The resulting gradient is the Rao–Blackwellisation of the standard one → law of total variance → **positive semi-definite variance gap: it can never increase gradient variance, for any model or architecture.** For a linear network the guarantee strengthens to the fitted weights themselves.
  - **Empirical payoff (Bayesian linear regression, MLP, Gibbs as gold benchmark):** lower loss in **every cell** of the experimental grid; closes about **half the gap** between the standard NBE and the Bayes-optimal (Gibbs) floor; matches the standard NBE at **a quarter of the batch size**.
- **Contribution 2 — Importance sampling (the contrast):** More general (needs no conjugacy) — reshape the simulation distribution, then reweight back to the original Bayes-risk objective — but carries **no guarantee**. On the autoregressive (AR1) model with a recurrent (GRU) network it gives **no accuracy improvement**: the estimator already sits near the Bayes-optimal floor across the prior, and the proposal oversamples the regions it already handles best; pushed harder, reweighting only lowers the effective sample size and adds noise.
- **The punchline:** Conjugacy — long exploited to make Gibbs sampling possible — turns out to be equally useful for **training the estimators meant to replace it**.
- **The sharpest contrast (full version only):** The guarantee carries through to the *variance of the fitted weights* for RB, but nothing in IS does — that asymmetry is the deepest difference between the two techniques.

## 4. Narrative approach

**Chosen — "A classical noise-reduction idea, applied somewhere new."**
Arc: motivate the problem → deliver the pivot observation (training is Monte Carlo) → introduce the variance-reduction *intuition* from scratch (replace a noisy random draw by its exact conditional average; *then* name it Rao–Blackwellisation) → apply it and reach the guarantee (the peak, stated plainly, not proved) → the linear-regression experiment confirms it → importance sampling as a one-module honest contrast → close on the conjugacy punchline.

**Rejected alternatives:**
- *Result-first hook* — open cold on the headline number, backfill later. Punchier but risky for non-specialists who need the setup; can lose the room in the first two minutes.
- *Balanced two-technique study* — equal time on RB and IS. Dilutes the win and contradicts the chosen emphasis.

## 5. Design principles

1. **Understanding over rigour.** At most **one equation per slide**, and only when it clarifies. Everything else is words and pictures.
2. **Introduce, never assume.** No prior knowledge of NBEs, MCMC/Gibbs, or Rao–Blackwell is presumed. Concepts are carried by intuition and diagrams; technical names are attached *after* the idea lands.
3. **The guarantee is the emotional peak**, delivered as *"for free, provably, it never hurts and usually helps"* — the one-line reason (law of total variance) is stated, not derived.
4. **Rigour lives in backup slides.** Proofs, the linear-network weight argument, IS proposal families, and MCMC/HMC detail are available for Q&A but never slow the main talk.
5. **IS is a liftable module** (slides 10–11). The spine before and after never references it.

## 6. Slide-by-slide plan (target ~13:30)

| # | Slide | ~Time | Content | Figure(s) |
|---|-------|------|---------|-----------|
| 1 | Title | 0:20 | Title, author, supervisors, UWA, degree | uwa.jpeg |
| 2 | The talk in one breath | 0:40 | Roadmap embedded in the hook. Full + short (IS-dropped) versions of the closing clause. | — |
| 3 | Inference is expensive *and repeated* | 1:00 | Bayesian inference needs the posterior; standard tool (MCMC) is re-run from scratch for every new dataset — the pain amortisation solves. | — (simple diagram) |
| 4 | Neural Bayes estimators: amortise it | 1:30 | Bayes estimator = the function that best predicts parameters from data (posterior mean under squared error). Train a neural net **once** on simulated (θ, y) pairs to approximate it; deployment = a single forward pass. | — (simulate→train→forward diagram) |
| 5 | **The pivot:** training is Monte Carlo | 1:30 | The training objective is an MC estimate of the Bayes risk → the gradient is random, with variance → that variance is about *sampling the loss*, not the problem → it can be reduced. | — |
| 6 | A trick for killing Monte-Carlo noise | 0:50 | Intuition first: if you average random draws and can replace some with their exact conditional average, noise drops. *Then* name it: "statisticians call this Rao–Blackwellisation." | — (before/after noise sketch) |
| 7 | The Rao–Blackwellised loss | 1:20 | When a parameter block has a tractable conditional posterior, integrate it out: substitute the conditional posterior mean for the sampled target; residual variance is a constant that drops from the gradient. | — (≤1 equation) |
| 8 | **The guarantee** (peak) | 1:10 | Plain statement: never increases gradient variance — any model, any architecture. One-line reason (law of total variance). For linear nets it even reduces the variance of the trained weights. | grad_variance.png |
| 9 | Does it pay off? Linear regression | 2:00 | Setup in one line (linear regression, MLP, Gibbs = gold floor). Three numbers: beats standard NBE in every cell; closes ~half the gap to Gibbs; matches it at ¼ batch size. | gibbs_vs_nn.png, rb_tau_vs_samples.png |
| 10 | *(IS module)* The contrast: importance sampling | 1:30 | Second technique — no conjugacy needed, reshape + reweight — but no guarantee. On AR1/GRU it didn't help: estimator already near the floor, proposal mismatched; pushed hard, ESS collapses. | ess_vs_alpha.png (+ optional error_map.png) |
| 11 | *(IS module)* What separates them | 1:00 | RB: guaranteed, carries to the estimator's weights, needs conjugacy. IS: general but contingent, doesn't carry to the weights. | — (contrast table) |
| 12 | Takeaway | 0:50 | Punchline: conjugacy — the classical trick behind Gibbs — is equally useful for training the estimators meant to replace it. Optional one-line future work. | — |
| 13 | Thanks / Q&A | 0:10 | Acknowledgements; backup slides follow. | — |

**Peak:** slides 8–9. **Cut point:** end of slide 9 (see §7).

## 7. Importance-sampling cut strategy

- IS is confined to **slides 10–11**. Removing them takes the talk from ~13:30 to **~11:00**.
- **Slide 2 hook** is written so the IS clause is a single removable phrase ("…and I'll contrast it with a second, more general technique"). Two speaker-script versions of that line: full and short.
- **Speaker script marks `CUT POINT`** after slide 9 with an alternate one-sentence transition straight into slide 12 (Takeaway).
- The closing punchline (slide 12) does not depend on IS and is unchanged either way.

## 8. Backup slides (after slide 13, for Q&A only — not counted in the 15 min)

- One-line law-of-total-variance argument for the variance gap.
- The linear-network result: reduction in the variance of the fitted weights (the last-layer argument).
- IS detail: proposal families, ESS vs α, region/error breakdown of why it didn't help.
- MCMC/HMC one-slide reminder (leapfrog/Hamiltonian) if a professor probes the baseline.
- Architecture specifics (MLP for linear regression; GRU for AR1).
- Additional experimental grid cells / training curves.

## 9. Deliverables

1. **Beamer source** — organised so the IS module and backup slides are clearly delimited and easy to cut.
2. **Figures** — copied from `tex/figures/` into `presentation/figures/`.
3. **Per-slide speaker script** with a running time budget, the `CUT POINT` marker, and full/short hook variants.
4. **Backup slides** (§8).
5. **Compiled `presentation.pdf`.**

## 10. Tooling & build

- **Beamer.** Target the clean *metropolis* theme aesthetic; if it does not compile cleanly on the Windows/TeX setup (Fira fonts, `pgfopts`), fall back to a minimal custom theme with no external font dependencies. The talk must compile with `pdflatex` (no forced XeLaTeX/LuaLaTeX requirement).
- Reuse thesis notation/macros where helpful to stay consistent with the written thesis.

## 11. Figures to verify during implementation

Confirm each hero figure actually shows what the story needs; swap or regenerate from `experiments/` scripts if not:

- `gibbs_vs_nn.png` — RB NBE vs standard NBE vs Gibbs floor (the headline result).
- `rb_tau_vs_samples.png` — RB loss vs batch size / samples (the "¼ batch size" claim).
- `grad_variance.png` — gradient-variance reduction (illustrates the mechanism/guarantee).
- `ess_vs_alpha.png` — IS effective sample size vs α (the "ESS collapses" claim).
- Optional IS support: `error_map.png`, `region_breakdown.png`, `binned_mse.png`, `proposals.png`.

## 12. Risks & open questions

- **Metropolis compile risk** on Windows — mitigated by the fallback theme (§10).
- **Figure legibility on a projector** — some thesis figures are dense; may need font-size/crop tweaks or regeneration for slide use.
- **13:30 is a target, not a guarantee** — must be validated by rehearsal against the speaker-script timings; the IS cut is the primary release valve.

## 13. Out of scope

- Deep MCMC/HMC/leapfrog theory (thesis ch. 2) — compressed to one motivating line; detail relegated to backup.
- Full amortisation theory and importance-sampling derivations — intuition only in the main talk.
- Any new experiments or results beyond what the thesis already contains.
