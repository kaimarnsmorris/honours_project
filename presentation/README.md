# Honours Presentation

Beamer slide deck for the honours thesis talk *Variance Reduction for Training Neural Bayes Estimators* — strictly under 15 minutes, talk only (Q&A separate).

## Files

| File | What it is |
|------|-----------|
| `presentation.tex` | The whole deck: 13 main frames + 5 backup frames. Single, banner-delimited file. |
| `SPEAKER_SCRIPT.md` | Per-slide talking points, the timing budget, and the IS **CUT POINT**. Rehearse from this. |
| `figures/` | Slide-ready images. `rb_hero.png` is derived (see below); the rest are copied from `../tex/figures/`. |
| `tools/check_deck.py` | Structural check (frame balance + figure paths). Needs no LaTeX: `python tools/check_deck.py presentation.tex`. |
| `tools/make_rb_hero.py` | Regenerates `figures/rb_hero.png` by cropping the top row of `rb_tau_curves.png`. |

## Compile

There is **no LaTeX toolchain on this machine** (MiKTeX was removed), so compile one of these ways.

**Overleaf (easiest):**
1. New Project → Upload, and add `presentation.tex` plus the whole `figures/` folder.
2. Menu → Compiler → **pdfLaTeX** (leave it here for the default theme).
3. Recompile. Run it twice if the slide-number footer (`n / N`) looks wrong on the first pass.

**Local (reinstalled MiKTeX / TeX Live):**
```
pdflatex presentation.tex
pdflatex presentation.tex      # second pass for the slide-number total
```
or simply `latexmk -pdf presentation.tex`. On MiKTeX, accept the prompts to auto-install any missing packages (`beamer`, `pgf`/`tikz`, `booktabs` — all standard).

## Optional: the metropolis look

The default theme is deliberately plain and dependency-free so it compiles anywhere. For the sleeker *metropolis* look:
1. In `presentation.tex`, comment out the `% ---- Theme: clean, self-contained` block and uncomment `\usetheme{metropolis}`.
2. Compile with **XeLaTeX** (on Overleaf: Compiler → XeLaTeX) so the Fira fonts render.

## Cutting importance sampling (if you run long)

The IS material is two frames, fully self-contained:
1. Delete everything between `% >>> IS MODULE START` and `% >>> IS MODULE END` in `presentation.tex`.
2. On frame 2, swap the hook to the short line (it's commented right below the full one).

This drops the talk from ~13:50 to ~11:20. `python tools/check_deck.py presentation.tex` should then report **10** main frames.

## First-compile QA checklist

- [ ] 13 main frames + 5 backup frames (18 total; `check_deck.py` prints the count).
- [ ] Nothing overflows the bottom of a slide — watch **frame 2** (four beats) and **frame 9** (figure + callouts).
- [ ] `rb_hero.png` is legible at projector distance: the legend, the green floor line, and orange-below-blue all read clearly.
- [ ] The guarantee inequality on frame 8 and the table on frame 11 render cleanly.
- [ ] The NBE diagram on frame 4 (`y → network → θ̂`) is centred and uncramped.
- [ ] Rehearse against `SPEAKER_SCRIPT.md`: you hit the guarantee (frame 8) by ~7–8 min and finish under 15.
