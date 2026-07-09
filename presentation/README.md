# Honours Presentation

Beamer slide deck for the honours thesis talk *Variance Reduction for Training Neural Bayes Estimators* — strictly under 15 minutes, talk only (Q&A separate).

## Files

| File | What it is |
|------|-----------|
| `presentation.tex` | The whole deck: title + 11 numbered talk slides + standout thanks + 5 backup frames. Single, banner-delimited file. |
| `SPEAKER_SCRIPT.md` | Per-slide talking points, the timing budget, and the IS **CUT POINT**. Rehearse from this. |
| `figures/` | Slide-ready images. `rb_hero.png` is derived (see below); the rest are copied from `../tex/figures/`. |
| `tools/check_deck.py` | Structural check (frame balance + figure paths). Needs no LaTeX: `python tools/check_deck.py presentation.tex`. |
| `tools/make_rb_hero.py` | Regenerates `figures/rb_hero.png` by cropping the top row of `rb_tau_curves.png`. |

## Compile

The deck uses the **metropolis** theme, which needs a Unicode engine — **XeLaTeX or LuaLaTeX, not pdfLaTeX**. It has been compiled and visually checked end-to-end with Tectonic. Figures resolve via `\graphicspath{{figures/}}`.

**Tectonic (already set up on this machine, verified):**
```
conda run -n tex tectonic presentation.tex
```
Produces `presentation.pdf`. Tectonic uses XeTeX, so it auto-fetches the metropolis Fira fonts and every LaTeX package, and runs the extra pass for the slide numbers. (The `tex` env was made with `conda create -n tex -c conda-forge tectonic`; the first compile downloads a bundle.)

**Overleaf (easiest to edit):**
1. New Project → Upload, and add `presentation.tex` plus the whole `figures/` folder.
2. Menu → Compiler → **XeLaTeX** (required by metropolis; pdfLaTeX will fail).
3. Recompile. Overleaf reruns automatically, so the slide numbers resolve.

**Local (reinstalled MiKTeX / TeX Live):**
```
xelatex presentation.tex
xelatex presentation.tex      # second pass for the slide-number total
```
or simply `latexmk -xelatex presentation.tex`. On MiKTeX, accept the prompts to auto-install missing packages (`beamer`, `pgf`/`tikz`, `booktabs`, `metropolis`, `fira`). Use **xelatex**, not pdflatex.

## Theme

The deck uses **metropolis** — dark-teal frame titles, Fira Sans, orange accents, a foot progress bar, and a standout closing slide. It needs XeLaTeX/LuaLaTeX (see above). If you ever need a pdfLaTeX-only build, swap `\usetheme{metropolis}` and the `\metroset{...}` line for `\usetheme{default}` and compile with pdflatex — the slide content is theme-independent.

## Cutting importance sampling (if you run long)

The IS material is two frames, fully self-contained:
1. Delete everything between `% >>> IS MODULE START` and `% >>> IS MODULE END` in `presentation.tex`.
2. On frame 2, swap the hook to the short line (it's commented right below the full one).

This drops the talk from ~13:50 to ~11:20. `python tools/check_deck.py presentation.tex` should then report **15** frames (down from 17).

## First-compile QA checklist

- [ ] 18 slides total — title + 11 numbered talk slides + standout thanks + 5 backup; footer shows `n/11`; `check_deck.py` reports 17 frames.
- [ ] Nothing overflows the bottom of a slide — watch **frame 2** (four beats) and **frame 9** (figure + callouts).
- [ ] `rb_hero.png` is legible at projector distance: the legend, the green floor line, and orange-below-blue all read clearly.
- [ ] The guarantee inequality on frame 8 and the table on frame 11 render cleanly.
- [ ] The NBE diagram on frame 4 (`y → network → θ̂`) is centred and uncramped.
- [ ] Rehearse against `SPEAKER_SCRIPT.md`: you hit the guarantee (frame 8) by ~7–8 min and finish under 15.
