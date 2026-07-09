# Honours Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable, pdflatex-safe LaTeX Beamer deck (+ figures, speaker script, backup slides) for a strictly-under-15-minute honours thesis talk that leaves three maths professors understanding the thesis's key points.

**Architecture:** A single `presentation.tex` (16:9, self-contained custom beamer theme, thesis-matching notation macros) with clearly banner-delimited sections; the importance-sampling module is fenced with removable markers so it can be cut if time is tight. Figures are copied from `../tex/figures/`; one dense grid figure is cropped to a legible hero panel with Pillow. A separate `SPEAKER_SCRIPT.md` carries the words and the running time budget. The user compiles (Overleaf or a local TeX install); every task is gated by a Python structural check and, for figures, by reading the produced image back.

**Tech Stack:** LaTeX Beamer (pdflatex-compatible, no external fonts), Python 3.13 + Pillow 11 (figure cropping + structural checks). No LaTeX toolchain is installed locally — the deck is authored for the user to compile.

## Global Constraints

- **Time budget:** target **~13:30** of content, hard cap **15:00** (talk only; Q&A separate). Dropping the IS module (frames 10–11) must yield **~11:00** with no dangling references.
- **Audience:** 3 maths-department professors, ~1 statistician. Assume mathematical maturity but **no** knowledge of NBEs, MCMC/Gibbs, or the Rao–Blackwell theorem. Introduce every domain concept from intuition; attach the technical name only after the idea lands.
- **Depth:** understanding over rigour. **At most one equation per content slide.** Proofs/derivations live in backup slides only.
- **Emphasis:** Rao–Blackwellisation is the star (peak = the guarantee, frames 8–9); importance sampling is a liftable one-module contrast.
- **Portability:** must compile with **pdflatex** on a stock TeX distribution / Overleaf with no manual package hunting. No `metropolis`, no Fira, no XeLaTeX requirement in the default build.
- **Notation:** mirror the thesis — `θ̂_γ` for the network, `θ=(τ,β)`, `y` for data, `γ` for weights, `L` for Bayes risk.
- **Verified facts (do not alter):** RB model = Bayesian linear regression, θ=(τ,β), **β is the conjugate block integrated out**, τ~U(0.01,1) sampled, Gibbs → Bayes floor. Headline numbers: floor **L\*=0.126**; RB closes **~52%** of the MC→floor gap at baseline; RB at **batch 4 matches MC at batch 16** (¼ batch); RB beats MC in **every** grid cell (source: `../tex/chapters/04-rao-blackwellisation.tex` l.834–861). IS: AR(1)+GRU already sits on the Gibbs floor across the prior; aggressive proposals collapse ESS and inflate gradient variance (`../tex/chapters/05-importance-sampling.tex`).

---

## File Structure

```
presentation/
├── presentation.tex          # the whole deck (single file, banner-delimited)
├── SPEAKER_SCRIPT.md         # per-slide script, timing budget, CUT POINT, hook variants
├── README.md                 # how to compile (Overleaf + local), QA checklist
├── figures/                  # slide-ready assets (copied/derived from ../tex/figures)
│   ├── uwa.jpeg
│   ├── rb_tau_curves.png     # full RB grid (backup)
│   ├── rb_hero.png           # DERIVED: legible top-row crop for frame 9
│   ├── rb_tau_vs_samples.png # backup
│   ├── gibbs_vs_nn.png       # IS frame 10
│   ├── ess_vs_alpha.png      # IS frame 10
│   ├── grad_variance.png     # IS backup
│   ├── normal_model_estimates.png  # optional aid / backup
│   ├── proposals.png         # backup
│   └── region_breakdown.png  # backup
└── tools/
    ├── check_deck.py         # structural gate: frame balance + figure paths
    └── make_rb_hero.py       # crops rb_tau_curves.png -> rb_hero.png
```

**Single-file rationale:** ~13 content frames + backups is small; one commented file is easiest to compile and for the user to edit. Section banners (`% ===== ACT N =====`) and IS fence markers (`% >>> IS MODULE START` / `% <<< IS MODULE END`) give the structure without fragmentation.

## Verification philosophy (no local LaTeX)

Because nothing here can run `pdflatex`, each task is gated by what *can* be checked locally:
1. **`tools/check_deck.py`** — hard-fails on unbalanced `\begin{frame}`/`\end{frame}` and on any `\includegraphics` whose target file is missing; warns on brace imbalance. Run after every editing task.
2. **Figure read-back** — after producing/cropping any image, Read it to confirm it renders legibly.
3. **User compile checkpoints** — the plan flags points where the user should compile and eyeball. The final task ships a README compile command + a visual QA checklist (overflow, legibility at projector distance, frame count, timing).

---

### Task 1: Scaffold — theme, notation, title & closing frames, structural checker

**Files:**
- Create: `presentation/presentation.tex`
- Create: `presentation/tools/check_deck.py`
- Create: `presentation/figures/` (copy `uwa.jpeg` only for now)

**Interfaces:**
- Produces: `presentation.tex` with a working preamble (theme + macros `\E \Var \bt \by \bg \net \RB \MC \Normal \given`), a title frame, and a thanks frame. Later tasks insert frames between them at the marked banners.
- Produces: `tools/check_deck.py` exposing CLI `python tools/check_deck.py presentation.tex` → exit 0 = pass.

- [ ] **Step 1: Copy the UWA logo**

```bash
cd "C:/Users/kaima/OneDrive/Documents/GitHub/honours_project/presentation"
mkdir -p figures tools
cp ../tex/uwa.jpeg figures/uwa.jpeg
```

- [ ] **Step 2: Write the structural checker**

Create `presentation/tools/check_deck.py`:

```python
#!/usr/bin/env python3
"""Structural gate for the Beamer deck: no LaTeX toolchain required.
Hard-fails on frame imbalance and missing figure files; warns on brace imbalance."""
import re, sys, pathlib

def strip_comments(tex):
    # remove % comments (not escaped \%)
    return re.sub(r'(?<!\\)%.*', '', tex)

def main(path):
    p = pathlib.Path(path)
    raw = p.read_text(encoding='utf-8')
    tex = strip_comments(raw)
    figdir = p.parent / 'figures'
    errors, warnings = [], []

    nb = len(re.findall(r'\\begin\{frame\}', tex))
    ne = len(re.findall(r'\\end\{frame\}', tex))
    if nb != ne:
        errors.append(f'frame imbalance: {nb} \\begin{{frame}} vs {ne} \\end{{frame}}')

    opens = tex.count('{'); closes = tex.count('}')
    if opens != closes:
        warnings.append(f'brace imbalance (approx): {opens} {{ vs {closes} }}')

    for m in re.finditer(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', tex):
        name = m.group(1)
        cands = [figdir / name] + [figdir / (name + e) for e in ('.png', '.jpg', '.jpeg', '.pdf')]
        if not any(c.exists() for c in cands):
            errors.append(f'missing figure: {name} (looked in {figdir})')

    print(f'frames: {nb} begin / {ne} end')
    for w in warnings: print('WARN:', w)
    for e in errors: print('ERROR:', e)
    if errors:
        sys.exit(1)
    print('OK')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'presentation.tex')
```

- [ ] **Step 3: Write the deck preamble + title + thanks frames**

Create `presentation/presentation.tex`:

```latex
% ============================================================
%  Honours presentation — Kai Marns-Morris
%  Variance Reduction for Training Neural Bayes Estimators
%  Build: pdflatex presentation.tex   (twice, for the slide numbers)
%  Portable: no external fonts, no metropolis, pdflatex-safe.
% ============================================================
\documentclass[aspectratio=169,11pt]{beamer}

% ---- Theme: clean, self-contained (no external packages) ----
\usetheme{default}
\usecolortheme{default}
\setbeamertemplate{navigation symbols}{}   % no nav clutter
\setbeamertemplate{itemize items}[circle]
\setbeamertemplate{caption}{\raggedright\insertcaption\par}
\definecolor{accent}{RGB}{16,78,139}        % deep blue
\definecolor{floorcol}{RGB}{34,139,34}      % floor green (match figures)
\setbeamercolor{frametitle}{fg=accent}
\setbeamercolor{title}{fg=accent}
\setbeamercolor{structure}{fg=accent}
\setbeamerfont{frametitle}{series=\bfseries}
\setbeamertemplate{frametitle}{\vspace{0.4em}\insertframetitle\par}
% simple footline: short title | slide number
\setbeamertemplate{footline}{%
  \hfill\usebeamerfont{page number in head/foot}%
  \color{gray}\scriptsize\insertframenumber\,/\,\inserttotalframenumber\hspace*{1em}\vspace*{0.4em}}

% ---- Optional upgrade (Overleaf): comment the theme block above and
% uncomment below; compile with XeLaTeX for best fonts.
% \usetheme{metropolis}

% ---- Maths ----
\usepackage{amsmath,amssymb}
\usepackage{bm}
\usepackage{booktabs}

% ---- Notation macros (mirror the thesis) ----
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\operatorname{Var}}
\newcommand{\Normal}{\mathcal{N}}
\newcommand{\given}{\,|\,}
\newcommand{\bt}{\bm{\theta}}
\newcommand{\by}{\mathbf{y}}
\newcommand{\bg}{\bm{\gamma}}
\newcommand{\bb}{\bm{\beta}}
\newcommand{\net}{\hat{\bm{\theta}}_{\bm{\gamma}}}
\newcommand{\RB}{\mathrm{RB}}
\newcommand{\MC}{\mathrm{MC}}

% ---- Title metadata ----
\title{Variance Reduction for Training\\Neural Bayes Estimators}
\author{Kai Marns-Morris}
\institute{The University of Western Australia}
\date{Honours (BPhil) 2026}

\begin{document}

% ===== FRAME 1: TITLE =====
\begin{frame}[plain]
  \centering
  \IfFileExists{figures/uwa.jpeg}{\includegraphics[height=0.16\textheight]{uwa.jpeg}\par\vspace{0.8em}}{}
  {\usebeamercolor[fg]{title}\Large\bfseries Variance Reduction for Training\\[0.2em]Neural Bayes Estimators\par}
  \vspace{1.4em}
  {\large Kai Marns-Morris\par}
  \vspace{1.0em}
  {\footnotesize Supervisors: Dr Michael Bertolacci \quad A/Prof Edward Cripps\par}
  \vspace{0.6em}
  {\footnotesize The University of Western Australia \,\textbullet\, Honours (BPhil) 2026\par}
\end{frame}

% ===== ACT 1: SETUP (frames 2-5 inserted here) =====

% ===== ACT 2: RAO-BLACKWELL CORE (frames 6-9 inserted here) =====

% >>> IS MODULE START (frames 10-11 — remove this whole block if tight for time) <<<
% >>> IS MODULE END <<<

% ===== ACT 3: CLOSE (frame 12 inserted here) =====

% ===== FRAME 13: THANKS =====
\begin{frame}[plain]
  \centering
  {\usebeamercolor[fg]{title}\Large\bfseries Thank you\par}
  \vspace{1em}
  {\large Questions?\par}
  \vspace{1.2em}
  {\footnotesize Supervisors: Dr Michael Bertolacci \& A/Prof Edward Cripps\par}
\end{frame}

% ===== BACKUP SLIDES (after Thanks; not counted in the 15 min) =====

\end{document}
```

- [ ] **Step 4: Run the structural check**

```bash
cd "C:/Users/kaima/OneDrive/Documents/GitHub/honours_project/presentation"
python tools/check_deck.py presentation.tex
```
Expected: `frames: 2 begin / 2 end` then `OK` (exit 0).

- [ ] **Step 5: Commit**

```bash
git add presentation/presentation.tex presentation/tools/check_deck.py presentation/figures/uwa.jpeg
git commit -m "presentation: scaffold beamer deck (theme, macros, title/thanks, checker)"
```

---

### Task 2: Figure preparation — copy assets + crop the RB hero panel

**Files:**
- Create: `presentation/tools/make_rb_hero.py`
- Create (copy): `presentation/figures/{rb_tau_curves,rb_tau_vs_samples,gibbs_vs_nn,ess_vs_alpha,grad_variance,normal_model_estimates,proposals,region_breakdown}.png`
- Create (derived): `presentation/figures/rb_hero.png`

**Interfaces:**
- Produces: `figures/rb_hero.png` — a single legible horizontal strip (the top row of `rb_tau_curves.png`: p=2 across batch sizes 4/16/64/256, showing MC/RB/floor with the embedded legend). Consumed by frame 9.

- [ ] **Step 1: Copy the needed figures**

```bash
cd "C:/Users/kaima/OneDrive/Documents/GitHub/honours_project/presentation"
for f in rb_tau_curves rb_tau_vs_samples gibbs_vs_nn ess_vs_alpha grad_variance normal_model_estimates proposals region_breakdown; do cp "../tex/figures/$f.png" "figures/$f.png"; done
ls figures/
```
Expected: all listed PNGs plus `uwa.jpeg` present.

- [ ] **Step 2: Write the crop script**

Create `presentation/tools/make_rb_hero.py`. The source is a 5-row (p=2,5,10,20,30) × 4-col (batch=4,16,64,256) grid with a suptitle, column titles, row labels, shared bottom x-label, and the legend embedded in the top-left panel. We crop a horizontal strip from just below the suptitle to just below row 0, full width, so the strip keeps the column titles, the four p=2 panels, the floor line, and the legend. Fractions are tuned in Step 3.

```python
#!/usr/bin/env python3
"""Crop the top row (p=2, all batch sizes) out of rb_tau_curves.png -> rb_hero.png."""
import sys
from PIL import Image

# vertical fractions of the full image to keep (tune via read-back)
TOP = 0.055   # just below the suptitle, keeping column titles
BOT = 0.265   # just below row 0 (p=2)

def main():
    src = Image.open('figures/rb_tau_curves.png')
    w, h = src.size
    box = (0, int(TOP * h), w, int(BOT * h))
    src.crop(box).save('figures/rb_hero.png')
    print(f'source {w}x{h} -> rb_hero.png crop {box}')

if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Run, then verify by reading the image; adjust TOP/BOT until clean**

```bash
cd "C:/Users/kaima/OneDrive/Documents/GitHub/honours_project/presentation"
python tools/make_rb_hero.py
```
Then Read `figures/rb_hero.png`. **Acceptance:** the strip shows the four `batch_size=` column titles, the four p=2 panels with MC (blue) above RB (orange) above the green floor, and the legend — no clipped titles, no row-1 panels bleeding in. If clipped, nudge `TOP`/`BOT` by ±0.02 and re-run. (If a horizontal strip proves unreadable at projector size, fall back to a single-panel crop of p=10/batch=16 — row index 2, col index 1 — by adding LEFT/RIGHT fractions; verify the same way.)

- [ ] **Step 4: Structural check still passes**

```bash
python tools/check_deck.py presentation.tex
```
Expected: `OK` (figures now resolve for later tasks).

- [ ] **Step 5: Commit**

```bash
git add presentation/tools/make_rb_hero.py presentation/figures/
git commit -m "presentation: add slide figures and cropped RB hero panel"
```

---

### Task 3: Act 1 — the setup (frames 2–5)

**Files:**
- Modify: `presentation/presentation.tex` (insert at `% ===== ACT 1: SETUP`)

**Interfaces:**
- Consumes: preamble macros from Task 1.
- Produces: frames 2–5. Frame 2's hook has a full and a short (IS-dropped) closing line; the short line is in a comment for the user to swap.

- [ ] **Step 1: Insert the four setup frames**

Replace the line `% ===== ACT 1: SETUP (frames 2-5 inserted here) =====` with:

```latex
% ===== ACT 1: SETUP =====

% --- FRAME 2: the talk in one breath (roadmap = hook) ---
\begin{frame}{The problem, in one breath}
  \begin{itemize}
    \item Bayesian inference usually means re-running a sampler \emph{from scratch} for every new dataset.
    \item[] \vspace{0.3em}
    \item \textbf{Neural Bayes estimators} avoid that: train once, then predict in a single forward pass.
    \item[] \vspace{0.3em}
    \item Training them is \emph{itself} a Monte Carlo computation --- so can classic \textbf{variance reduction} make training cheaper?
    \item[] \vspace{0.3em}
    \item[$\Rightarrow$] One technique comes with a \textbf{guarantee}; a second is more general but does not.
    % SHORT VERSION (if IS is cut): replace the line above with:
    % \item[$\Rightarrow$] One technique gives a \textbf{provable} reduction. That is today's story.
  \end{itemize}
\end{frame}

% --- FRAME 3: inference is expensive AND repeated ---
\begin{frame}{Bayesian inference is expensive --- and repeated}
  \begin{itemize}
    \item For each dataset $\by$ we want the posterior $p(\bt \given \by)$.
    \item The standard tool --- MCMC --- is accurate but must be \textbf{re-run from scratch} for every new $\by$.
  \end{itemize}
  \vspace{0.8em}
  \begin{center}
  \small
  $\by_1 \rightarrow$ \fbox{MCMC} $\rightarrow p(\bt\given\by_1)$ \qquad
  $\by_2 \rightarrow$ \fbox{MCMC} $\rightarrow p(\bt\given\by_2)$ \qquad
  $\cdots$ \\[0.4em]
  \textcolor{accent}{the same expensive computation, paid again every time}
  \end{center}
\end{frame}

% --- FRAME 4: neural Bayes estimators amortise it ---
\begin{frame}{Neural Bayes estimators: pay once}
  \begin{itemize}
    \item Under squared-error loss the best estimator is the \textbf{posterior mean} --- a fixed function of the data.
    \item Approximate that function \emph{once} with a neural network $\net(\by)$, trained on simulated pairs $(\bt,\by)$.
    \item Deployment for any new dataset is a \textbf{single forward pass}.
  \end{itemize}
  \vspace{0.8em}
  \begin{center}\small
  simulate $(\bt,\by)$ $\rightarrow$ \fbox{train $\net$ once} $\rightarrow$ new $\by \rightarrow$ \fbox{forward pass} $\rightarrow \hat{\bt}$
  \end{center}
\end{frame}

% --- FRAME 5: the pivot — training is Monte Carlo ---
\begin{frame}{The observation this thesis starts from}
  Training minimises the \textbf{Bayes risk}
  \[
    L(\bg) \;=\; \E_{\bt,\by}\!\big[\, \lVert \bt - \net(\by) \rVert^2 \,\big].
  \]
  \begin{itemize}
    \item In practice this expectation is a \textbf{Monte Carlo average} over simulated draws.
    \item So the loss --- and the gradient that drives training --- is a \textbf{random quantity with variance}.
    \item That variance comes from \emph{how we sample the loss}, not from the inference problem. \textcolor{accent}{So we can reduce it.}
  \end{itemize}
\end{frame}
```

- [ ] **Step 2: Structural check**

```bash
cd "C:/Users/kaima/OneDrive/Documents/GitHub/honours_project/presentation"
python tools/check_deck.py presentation.tex
```
Expected: `frames: 6 begin / 6 end`, `OK`.

- [ ] **Step 3: Commit**

```bash
git add presentation/presentation.tex
git commit -m "presentation: act 1 setup frames (problem, NBEs, the pivot)"
```

---

### Task 4: Act 2 — the Rao–Blackwell core (frames 6–9), including the peak

**Files:**
- Modify: `presentation/presentation.tex` (insert at `% ===== ACT 2`)

**Interfaces:**
- Consumes: `figures/rb_hero.png` (Task 2), macros (Task 1).
- Produces: frames 6–9. Frame 8 is the guarantee (peak); frame 9 is the empirical payoff.

- [ ] **Step 1: Insert the four RB frames**

Replace `% ===== ACT 2: RAO-BLACKWELL CORE (frames 6-9 inserted here) =====` with:

```latex
% ===== ACT 2: RAO-BLACKWELL CORE =====

% --- FRAME 6: a trick for killing Monte Carlo noise (intro the intuition) ---
\begin{frame}{A trick for killing Monte Carlo noise}
  \begin{itemize}
    \item You are estimating an average by sampling. Suppose part of it you can compute \textbf{exactly}.
    \item Then replace that sampled piece by its \textbf{exact conditional mean} instead of drawing it.
    \item Averaging out a random draw can only \emph{remove} variance --- never add it.
  \end{itemize}
  \vspace{0.6em}
  \begin{center}\small
  noisy draws \; $\rightarrow$ \; replace one coordinate by its exact mean \; $\rightarrow$ \; \textcolor{accent}{tighter estimate}
  \end{center}
  \vspace{0.4em}
  \footnotesize Statisticians call this \textbf{Rao--Blackwellisation}.
\end{frame}

% --- FRAME 7: the Rao-Blackwellised loss ---
\begin{frame}{The Rao--Blackwellised loss}
  Split the parameters $\bt = (\tau, \bb)$. Suppose $\bb$ has a \textbf{tractable conditional posterior} given $(\tau,\by)$ (here: conjugate).
  \[
    \text{sampled target } \bb
    \;\;\longrightarrow\;\;
    \E[\,\bb \given \tau, \by\,] \quad(\text{its conditional posterior mean})
  \]
  \begin{itemize}
    \item We integrate $\bb$ out of the loss analytically instead of sampling it.
    \item The leftover conditional variance is a \textbf{constant} --- it drops out of the gradient.
  \end{itemize}
\end{frame}

% --- FRAME 8: THE GUARANTEE (peak) ---
\begin{frame}{The guarantee}
  The Rao--Blackwellised gradient is a \textbf{conditional average} of the standard one. The law of total variance then gives
  \[
    \Var(\nabla_{\RB}) \;\preceq\; \Var(\nabla_{\MC})
    \qquad(\text{positive semi-definite gap}).
  \]
  \begin{itemize}
    \item It can \textbf{never} increase the gradient variance --- for \emph{any} model, \emph{any} architecture.
    \item For a linear network the same reduction carries through to the \textbf{fitted weights} $\hat{\bg}$ themselves.
  \end{itemize}
  \vspace{0.4em}
  \begin{center}\textcolor{accent}{\bfseries For free, provably: never worse, usually better.}\end{center}
\end{frame}

% --- FRAME 9: does it pay off? (empirical payoff) ---
\begin{frame}{Does it pay off? Bayesian linear regression}
  \begin{columns}[T]
    \begin{column}{0.62\textwidth}
      \includegraphics[width=\textwidth]{rb_hero.png}\\
      {\scriptsize MC (blue) vs RB (orange) vs Bayes floor $L^*$ (green), by batch size.}
    \end{column}
    \begin{column}{0.36\textwidth}
      \small Integrate out $\bb$; Gibbs gives the Bayes-optimal floor $L^*$.
      \vspace{0.6em}
      \begin{itemize}
        \item lower loss in \textbf{every} grid cell
        \item closes \textbf{$\sim$half} ($52\%$) the gap to $L^*$
        \item matches the standard NBE at \textbf{$\tfrac14$ the batch size}
      \end{itemize}
    \end{column}
  \end{columns}
\end{frame}
```

- [ ] **Step 2: Structural check** (verifies `rb_hero.png` resolves)

```bash
cd "C:/Users/kaima/OneDrive/Documents/GitHub/honours_project/presentation"
python tools/check_deck.py presentation.tex
```
Expected: `frames: 10 begin / 10 end`, `OK`.

- [ ] **Step 3: Commit**

```bash
git add presentation/presentation.tex
git commit -m "presentation: act 2 Rao-Blackwell core (trick, RB loss, guarantee, payoff)"
```

---

### Task 5: The importance-sampling module (frames 10–11) — fenced & liftable

**Files:**
- Modify: `presentation/presentation.tex` (between the `% >>> IS MODULE START` / `% END` markers)

**Interfaces:**
- Consumes: `figures/gibbs_vs_nn.png`, `figures/ess_vs_alpha.png`.
- Produces: frames 10–11 inside the fence. Removing everything between the two markers must leave a valid deck (checker still passes, frame count drops by 2).

- [ ] **Step 1: Insert the two IS frames between the fence markers**

Replace the two fence-marker lines with:

```latex
% >>> IS MODULE START (frames 10-11 — remove this whole block if tight for time) <<<

% --- FRAME 10: the contrast — importance sampling ---
\begin{frame}{A contrasting lever: importance sampling}
  No conjugacy needed --- reshape the simulation distribution, then reweight back to the same objective. \textcolor{accent}{The price: no guarantee.}
  \vspace{0.4em}
  \begin{columns}[T]
    \begin{column}{0.55\textwidth}
      \includegraphics[width=\textwidth]{gibbs_vs_nn.png}\\
      {\scriptsize AR(1)+GRU already tracks the Gibbs (Bayes) reference across the prior.}
    \end{column}
    \begin{column}{0.43\textwidth}
      \includegraphics[width=\textwidth]{ess_vs_alpha.png}\\
      {\scriptsize Push the proposal hard and the effective sample size collapses.}
    \end{column}
  \end{columns}
  \vspace{0.3em}
  \footnotesize Little error left to recover, and aggressive reweighting only adds noise.
\end{frame}

% --- FRAME 11: what separates them ---
\begin{frame}{What separates the two techniques}
  \begin{center}
  \renewcommand{\arraystretch}{1.3}
  \begin{tabular}{lcc}
    \toprule
     & \textbf{Rao--Blackwell} & \textbf{Importance sampling} \\
    \midrule
    needs & a conjugate block & nothing special \\
    variance guarantee & \textbf{yes, always} & none \\
    carries to fitted weights & \textbf{yes} (linear net) & no \\
    \bottomrule
  \end{tabular}
  \end{center}
  \vspace{0.8em}
  \begin{center}\textcolor{accent}{Where conjugacy exists, Rao--Blackwell is the safer, stronger lever.}\end{center}
\end{frame}

% >>> IS MODULE END <<<
```

- [ ] **Step 2: Structural check (full deck)**

```bash
cd "C:/Users/kaima/OneDrive/Documents/GitHub/honours_project/presentation"
python tools/check_deck.py presentation.tex
```
Expected: `frames: 12 begin / 12 end`, `OK`.

- [ ] **Step 3: Verify the module is cleanly liftable**

```bash
python - <<'PY'
import re, pathlib
tex = pathlib.Path('presentation.tex').read_text(encoding='utf-8')
cut = re.sub(r'% >>> IS MODULE START.*?% >>> IS MODULE END <<<', '', tex, flags=re.S)
nb, ne = len(re.findall(r'\\begin\{frame\}', cut)), len(re.findall(r'\\end\{frame\}', cut))
print('after cut:', nb, 'begin /', ne, 'end')
assert nb == ne == 10, 'cut left an unbalanced deck'
print('OK: IS module removes cleanly, 10 frames remain')
PY
```
Expected: `OK: IS module removes cleanly, 10 frames remain`.

- [ ] **Step 4: Commit**

```bash
git add presentation/presentation.tex
git commit -m "presentation: fenced liftable importance-sampling module (frames 10-11)"
```

---

### Task 6: Act 3 close (frame 12) + backup slides

**Files:**
- Modify: `presentation/presentation.tex` (insert at `% ===== ACT 3` and `% ===== BACKUP SLIDES`)

**Interfaces:**
- Consumes: macros; backup figures `rb_tau_curves.png`, `rb_tau_vs_samples.png`, `grad_variance.png`, `proposals.png`, `region_breakdown.png`.
- Produces: frame 12 (takeaway) + backup frames after Thanks.

- [ ] **Step 1: Insert the takeaway frame**

Replace `% ===== ACT 3: CLOSE (frame 12 inserted here) =====` with:

```latex
% ===== ACT 3: CLOSE =====

% --- FRAME 12: takeaway ---
\begin{frame}{Takeaway}
  \begin{itemize}
    \item Training a neural Bayes estimator is Monte Carlo --- so the Monte Carlo variance-reduction toolbox applies to \emph{training}.
    \item \textbf{Rao--Blackwellisation} is a cheap, reliable lever: a \emph{provable} cut in gradient variance, and real gains toward the Bayes floor.
  \end{itemize}
  \vspace{1.0em}
  \begin{center}\usebeamercolor[fg]{title}\bfseries
  Conjugacy --- the classical trick behind Gibbs sampling ---\\
  is just as useful for training the estimators meant to replace it.
  \end{center}
\end{frame}
```

- [ ] **Step 2: Insert backup slides**

Replace `% ===== BACKUP SLIDES (after Thanks; not counted in the 15 min) =====` with that banner followed by:

```latex
% ===== BACKUP SLIDES (after Thanks; not counted in the 15 min) =====

\begin{frame}{Backup: why the variance can only shrink}
  Write the standard gradient $g$ and condition on $(\tau,\by)$. By the law of total variance,
  \[
    \Var(g) = \underbrace{\Var\big(\E[g\given\tau,\by]\big)}_{\Var(\nabla_{\RB})} + \underbrace{\E\big[\Var(g\given\tau,\by)\big]}_{\succeq\,0}.
  \]
  The second term is positive semi-definite, so $\Var(\nabla_{\RB}) \preceq \Var(g)$.
\end{frame}

\begin{frame}{Backup: RB vs MC across the whole grid}
  \begin{center}\includegraphics[height=0.82\textheight]{rb_tau_curves.png}\end{center}
\end{frame}

\begin{frame}{Backup: convergence vs total samples seen}
  \begin{center}\includegraphics[height=0.82\textheight]{rb_tau_vs_samples.png}\end{center}
\end{frame}

\begin{frame}{Backup: importance sampling --- gradient variance \& proposals}
  \begin{columns}[c]
    \begin{column}{0.5\textwidth}\includegraphics[width=\textwidth]{grad_variance.png}\end{column}
    \begin{column}{0.5\textwidth}\includegraphics[width=\textwidth]{region_breakdown.png}\end{column}
  \end{columns}
  \vspace{0.3em}
  {\scriptsize Left: per-minibatch gradient variance blows up as the proposal is pushed away from uniform. Right: the error lives at the boundary, which the proposals do not target.}
\end{frame}

\begin{frame}{Backup: models \& architectures}
  \begin{itemize}
    \item \textbf{RB study:} Bayesian linear regression, $\bt=(\tau,\bb)$; $\bb$ (coefficients) integrated out, $\tau\sim\mathrm{U}(0.01,1)$ sampled; multilayer perceptron; Gibbs sampler for the Bayes floor $L^*$.
    \item \textbf{IS study:} AR(1) process, $T=100$; parameters $(\rho,\sigma)$; recurrent (GRU) network; $\mathrm{Beta}(\alpha,\alpha)$ proposal family on $\rho$.
  \end{itemize}
\end{frame}
```

- [ ] **Step 3: Structural check**

```bash
cd "C:/Users/kaima/OneDrive/Documents/GitHub/honours_project/presentation"
python tools/check_deck.py presentation.tex
```
Expected: `frames: 18 begin / 18 end`, `OK` (13 main + 5 backup).

- [ ] **Step 4: Commit**

```bash
git add presentation/presentation.tex
git commit -m "presentation: takeaway frame + Q&A backup slides"
```

---

### Task 7: Speaker script + timing budget

**Files:**
- Create: `presentation/SPEAKER_SCRIPT.md`

**Interfaces:**
- Consumes: the final frame list.
- Produces: a rehearsal script with per-slide talking points, a running time budget summing to ~13:30, the `CUT POINT`, and full/short hook variants.

- [ ] **Step 1: Write the speaker script**

Create `presentation/SPEAKER_SCRIPT.md` with one section per frame. Each section: **cumulative time target**, 2–4 spoken talking points (full sentences, plain language), and any delivery note. Include verbatim:
- Frame 2 hook — **full**: "...one comes with a guarantee, the other is more general but doesn't." **short (IS cut)**: "...this one comes with a provable guarantee — that's today's story."
- After Frame 9: a bold `>>> CUT POINT — if running long, skip frames 10–11 and go straight to the Takeaway with:` followed by the bridge line: *"Rao–Blackwellisation isn't the only lever — there's a more general one, importance sampling, that I'm happy to discuss in questions. But the guarantee is the heart of it, so let me close."*
- A timing table at the top:

```
| Frame | Slide                         | Target (cum) |
|-------|-------------------------------|--------------|
| 1     | Title                         | 0:20         |
| 2     | Problem in one breath         | 1:00         |
| 3     | Expensive and repeated        | 2:00         |
| 4     | NBEs: pay once                | 3:30         |
| 5     | The pivot                     | 5:00         |
| 6     | Trick for killing noise       | 5:50         |
| 7     | The RB loss                   | 7:10         |
| 8     | The guarantee (peak)          | 8:20         |
| 9     | Does it pay off?              | 10:20        |
| 10    | IS contrast     [CUTTABLE]    | 11:50        |
| 11    | What separates them [CUTTABLE]| 12:50        |
| 12    | Takeaway                      | 13:40        |
| 13    | Thanks / Q&A                  | 13:50        |
```
Note under the table: "Without frames 10–11: Takeaway lands ~11:10. Hard cap 15:00."

- [ ] **Step 2: Sanity-check the timing sums**

```bash
cd "C:/Users/kaima/OneDrive/Documents/GitHub/honours_project/presentation"
python - <<'PY'
full = 13*60+50
cut = full - (90+60)  # frames 10,11
print(f'full ~{full//60}:{full%60:02d}  | IS-cut ~{cut//60}:{cut%60:02d}')
assert full < 15*60 and cut < 15*60
print('both under hard cap 15:00')
PY
```
Expected: `full ~13:50 | IS-cut ~11:20`, `both under hard cap 15:00`.

- [ ] **Step 3: Commit**

```bash
git add presentation/SPEAKER_SCRIPT.md
git commit -m "presentation: speaker script with timing budget and IS cut point"
```

---

### Task 8: Final assembly — README, compile instructions, QA checklist

**Files:**
- Create: `presentation/README.md`

**Interfaces:**
- Consumes: the whole deck.
- Produces: compile instructions (Overleaf + local) and a visual QA checklist for the user's first compile.

- [ ] **Step 1: Write the README**

Create `presentation/README.md` covering:
- **What this is** and the file map.
- **Compile — Overleaf:** upload `presentation.tex` + `figures/`; menu → compiler → *pdfLaTeX* (or *XeLaTeX* if switching to metropolis); compile twice.
- **Compile — local:** `pdflatex presentation.tex` twice (or `latexmk -pdf presentation.tex`). Note MiKTeX will offer to auto-install any missing packages — accept.
- **Optional metropolis upgrade:** comment the custom-theme block, uncomment `\usetheme{metropolis}`, compile with XeLaTeX.
- **To cut importance sampling:** delete everything between `% >>> IS MODULE START` and `% >>> IS MODULE END`, and swap the Frame 2 hook to the short line (commented in place).
- **Visual QA checklist (first compile):** 13 main frames + 5 backup; nothing overflows the slide; `rb_hero.png` legend/floor readable at projector distance; the guarantee inequality renders; timing rehearsed against `SPEAKER_SCRIPT.md`.

- [ ] **Step 2: Final structural check**

```bash
cd "C:/Users/kaima/OneDrive/Documents/GitHub/honours_project/presentation"
python tools/check_deck.py presentation.tex
```
Expected: `frames: 18 begin / 18 end`, `OK`.

- [ ] **Step 3: Commit**

```bash
git add presentation/README.md
git commit -m "presentation: README with compile instructions and QA checklist"
```

- [ ] **Step 4: Hand off for compile**

Tell the user the deck is complete, point them at `README.md` to compile, and offer to adjust any slide after they eyeball the rendered PDF.

---

## Self-Review

**Spec coverage:** Every spec section maps to a task — narrative arc → Tasks 3–6; RB star (peak) → Task 4 frames 8–9; IS liftable module → Task 5 (+ verified cut in Step 3); depth/one-equation rule → enforced in frame content; backup slides (§8) → Task 6; speaker script + timing + CUT POINT (§9) → Task 7; Beamer portability & optional metropolis (§10) → Task 1 + README; figure verification (§11) → Task 2 (read-back) and the corrected figure assignments above. Timing (§2) → Task 7 table (~13:50 full / ~11:20 cut, both < 15:00).

**Placeholder scan:** No TBD/TODO; every frame's actual text, equation, and figure is specified; the crop fractions have concrete starting values with a read-back tuning loop.

**Type/name consistency:** macros `\net,\bt,\bb,\by,\bg,\RB,\MC` defined in Task 1 and used consistently in Tasks 3–6; `figures/rb_hero.png` produced in Task 2 and consumed in Task 4; IS fence markers written in Task 1, filled in Task 5, and cut-tested against them. `check_deck.py` interface (`python tools/check_deck.py presentation.tex`) identical across all tasks.

**Note on verification:** because no LaTeX toolchain is installed, "tests" are the structural checker + figure read-back + user compile checkpoints (see *Verification philosophy*), not a LaTeX build. This is the correct adaptation given the user compiles themselves.
