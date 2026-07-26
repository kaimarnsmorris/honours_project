import json, copy, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "4_rb_vs_mc_longrun.ipynb")

with open(SRC, "r", encoding="utf-8") as f:
    base = json.load(f)


def cell_by_id(nb, idx):
    # longrun notebook cells carry no id field; address by index
    return nb["cells"][idx]


def get_src(cell):
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def repl(cell, old, new, n=1):
    t = get_src(cell)
    assert t.count(old) == n, f"expected {n} of {old!r}, found {t.count(old)}"
    cell["source"] = t.replace(old, new)


def clear_outputs(nb):
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None


# --- setup-cell (cell-3) noise-range definitions ---------------------------
SETUP_OLD = ("sigma_low  = 0.1     # noise std range -- a low floor keeps some "
             "*informative* datasets in the mix\nsigma_high = 10.0")
SETUP_UNIFORM = ("tau_low    = 0.01    # precision range; sigma = 1/sqrt(tau) "
                 "gives sigma in (1, 10)\ntau_high   = 1.0")
SETUP_GAMMA = ("gamma_shape = 3.0     # tau ~ Gamma(shape, rate); moments matched "
               "to U(0.01, 1)\ngamma_rate  = 6.0")

# --- sample_batch sigma line (cell-4) --------------------------------------
SB_OLD = ("    sigma = torch.rand(batch_size, device=device) "
          "* (sigma_high - sigma_low) + sigma_low")
SB_UNIFORM = ("    tau   = torch.rand(batch_size, device=device) "
              "* (tau_high - tau_low) + tau_low\n"
              "    sigma = 1.0 / torch.sqrt(tau)")
SB_GAMMA = ("    tau   = torch.distributions.Gamma(gamma_shape, gamma_rate)"
            ".sample((batch_size,)).to(device)\n"
            "    sigma = 1.0 / torch.sqrt(tau)")

# --- markdown (cell-0) prior line ------------------------------------------
MD_OLD = "\\sigma \\sim \\mathcal U(\\sigma_{\\text{low}},\\sigma_{\\text{high}}),\\qquad"
MD_UNIFORM = "\\tau = 1/\\sigma^2 \\sim \\mathcal U(0.01, 1),\\qquad"
MD_GAMMA = "\\tau = 1/\\sigma^2 \\sim \\text{Gamma}(3, 6),\\qquad"

NOTE_UNIFORM = ("\n\n*Variant:* noise precision is drawn $\\tau\\sim\\mathcal U(0.01,1)$ "
                "and $\\sigma=1/\\sqrt{\\tau}\\in(1,10)$, replacing the original "
                "$\\sigma\\sim\\mathcal U(0.1,10)$.")
NOTE_GAMMA = ("\n\n*Variant:* noise precision is drawn from a **conjugate** "
              "$\\tau\\sim\\text{Gamma}(3,6)$ (first two moments matched to "
              "$\\mathcal U(0.01,1)$), with $\\sigma=1/\\sqrt{\\tau}$. Comparing against the "
              "uniform-$\\tau$ variant shows whether a realistic conjugate prior reproduces the "
              "same RB-vs-MC behaviour.")


def build(variant):
    nb = copy.deepcopy(base)
    clear_outputs(nb)

    md = cell_by_id(nb, 0)
    repl(md, MD_OLD, MD_UNIFORM if variant == "uniform" else MD_GAMMA)
    md["source"] = get_src(md) + (NOTE_UNIFORM if variant == "uniform" else NOTE_GAMMA)

    setup = cell_by_id(nb, 3)
    repl(setup, SETUP_OLD, SETUP_UNIFORM if variant == "uniform" else SETUP_GAMMA)

    sb = cell_by_id(nb, 4)
    repl(sb, SB_OLD, SB_UNIFORM if variant == "uniform" else SB_GAMMA)

    out = os.path.join(HERE, f"4_rb_vs_mc_longrun_tau_{variant}.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", out)


build("uniform")
build("gamma")
