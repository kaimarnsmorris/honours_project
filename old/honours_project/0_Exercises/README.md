
# Set up procedure

The easiest way is to use a conda environment:

```bash
conda create -n honours_toy_problems python=3.11
conda activate honours_toy_problems
conda install -c conda-forge notebook ipykernel numpy matplotlib pytorch
```

You can try install cmdstanpy via conda too, though it didn't work for me:

```bash
conda install -c conda-forge cmdstanpy
```

Instead, install cmdstanpy via pip:

```bash
pip install cmdstanpy
```

If you use python you then need to run `python` and then (this may take a while):

```python
import cmdstanpy
cmdstanpy.install_cmdstan()
```

(You may need to check the documentation for CmdStanPy at https://mc-stan.org/cmdstanpy/installation.html)

Then launch jupyter notebook (or use it within VSCode if that's what you use):

```bash
jupyter notebook
```




