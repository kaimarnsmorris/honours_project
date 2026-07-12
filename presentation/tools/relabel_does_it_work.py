"""Relabel MH -> MCMC on the exact original figure, preserving the scatter."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm

fp = fm.findfont('DejaVu Sans')
im = Image.open('figures/normal_model_estimates.png').convert('RGB')
W, H = im.size
g = np.asarray(im.convert('L'))
rgb = np.asarray(im).astype(int)
d = ImageDraw.Draw(im)
INK = (20, 20, 20)

# ---------------- TITLES: line 2 measured at y 110..168 ----------------
y0, y1 = 110, 168
fs = 74
font = ImageFont.truetype(fp, fs)
titles = ['MCMC MSE: 0.0950,  NN MSE: 0.0974', 'MCMC MSE: 0.0525,  NN MSE: 0.0747']
for (a, b), txt in zip([(0, W // 2), (W // 2, W)], titles):
    sub = g[y0:y1, a:b]
    cols = np.where((sub < 120).any(axis=0))[0]
    cx = a + (cols.min() + cols.max()) // 2                 # true text centre in this panel
    d.rectangle([a + int(0.02 * (b - a)), y0 - 16, b - int(0.02 * (b - a)), y1 + 16], fill=(255, 255, 255))
    bb = d.textbbox((0, 0), txt, font=font)
    d.text((cx - (bb[2] - bb[0]) // 2, y0 - bb[1] - 2), txt, fill=INK, font=font)

# ---------------- LEGEND: 'MH' text (dark, top legend row) -> 'MCMC' ----------------
dark = g < 120
for a in (0, W // 2):
    mask = np.zeros_like(dark)
    mask[240:322, a + 420:a + 980] = True                   # top legend row, right of the marker
    ys_, xs_ = np.where(dark & mask)
    if len(ys_) == 0:
        print('half', a, 'no MH text'); continue
    tx0, tx1 = int(xs_.min()), int(xs_.max())
    ty0, ty1 = int(ys_.min()), int(ys_.max())
    cap = ty1 - ty0
    fs = int(round(cap / 0.70))                             # DejaVu cap-height ~0.70*fontsize
    lf = ImageFont.truetype(fp, fs)
    print('half', a, 'MH box', tx0, tx1, ty0, ty1, 'cap', cap, 'fs', fs)
    d.rectangle([tx0 - 8, ty0 - 8, tx0 + int(3.0 * cap), ty1 + 8], fill=(255, 255, 255))  # erase 'MH'
    asc = d.textbbox((0, 0), 'MCMC', font=lf)[1]
    d.text((tx0, ty0 - asc), 'MCMC', fill=INK, font=lf)     # top-aligned to original 'MH'

im.save('figures/does_it_work.png')
print('saved')
