#!/usr/bin/env python3
"""Crop the top row (p=2, all batch sizes) out of rb_tau_curves.png -> rb_hero.png."""
import sys
from PIL import Image

# vertical fractions of the full image to keep (tuned via read-back)
TOP = 0.070   # start at the p=2 panel tops (batch sizes named in the caption)
BOT = 0.242   # cut in the white gap above the p=5 row so no sliver bleeds in

def main():
    src = Image.open('figures/rb_tau_curves.png')
    w, h = src.size
    box = (0, int(TOP * h), w, int(BOT * h))
    src.crop(box).save('figures/rb_hero.png')
    print(f'source {w}x{h} -> rb_hero.png crop {box}')

if __name__ == '__main__':
    main()
