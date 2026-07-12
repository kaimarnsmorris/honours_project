#!/usr/bin/env python3
"""Crop the p=10 row (all batch sizes) out of rb_tau_curves.png -> rb_hero.png.

The source is a 5x4 grid: rows p = 2, 5, 10, 20, 30; cols batch = 4, 16, 64, 256.
Row bands (detected): p2 .060-.232, p5 .243-.415, p10 .425-.598, p20 .607-.780, p30 .790-.963.
"""
import sys
from PIL import Image

# vertical fractions of the full image to keep (p=10 band, small pad into the gaps)
TOP = 0.421   # a hair above the p=10 panel tops
BOT = 0.601   # cut in the white gap above the p=20 row so no sliver bleeds in

def main():
    src = Image.open('figures/rb_tau_curves.png')
    w, h = src.size
    box = (0, int(TOP * h), w, int(BOT * h))
    src.crop(box).save('figures/rb_hero.png')
    print(f'source {w}x{h} -> rb_hero.png crop {box}')

if __name__ == '__main__':
    main()
