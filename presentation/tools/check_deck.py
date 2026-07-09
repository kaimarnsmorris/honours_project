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
