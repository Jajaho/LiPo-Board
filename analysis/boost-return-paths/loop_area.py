# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "scipy>=1.13", "pillow>=10", "contourpy>=1.2"]
# ///
"""Effective area of the switching ("hot") loop.

The area is the loop's magnetic moment divided by its current: the forward path on the traces
(as line currents) plus the solved return current in the ground copper of both layers.
Two hand-drawn references are computed with the shoelace formula for comparison.
"""
import json

import numpy as np

import board as B

# SW pin -> L1 pad -> D1 -> C3 (through the capacitor to its ground pad), from the KiCad tracks
FORWARD = [(157.866, 104.079), (156.642, 104.079), (155.052, 102.489), (155.052, 99.568), (155.941, 98.679),
           (157.091, 98.679), (161.141, 98.679), (161.731, 99.269), (164.831, 99.269), (164.831, 103.169)]
INSIDE_IC = [(157.866, 105.029), (157.866, 104.079)]
DIRECT = FORWARD + INSIDE_IC
TIGHTEST = FORWARD + [(166.4, 104.6), (166.4, 108.9), (164.2, 111.38), (156.576, 105.029)] + INSIDE_IC


def seg_moment(pts, I=1.0):
    return 0.5 * I * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]))


def field_moment(V):
    h = 1 / B.S
    m = 0.0
    for L in 'FB':
        v = V[L]
        ys = B.Y0 + np.arange(v.shape[0])[:, None] / B.S
        xs = B.X0 + np.arange(v.shape[1])[None, :] / B.S
        ix = (v[:, :-1] - v[:, 1:]) / B.RS          # current in +x across each vertical cell edge
        ok = ~np.isnan(ix)
        m += 0.5 * np.sum(-np.broadcast_to(ys, ix.shape)[ok] * ix[ok] * h)
        iy = (v[:-1, :] - v[1:, :]) / B.RS          # current in +y across each horizontal cell edge
        ok = ~np.isnan(iy)
        m += 0.5 * np.sum(np.broadcast_to(xs, iy.shape)[ok] * iy[ok] * h)
    return m


def main():
    out = {}
    for src in ('gerber', 'kicad'):
        d = np.load(B.work_path(f'V_{src}_hot.npz'))
        m = seg_moment(FORWARD) + field_moment({'F': d['F'], 'B': d['B']}) + seg_moment(INSIDE_IC)
        out[f'{src}_spread'] = round(float(abs(m)), 1)
    out['direct_top'] = round(abs(seg_moment(DIRECT)), 1)
    out['tightest_route'] = round(abs(seg_moment(TIGHTEST)), 1)
    B.work_path('loop_area.json').write_text(json.dumps(out, indent=1))
    print('hot-loop area [mm2]:', out)


if __name__ == '__main__':
    main()
