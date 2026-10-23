# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "scipy>=1.13", "pillow>=10", "contourpy>=1.2"]
# ///
"""Build the interactive page: geometry, heatmaps and flow fields, merged into template.html.

Uses the as-fabricated (Gerber) solve. Writes work/viz_data.json and boost-return-paths.html.
"""
import base64
import gzip
import io
import json
import math

import contourpy
import numpy as np
from PIL import Image

import board as B
from board import S, X0, Y0, H, W, RS, find, find1

DECADES = 2.1           # dynamic range of the current-density heatmaps
G2 = 2                  # particle field cell = G2 raster cells (0.1 mm)

ROLE = {'N$1': 'sw', '5V': 'vout', 'BAT+': 'vin', 'BAT-': 'batn', 'N$6': 'fb', 'N$8': 'en', 'GND': 'gnd', 'VBUS': 'vbus'}


def r2(v):
    return round(float(v), 2)


def rdp(pts, eps):
    if len(pts) < 4:
        return pts
    keep = np.zeros(len(pts), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        a, b = stack.pop()
        if b <= a + 1:
            continue
        p, q = pts[a], pts[b]
        seg = q - p
        L = np.hypot(*seg)
        sub = pts[a + 1:b]
        if L < 1e-12:
            d = np.hypot(*(sub - p).T)
        else:
            d = np.abs(seg[0] * (sub[:, 1] - p[1]) - seg[1] * (sub[:, 0] - p[0])) / L
        k = int(np.argmax(d))
        if d[k] > eps:
            m = a + 1 + k
            keep[m] = True
            stack += [(a, m), (m, b)]
    return pts[keep]


def mask_path(mask, eps=0.012):
    z = np.pad(mask.astype(float), 1)
    xs = X0 + (np.arange(z.shape[1]) - 1) / S
    ys = Y0 + (np.arange(z.shape[0]) - 1) / S
    gen = contourpy.contour_generator(x=xs, y=ys, z=z)
    parts = []
    for line in gen.lines(0.5):
        if len(line) < 4:
            continue
        line = rdp(np.asarray(line), eps)
        parts.append('M' + ' '.join(f'{r2(x)} {r2(y)}' for x, y in line) + 'Z')
    return ''.join(parts)


def iso_paths(field, levels, eps=0.02):
    z = np.nan_to_num(field, nan=-1.0)
    xs = X0 + np.arange(z.shape[1]) / S
    ys = Y0 + np.arange(z.shape[0]) / S
    gen = contourpy.contour_generator(x=xs, y=ys, z=z)
    out = []
    for lv in levels:
        d = ''
        for line in gen.lines(lv):
            if len(line) < 6:
                continue
            line = rdp(np.asarray(line), eps)
            d += 'M' + ' '.join(f'{r2(x)} {r2(y)}' for x, y in line)
        out.append({'level': lv, 'd': d})
    return out


def jfield(V):
    ex = np.nan_to_num((V[:, :-1] - V[:, 1:]) / RS)
    ey = np.nan_to_num((V[:-1, :] - V[1:, :]) / RS)
    Jx = np.zeros(V.shape)
    Jy = np.zeros(V.shape)
    Jx[:, :-1] += ex / 2; Jx[:, 1:] += ex / 2
    Jy[:-1, :] += ey / 2; Jy[1:, :] += ey / 2
    Jx *= S; Jy *= S                         # A per mm of copper width
    Jx[np.isnan(V)] = 0; Jy[np.isnan(V)] = 0
    return Jx, Jy


def gray_uri(t):
    """Scalar map 0..1 as an 8-bit grey PNG; the page colours it with the ramp of the chosen figure style."""
    img = np.round(np.clip(np.nan_to_num(t), 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img, 'L').save(buf, 'PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def load_V(scenario):
    d = np.load(B.work_path(f'V_gerber_{scenario}.npz'))
    return {'F': d['F'], 'B': d['B']}


AC = ('hot', 'input')       # scenarios the AC page draws from the AC (R + jwL) solve


def load_J(scenario, gnd, ac=False):
    """Ground-copper current per layer: (Jx, Jy, |J|, copper) in A/mm per 1 A.

    With ac, hot and input come from solve_ac.py: |J| is the amplitude, Jx/Jy the real part, i.e. the
    flow at the instant the 1 A source peaks. Otherwise (and for dc) from solve.py."""
    path = B.work_path(f'Jac_gerber_{scenario}.npz')
    out = {}
    if ac and scenario in AC:
        d = np.load(path)
        for L in 'FB':
            jx, jy = np.nan_to_num(d[L + 'X']), np.nan_to_num(d[L + 'Y'])
            cu = gnd[L]
            mag = np.where(cu, np.sqrt(abs(jx) ** 2 + abs(jy) ** 2), 0)
            out[L] = (np.where(cu, jx.real, 0), np.where(cu, jy.real, 0), mag, cu)
        return out
    V = load_V(scenario)
    for L in 'FB':
        jx, jy = jfield(V[L])
        out[L] = (jx, jy, np.hypot(jx, jy), ~np.isnan(V[L]))
    return out


def via_currents(scenario, res, ac=False):
    """Current changing layer at each ground via / plated pin, + = top -> bottom."""
    if ac and scenario in AC:
        sol = json.loads(B.work_path('solve_ac_gerber.json').read_text())
        return [(x, y, re) for x, y, net, re, im in sol[scenario]['vias'] if net == 'GND']
    return res[scenario + '_vias']


def fab_outlines():
    """Component body outlines from the F.Fab layer, in board coordinates."""
    fab = []
    for fp in find(B.read_pcb_tree(), 'footprint'):
        at = find1(fp, 'at')
        fx, fy = float(at[1]), float(at[2])
        t = math.radians(float(at[3]) if len(at) > 3 else 0)

        def tf(x, y):
            x, y = float(x), float(y)
            return (r2(fx + x * math.cos(t) + y * math.sin(t)), r2(fy - x * math.sin(t) + y * math.cos(t)))

        for g in fp:
            if not isinstance(g, list) or not g or not str(g[0]).startswith('fp_'):
                continue
            layer = find1(g, 'layer')
            if not layer or layer[1] != 'F.Fab':
                continue
            if g[0] == 'fp_line':
                a = tf(*find1(g, 'start')[1:3]); b = tf(*find1(g, 'end')[1:3])
                fab.append(f'M{a[0]} {a[1]}L{b[0]} {b[1]}')
            elif g[0] == 'fp_poly':
                pts = [tf(p[1], p[2]) for p in find1(g, 'pts')[1:] if p[0] == 'xy']
                fab.append('M' + 'L'.join(f'{x} {y}' for x, y in pts) + 'Z')
            elif g[0] == 'fp_circle':
                c = tf(*find1(g, 'center')[1:3]); e = tf(*find1(g, 'end')[1:3])
                r = math.hypot(e[0] - c[0], e[1] - c[1])
                fab.append(f'M{r2(c[0] - r)} {c[1]}a{r2(r)} {r2(r)} 0 1 0 {r2(2 * r)} 0a{r2(r)} {r2(r)} 0 1 0 {r2(-2 * r)} 0')
            elif g[0] == 'fp_arc':
                a = tf(*find1(g, 'start')[1:3]); m = tf(*find1(g, 'mid')[1:3]); b = tf(*find1(g, 'end')[1:3])
                fab.append(f'M{a[0]} {a[1]}Q{r2(2 * m[0] - (a[0] + b[0]) / 2)} {r2(2 * m[1] - (a[1] + b[1]) / 2)} {b[0]} {b[1]}')
    return ''.join(fab)


def build_payload(ac=False):
    pcb = B.load_pcb()
    R = np.load(B.work_path('raster_gerber.npz'))
    cn = json.loads(B.work_path('comp_net_gerber.json').read_text())
    res = json.loads(B.work_path('solve_gerber.json').read_text())
    lab = {'F': R['labF'], 'B': R['labB']}
    pads = B.pad_index(pcb)

    # ground islands as fabricated
    def comp_of(L, name):
        p = pads[name]
        return lab[L][B.ij(p['x'], p['y'])]

    islands = {}
    for k in (k for k, v in cn.items() if v == ['GND']):
        L, c = k[0], int(k[1:])
        m = lab[L] == c
        if L == 'B':
            role = 'bottom'
        elif c == comp_of('F', 'IC1.2'):
            role = 'ic'
        elif c == comp_of('F', 'C3.2'):
            role = 'main'
        else:
            role = 'sw'
        islands[role] = {'layer': L, 'area': r2(m.sum() / S / S), 'd': mask_path(m)}

    # other copper from KiCad
    tracks = [{'l': s['layer'][0], 'n': ROLE.get(s['net'], 'other'), 'w': s['w'],
               'p': [r2(s['x1']), r2(s['y1']), r2(s['x2']), r2(s['y2'])]} for s in pcb['segments']]
    padlist = []
    for fp in pcb['footprints']:
        for p in fp['pads']:
            w, h = (p['w'], p['h']) if round(p['rot']) % 180 == 0 else (p['h'], p['w'])
            layers = 'FB' if '*.Cu' in p['layers'] else ('F' if 'F.Cu' in p['layers'] else 'B')
            padlist.append({'ref': fp['ref'], 'pin': p['num'], 'net': p['net'] or '(no net)', 'n': ROLE.get(p['net'], 'other'),
                            's': p['shape'], 'x': r2(p['x']), 'y': r2(p['y']), 'w': r2(w), 'h': r2(h), 'L': layers})
    vias = [{'x': r2(v['x']), 'y': r2(v['y']), 'd': v['size'], 'n': ROLE.get(v['net'], 'other')} for v in pcb['vias']]
    parts = [{'ref': f['ref'], 'val': f['val'], 'x': r2(f['x']), 'y': r2(f['y'])} for f in pcb['footprints']]

    # current-density heatmaps and the packed flow field for the particles
    heat, fields, field_meta = {}, {}, {}
    GH, GW = H // G2, W // G2
    blob = bytearray()
    gnd = {L: np.zeros((H, W), bool) for L in 'FB'}
    for k in (k for k, v in cn.items() if v == ['GND']):
        gnd[k[0]] |= lab[k[0]] == int(k[1:])
    for s in ('hot', 'input', 'dc'):
        J = load_J(s, gnd, ac)
        mags = np.concatenate([J[L][2][J[L][3]] for L in 'FB'])
        hi = math.log10(np.percentile(mags[mags > 0], 99.7))
        lo = hi - DECADES
        heat[s] = {}
        field_meta[s] = {'lo': lo, 'hi': hi, 'jmax': r2(10 ** hi)}
        for L in 'FB':
            Jx, Jy, mag, copper = J[L]
            t = (np.log10(np.maximum(mag, 1e-12)) - lo) / (hi - lo)
            t[~copper] = 0
            heat[s][L] = gray_uri(t)
            bx = Jx[:GH * G2, :GW * G2].reshape(GH, G2, GW, G2).mean((1, 3))
            by = Jy[:GH * G2, :GW * G2].reshape(GH, G2, GW, G2).mean((1, 3))
            cu = copper[:GH * G2, :GW * G2].reshape(GH, G2, GW, G2).mean((1, 3)) > 0.5
            bm = np.hypot(bx, by)
            tm = np.clip((np.log10(np.maximum(bm, 1e-12)) - lo) / (hi - lo), 0, 1)
            m8 = np.where(cu, 1 + np.round(tm * 254), 0).astype(np.uint8)
            a8 = (np.round((np.arctan2(by, bx) % (2 * math.pi)) / (2 * math.pi) * 256) % 256).astype(np.uint8)
            fields[f'{s}{L}'] = len(blob)
            blob += m8.tobytes() + a8.tobytes()

    # ground potential of the hot-loop return (0 = IC1 pin 2, 1 = C3 pin 2) for the feedback view
    Vh = load_V('hot')
    vc3 = res['hot']['C3.2']
    pot, iso = {}, {}
    for L in 'FB':
        t = Vh[L] / vc3
        pot[L] = gray_uri(np.where(np.isnan(t), 0, np.clip(np.nan_to_num(t), 0, 1) * 0.97 + 0.03))
        iso[L] = iso_paths(t, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    via_cur = {s: [{'x': r2(x), 'y': r2(y), 'i': round(c, 3)} for x, y, c in via_currents(s, res, ac) if abs(c) >= 0.04]
               for s in ('hot', 'input', 'dc')}

    return {
        'board': {'x0': 126.0956, 'y0': 97.1041, 'x1': 170.9066, 'y1': 112.9031},
        'grid': {'x0': X0, 'y0': Y0, 'w': W, 'h': H, 's': S, 'gw': GW, 'gh': GH, 'g2': G2},
        'islands': islands, 'tracks': tracks, 'pads': padlist, 'vias': vias, 'fab': fab_outlines(), 'parts': parts,
        'heat': heat, 'pot': pot, 'iso': iso,
        'field': {'b64': base64.b64encode(gzip.compress(bytes(blob), 9, mtime=0)).decode(), 'offsets': fields, 'meta': field_meta},
        'viaCur': via_cur,
        'probes': {s: {k: round(v * 1e3, 3) for k, v in res[s].items()} for s in ('hot', 'input', 'dc')},
        'sinks': {s: res[s + '_sinks'] for s in ('hot', 'input', 'dc')},
    }


def main():
    """The DC page, and beside it the same page with the switching and input loops from solve_ac.py."""
    template = B.TEMPLATE.read_text(encoding='utf8')
    assert template.count('/*__DATA__*/null') == 1
    for page, ac in ((B.PAGE, False), (B.PAGE_AC, True)):
        data = json.dumps(build_payload(ac), separators=(',', ':'))
        B.work_path(f'viz_data{"_ac" if ac else ""}.json').write_text(data)
        page.write_text(template.replace('/*__DATA__*/null', data), encoding='utf8')
        print(f'wrote {page.name} ({page.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
