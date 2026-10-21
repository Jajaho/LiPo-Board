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

RAMP = {  # dark-panel sequential ramps: near-zero recedes into the board, high current is bright
    'F': [(0.00, (120, 60, 20)), (0.45, (214, 110, 40)), (0.80, (250, 170, 70)), (1.00, (255, 236, 190))],
    'B': [(0.00, (20, 60, 120)), (0.45, (40, 120, 214)), (0.80, (90, 190, 250)), (1.00, (215, 245, 255))],
    'P': [(0.00, (40, 30, 90)), (0.45, (110, 85, 210)), (0.80, (170, 150, 245)), (1.00, (235, 228, 255))],
}


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


def colorize(t, ramp, alpha_max=0.92, gamma=0.9):
    t = np.clip(t, 0, 1)
    stops = RAMP[ramp]
    rgb = np.zeros(t.shape + (3,))
    for (a, ca), (b, cb) in zip(stops[:-1], stops[1:]):
        m = (t >= a) & (t <= b)
        u = ((t - a) / (b - a))[m][:, None]
        rgb[m] = np.array(ca) * (1 - u) + np.array(cb) * u
    alpha = (t ** gamma) * alpha_max
    return np.dstack([rgb, alpha * 255]).astype(np.uint8)


def png_uri(img):
    buf = io.BytesIO()
    Image.fromarray(img, 'RGBA').save(buf, 'PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def load_V(scenario):
    d = np.load(B.work_path(f'V_gerber_{scenario}.npz'))
    return {'F': d['F'], 'B': d['B']}


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


def build_payload():
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
    for s in ('hot', 'input', 'dc'):
        V = load_V(s)
        J = {L: jfield(V[L]) for L in 'FB'}
        mags = np.concatenate([np.hypot(*J[L])[~np.isnan(V[L])] for L in 'FB'])
        hi = math.log10(np.percentile(mags[mags > 0], 99.7))
        lo = hi - DECADES
        heat[s] = {}
        field_meta[s] = {'lo': lo, 'hi': hi, 'jmax': r2(10 ** hi)}
        for L in 'FB':
            Jx, Jy = J[L]
            mag = np.hypot(Jx, Jy)
            t = (np.log10(np.maximum(mag, 1e-12)) - lo) / (hi - lo)
            t[np.isnan(V[L])] = 0
            heat[s][L] = png_uri(colorize(t, L, gamma=1.6))
            bx = Jx[:GH * G2, :GW * G2].reshape(GH, G2, GW, G2).mean((1, 3))
            by = Jy[:GH * G2, :GW * G2].reshape(GH, G2, GW, G2).mean((1, 3))
            cu = (~np.isnan(V[L]))[:GH * G2, :GW * G2].reshape(GH, G2, GW, G2).mean((1, 3)) > 0.5
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
        img = colorize(np.nan_to_num(np.clip(t, 0, 1.0)) * 0.97 + 0.03 * (~np.isnan(t)), 'P', alpha_max=0.85, gamma=1.0)
        img[np.isnan(t)] = 0
        pot[L] = png_uri(img)
        iso[L] = iso_paths(t, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

    via_cur = {s: [{'x': r2(x), 'y': r2(y), 'i': round(c, 3)} for x, y, c in res[s + '_vias'] if abs(c) >= 0.04]
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
    data = json.dumps(build_payload(), separators=(',', ':'))
    B.work_path('viz_data.json').write_text(data)
    template = B.TEMPLATE.read_text(encoding='utf8')
    assert template.count('/*__DATA__*/null') == 1
    B.PAGE.write_text(template.replace('/*__DATA__*/null', data), encoding='utf8')
    print(f'wrote {B.PAGE.name} ({B.PAGE.stat().st_size / 1e6:.2f} MB)')


if __name__ == '__main__':
    main()
