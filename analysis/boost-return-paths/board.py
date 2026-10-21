"""Shared inputs for the boost return-path analysis: paths, constants, board readers, rasterisers.

Copper is rasterised onto one grid in the KiCad board frame (mm, y down):
x 126..171 mm, y 97..113 mm at 0.05 mm per cell.
"""
import json
import math
import re
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PCB_FILE = REPO / 'src' / 'kicad-2019' / 'LiPo-BoardV1-2019.kicad_pcb'
GERBER_ZIP = REPO / 'outputs' / 'LiPo-BoardV1_2019-01-18.zip'
GERBER_MEMBERS = {'F': 'CAMOutputs/GerberFiles/copper_top.gbr', 'B': 'CAMOutputs/GerberFiles/copper_bottom.gbr'}
WORK = HERE / 'work'
TEMPLATE = HERE / 'template.html'
PAGE = HERE / 'boost-return-paths.html'

# raster grid
S = 20                              # cells per mm
X0, Y0, X1, Y1 = 126.0, 97.0, 171.0, 113.0
W, H = int((X1 - X0) * S), int((Y1 - Y0) * S)

# EAGLE (0,0) sits at the KiCad aux origin
GERBER_OX, GERBER_OY = 126.0956, 113.9191

# copper and plating
RS = 1.72e-8 / 35e-6                                            # 1 oz sheet resistance [ohm/sq]
R_VIA = 1.72e-8 * 1.6e-3 / (math.pi * 0.35e-3 * 20e-6)          # 0.35 mm drill, 20 um plating
R_THT = 1.72e-8 * 1.6e-3 / (math.pi * 1.0e-3 * 20e-6)           # plated pins
R_J3 = 0.2e-3                                                    # J3 carries a soldered wire


def work_path(name):
    WORK.mkdir(exist_ok=True)
    return WORK / name


def ij(x, y):
    return int(round((y - Y0) * S)), int(round((x - X0) * S))


def px(p):
    return ((p[0] - X0) * S, (p[1] - Y0) * S)


# ---------------------------------------------------------------- KiCad s-expressions
def parse_sexp(text):
    tok = re.compile(r'\(|\)|"(?:[^"\\]|\\.)*"|[^\s()"]+')
    stack = [[]]
    for m in tok.finditer(text):
        t = m.group(0)
        if t == '(':
            stack.append([])
        elif t == ')':
            x = stack.pop()
            stack[-1].append(x)
        elif t.startswith('"'):
            stack[-1].append(t[1:-1])
        else:
            stack[-1].append(t)
    return stack[0][0]


def find(node, name):
    return [c for c in node if isinstance(c, list) and c and c[0] == name]


def find1(node, name):
    r = find(node, name)
    return r[0] if r else None


def read_pcb_tree():
    return parse_sexp(PCB_FILE.read_text(encoding='utf8'))


def extract_pcb():
    """Footprints with absolute pad positions, tracks, vias, zones and board edges."""
    pcb = read_pcb_tree()
    out = {'footprints': [], 'segments': [], 'vias': [], 'zones': [], 'edges': []}
    for fp in find(pcb, 'footprint'):
        at = find1(fp, 'at')
        x, y = float(at[1]), float(at[2])
        rot = float(at[3]) if len(at) > 3 else 0
        props = {p[1]: p[2] for p in find(fp, 'property')}
        pads = []
        for p in find(fp, 'pad'):
            pat = find1(p, 'at')
            lx, ly = float(pat[1]), float(pat[2])
            prot = float(pat[3]) if len(pat) > 3 else 0
            r = math.radians(-rot)
            gx = x + lx * math.cos(r) - ly * math.sin(r)
            gy = y + lx * math.sin(r) + ly * math.cos(r)
            sz = find1(p, 'size')
            net = find1(p, 'net')
            pads.append(dict(num=p[1], type=p[2], shape=p[3], x=round(gx, 4), y=round(gy, 4),
                             w=float(sz[1]), h=float(sz[2]), rot=prot,
                             net=(net[1] if net else None), layers=find1(p, 'layers')[1:]))
        out['footprints'].append(dict(ref=props.get('Reference'), val=props.get('Value'), fp=fp[1],
                                      x=x, y=y, rot=rot, layer=find1(fp, 'layer')[1], pads=pads))
    for s in find(pcb, 'segment'):
        st, en, net = find1(s, 'start'), find1(s, 'end'), find1(s, 'net')
        out['segments'].append(dict(x1=float(st[1]), y1=float(st[2]), x2=float(en[1]), y2=float(en[2]),
                                    w=float(find1(s, 'width')[1]), layer=find1(s, 'layer')[1],
                                    net=net[1] if net else None))
    for v in find(pcb, 'via'):
        at, net = find1(v, 'at'), find1(v, 'net')
        out['vias'].append(dict(x=float(at[1]), y=float(at[2]), size=float(find1(v, 'size')[1]),
                                drill=float(find1(v, 'drill')[1]), net=net[1] if net else None))
    for z in find(pcb, 'zone'):
        net = find1(z, 'net')
        poly = [[float(p[1]), float(p[2])] for p in find1(find1(z, 'polygon'), 'pts')[1:]]
        fills = [dict(layer=find1(f, 'layer')[1],
                      pts=[[float(p[1]), float(p[2])] for p in find1(f, 'pts')[1:] if p[0] == 'xy'])
                 for f in find(z, 'filled_polygon')]
        layers = find1(z, 'layers') or find1(z, 'layer')
        out['zones'].append(dict(net=net[1] if net else None, layers=layers[1:], poly=poly, fills=fills,
                                 keepout=bool(find1(z, 'keepout'))))
    for g in find(pcb, 'gr_line') + find(pcb, 'gr_rect') + find(pcb, 'gr_arc'):
        if find1(g, 'layer')[1] == 'Edge.Cuts':
            out['edges'].append([g[0]] + [[float(c[1]), float(c[2])] for c in g[1:]
                                          if isinstance(c, list) and c[0] in ('start', 'end', 'mid')])
    return out


def load_pcb():
    path = work_path('pcb.json')
    if not path.exists():
        path.write_text(json.dumps(extract_pcb(), indent=0))
    return json.loads(path.read_text())


def pad_index(pcb):
    return {f"{fp['ref']}.{p['num']}": p for fp in pcb['footprints'] for p in fp['pads']}


# ---------------------------------------------------------------- EAGLE Gerbers (linear only, no arcs)
def _gerber_xy(x, y):
    return (round(GERBER_OX + x / 1e4, 4), round(GERBER_OY - y / 1e4, 4))


def parse_gerber(text):
    apertures = {}
    for m in re.finditer(r'%ADD(\d+)([A-Z0-9]+),([0-9.X]+)\*%', text):
        apertures[int(m.group(1))] = (m.group(2), [float(v) for v in m.group(3).split('X')])
    body = re.sub(r'%[^%]*%', lambda m: m.group(0).replace('\n', ''), text)
    ops, pol, cur_ap, x, y, region = [], 'D', None, 0, 0, None
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('%LPD'):
            pol = 'D'; continue
        if line.startswith('%LPC'):
            pol = 'C'; continue
        if line.startswith('%') or line.startswith('G04') or line in ('G75*', 'G01*', 'M02*'):
            continue
        if line == 'G36*':
            region = []; continue
        if line == 'G37*':
            if region:
                ops.append(dict(kind='region', pol=pol, contours=region))
            region = None; continue
        m = re.match(r'^D(\d+)\*$', line)
        if m and int(m.group(1)) >= 10:
            cur_ap = int(m.group(1)); continue
        m = re.match(r'^(?:G01)?(?:X(-?\d+))?(?:Y(-?\d+))?D0([123])\*$', line)
        if not m:
            raise ValueError('unhandled Gerber line: ' + line)
        nx = int(m.group(1)) if m.group(1) is not None else x
        ny = int(m.group(2)) if m.group(2) is not None else y
        d = m.group(3)
        if region is not None:
            if d == '2':
                region.append([_gerber_xy(nx, ny)])
            elif d == '1':
                region[-1].append(_gerber_xy(nx, ny))
        elif d == '1':
            ops.append(dict(kind='line', pol=pol, ap=apertures[cur_ap], p1=_gerber_xy(x, y), p2=_gerber_xy(nx, ny)))
        elif d == '3':
            ops.append(dict(kind='flash', pol=pol, ap=apertures[cur_ap], p=_gerber_xy(nx, ny)))
        x, y = nx, ny
    return ops


def raster_gerber(layer):
    """Copper of one layer ('F' or 'B') as fabricated, straight from the production zip."""
    with zipfile.ZipFile(GERBER_ZIP) as z:
        ops = parse_gerber(z.read(GERBER_MEMBERS[layer]).decode('ascii'))
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    for op in ops:
        c = 255 if op['pol'] == 'D' else 0
        if op['kind'] == 'region':
            for cont in op['contours']:
                if len(cont) > 2:
                    d.polygon([px(p) for p in cont], fill=c)
        elif op['kind'] == 'line':
            a = op['ap'][1]
            r = a[0] / 2 * S
            d.line([px(op['p1']), px(op['p2'])], fill=c, width=max(1, int(round(a[0] * S))))
            for p in (op['p1'], op['p2']):
                q = px(p)
                d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=c)
        elif op['kind'] == 'flash':
            t, a = op['ap']
            q = px(op['p'])
            if t == 'R':
                w, h = a[0] / 2 * S, a[1] / 2 * S
                d.rectangle([q[0] - w, q[1] - h, q[0] + w, q[1] + h], fill=c)
            else:
                r = a[0] / 2 * S
                d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=c)
    return np.array(im) > 127


def raster_kicad(pcb, layer):
    """Copper of one layer ('F' or 'B') as stored in the KiCad-2019 file: zone fills, tracks, pads, vias."""
    L = f'{layer}.Cu'
    im = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(im)
    for z in pcb['zones']:
        for f in z['fills']:
            if f['layer'] == L:
                d.polygon([px(p) for p in f['pts']], fill=255)
    for s in pcb['segments']:
        if s['layer'] == L:
            r = s['w'] / 2 * S
            d.line([px((s['x1'], s['y1'])), px((s['x2'], s['y2']))], fill=255, width=int(round(s['w'] * S)))
            for p in ((s['x1'], s['y1']), (s['x2'], s['y2'])):
                q = px(p)
                d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=255)
    for v in pcb['vias']:
        q = px((v['x'], v['y']))
        r = v['size'] / 2 * S
        d.ellipse([q[0] - r, q[1] - r, q[0] + r, q[1] + r], fill=255)
    for fp in pcb['footprints']:
        for p in fp['pads']:
            if L not in p['layers'] and '*.Cu' not in p['layers']:
                continue
            w, h = (p['w'], p['h']) if round(p['rot']) % 180 == 0 else (p['h'], p['w'])
            q = px((p['x'], p['y']))
            box = [q[0] - w / 2 * S, q[1] - h / 2 * S, q[0] + w / 2 * S, q[1] + h / 2 * S]
            if p['shape'] == 'rect':
                d.rectangle(box, fill=255)
            elif p['shape'] == 'circle':
                d.ellipse(box, fill=255)
            else:
                d.rounded_rectangle(box, radius=min(w, h) / 2 * S, fill=255)
    return np.array(im) > 127
