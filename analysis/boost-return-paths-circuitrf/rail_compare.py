"""Re-run the boost return-path scenarios with circuitRF's railRF (`circuitrf rail`) and compare
against the Python sheet-resistor solve.

GND is declared as the 'rail' and its return is an ideal dummy plane (Ideal Return layer, net REF),
so the reported drops are the potential differences inside the GND copper alone.
Dirichlet pads (held at 0 V in the Python model) become a grid of ideal 1 V sources across the pad;
injection pads become a grid of loads sharing the current; probes are a grid of observation ports
averaged per pad, mimicking the Python model's pad-mean potential.
"""
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / 'boost-return-paths-python'      # the Python solve: pad table and its results in work/
WS = HERE / 'gnd-solve'
EXE = os.environ.get('CIRCUITRF', str(Path(os.environ['LOCALAPPDATA']) / 'Programs' / 'circuitRF' / 'circuitRF.com'))
OUT = WS / 'out'

SCENARIOS = {
    'hot': ({'C3.2': 1.0}, ['IC1.2']),
    'input': ({'IC1.2': 1.0}, ['C1.2', 'C4.2']),
    'dc': ({'J3.1': 1.0, 'IC1.2': 0.5}, ['Q1.3']),
}
PROBES = ('IC1.2', 'R3.1', 'C3.2', 'C1.2', 'C4.2', 'J3.1', 'Q1.3', 'R7.1', 'IC2.9')

pcb = json.loads((REPO / 'work' / 'pcb.json').read_text())
PADS = {f"{fp['ref']}.{p['num']}": p for fp in pcb['footprints'] for p in fp['pads']}


def grid(name):
    """Anchor points (DBU, clay frame: x, -y) spread over a pad; skips the drill of a THT pad."""
    p = PADS[name]
    w, h = (p['w'], p['h']) if round(p['rot']) % 180 == 0 else (p['h'], p['w'])
    pts = []
    for fx in (-0.35, 0, 0.35):
        for fy in (-0.35, 0, 0.35):
            if p['type'] == 'thru_hole' and fx == 0 and fy == 0:
                continue
            x, y = p['x'] + fx * w, p['y'] + fy * h
            pts.append((round(x * 1e6), round(-y * 1e6)))
    return pts


def key(x, y):
    return f'{x / 1000:.1f},{y / 1000:.1f}'


def run(name, crail, accurate=True):
    inj, dirichlet = SCENARIOS[name]
    args, owner = [], {}
    for pad in dirichlet:
        for x, y in grid(pad):
            args += ['--source', f'@{x},{y}=1V']
            owner[key(x, y)] = pad
    for pad, amps in inj.items():
        g = grid(pad)
        for x, y in g:
            args += ['--load', f'@{x},{y}={amps / len(g) * 1e3:.9f}mA']
            owner[key(x, y)] = pad
    for pad in PROBES:
        if pad in inj or pad in dirichlet:
            continue
        for x, y in grid(pad):
            args += ['--load', f'@{x},{y}']
            owner[key(x, y)] = pad
    out = OUT / f'{Path(crail).stem}_{name}.csv'
    cmd = [EXE, 'rail', str(WS / crail), *(['--accurate'] if accurate else []), *args, '-o', str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode:
        sys.exit(r.stdout[-3000:] + r.stderr[-3000:])
    notes = [l for l in (r.stdout + r.stderr).splitlines() if 'moved' in l or 'coarsened' in l]
    v, src = {}, {}
    for row in csv.reader(l for l in out.read_text(encoding='utf-8').splitlines() if not l.startswith('#')):
        if len(row) < 5:
            continue
        m = re.match(r'\(([-\d.]+), ([-\d.]+)\)', row[2])
        if not m:
            continue
        k = f'{float(m.group(1)):.1f},{float(m.group(2)):.1f}'
        if row[0] == 'port':
            v.setdefault(owner[k], []).append(1 - float(row[3]))
        elif row[0] == 'source':
            src[owner[k]] = src.get(owner[k], 0) + float(row[3])
    res = {p: sum(v[p]) / len(v[p]) for p in v}
    for p in dirichlet:
        res[p] = 0.0
    return res, src, notes


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    crail = sys.argv[1] if len(sys.argv) > 1 else 'gnd.crail'
    py = json.loads((REPO / 'work' / 'solve_kicad.json').read_text())
    pg = json.loads((REPO / 'work' / 'solve_gerber.json').read_text())
    summary = {}
    for name in SCENARIOS:
        res, src, notes = run(name, crail)
        summary[name] = {'mV': res, 'sinks_A': src}
        print(f'\n== {name}   (mV; railRF accurate | Python KiCad fill | Python Gerbers)')
        for p in PROBES:
            print(f'  {p:6} {res[p] * 1e3:7.3f} | {py[name][p] * 1e3:7.3f} | {pg[name][p] * 1e3:7.3f}')
        print('  sinks (A):', {k: round(a, 4) for k, a in src.items()},
              '| python kicad', {k: round(a, 4) for k, a in py[name + '_sinks'].items()})
        for n in notes:
            print('  ', n.strip())
    (OUT / f'{Path(crail).stem}_compare.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
