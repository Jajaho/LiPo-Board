# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "scipy>=1.13", "pillow>=10", "contourpy>=1.2"]
# ///
"""Split each copper layer into connected pieces and name their nets.

Runs on both copper sources: the 2019 production Gerbers (what was fabricated) and the
KiCad-2019 file's stored zone fill. Writes work/raster_<src>.npz and work/comp_net_<src>.json.
"""
import json

import numpy as np
from scipy import ndimage

import board as B


def tag_components(lab, pcb, use_segments):
    comp_net = {}

    def tag(L, x, y, net):
        c = lab[L][B.ij(x, y)]
        if c:
            comp_net.setdefault(f'{L}{c}', set()).add(net)

    for fp in pcb['footprints']:
        for p in fp['pads']:
            for L in 'FB':
                if f'{L}.Cu' in p['layers'] or '*.Cu' in p['layers']:
                    tag(L, p['x'], p['y'], p['net'])
    for v in pcb['vias']:
        for L in 'FB':
            tag(L, v['x'], v['y'], v['net'])
    if use_segments:
        for s in pcb['segments']:
            tag(s['layer'][0], (s['x1'] + s['x2']) / 2, (s['y1'] + s['y2']) / 2, s['net'])
    return {k: sorted(map(str, v)) for k, v in comp_net.items()}


def report_gnd(src, lab, comp_net, pcb):
    pads = B.pad_index(pcb)
    print(f'[{src}] GND copper pieces:')
    for key, nets in sorted(comp_net.items(), key=lambda kv: (kv[0][0], int(kv[0][1:]))):
        if nets != ['GND']:
            continue
        L, c = key[0], int(key[1:])
        m = lab[L] == c
        members = [n for n, p in pads.items() if p['net'] == 'GND'
                   and (f'{L}.Cu' in p['layers'] or '*.Cu' in p['layers']) and lab[L][B.ij(p['x'], p['y'])] == c]
        vias = sum(1 for v in pcb['vias'] if v['net'] == 'GND' and lab[L][B.ij(v['x'], v['y'])] == c)
        print(f'  {key:4} {m.sum() / B.S ** 2:7.2f} mm2  vias={vias:2}  pads={members}')


def main():
    pcb = B.load_pcb()
    cu = {}
    for src in ('gerber', 'kicad'):
        cu[src] = {L: (B.raster_gerber(L) if src == 'gerber' else B.raster_kicad(pcb, L)) for L in 'FB'}
        lab = {L: ndimage.label(cu[src][L])[0] for L in 'FB'}
        comp_net = tag_components(lab, pcb, use_segments=(src == 'gerber'))
        np.savez_compressed(B.work_path(f'raster_{src}.npz'), F=cu[src]['F'], B=cu[src]['B'],
                            labF=lab['F'], labB=lab['B'])
        B.work_path(f'comp_net_{src}.json').write_text(json.dumps(comp_net))
        report_gnd(src, lab, comp_net, pcb)
    for L in 'FB':
        g, k = cu['gerber'][L], cu['kicad'][L]
        print(f'[diff] {L}.Cu  Gerber-only {(g & ~k).sum() / B.S ** 2:.1f} mm2,  KiCad-only {(k & ~g).sum() / B.S ** 2:.1f} mm2')


if __name__ == '__main__':
    main()
