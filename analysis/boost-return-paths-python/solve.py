# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "scipy>=1.13", "pillow>=10", "contourpy>=1.2"]
# ///
"""Resistive current-spreading solve in the GND copper (both layers + vias).

Scenarios (all currents in A, + = injected into the GND copper):
  dc    : 1 A load. Load return enters at J3, the MT3608 switch's average
          current leaves the IC at pin 2, everything goes back to the cell
          through Q1 pin 3 (protection FET, S2 = P-).
  hot   : 1 A unit of the switching "hot loop" return: C3- -> IC1 pin 2.
  input : 1 A unit of inductor ripple return: IC1 pin 2 -> C1- / C4-.

Writes work/V_<src>_<scenario>.npz (potential per cell, NaN off-copper) and work/solve_<src>.json.
"""
import json
import math

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

import board as B

SCENARIOS = {
    'dc': ({'J3.1': 1.0, 'IC1.2': 0.5}, ['Q1.3']),
    'hot': ({'C3.2': 1.0}, ['IC1.2']),
    'input': ({'IC1.2': 1.0}, ['C1.2', 'C4.2']),
}
PROBES = ('IC1.2', 'R3.1', 'C3.2', 'C1.2', 'C4.2', 'J3.1', 'Q1.3', 'R7.1', 'IC2.9')
PLATED_PINS = ('J3.1', 'S1.3', 'J1.MP1', 'J1.MP2', 'J1.MP3', 'J1.MP4')


def gnd_mask(src):
    d = np.load(B.work_path(f'raster_{src}.npz'))
    lab = {'F': d['labF'], 'B': d['labB']}
    comp_net = json.loads(B.work_path(f'comp_net_{src}.json').read_text())
    gnd = {L: np.zeros((B.H, B.W), bool) for L in 'FB'}
    for k, v in comp_net.items():
        if v == ['GND']:                         # floating copper text is left out
            gnd[k[0]] |= lab[k[0]] == int(k[1:])
    return gnd


class Network:
    def __init__(self, src, pcb):
        gnd = gnd_mask(src)
        self.pads = B.pad_index(pcb)
        self.idx, n = {}, 0
        for L in 'FB':
            m = np.full((B.H, B.W), -1, int)
            cnt = gnd[L].sum()
            m[gnd[L]] = np.arange(n, n + cnt)
            self.idx[L] = m
            n += cnt
        self.n = n
        rows, cols, vals = [], [], []
        g_sq = 1 / B.RS
        for L in 'FB':
            m = self.idx[L]
            for a, b in ((m[:, :-1], m[:, 1:]), (m[:-1, :], m[1:, :])):
                ok = (a >= 0) & (b >= 0)
                aa, bb = a[ok], b[ok]
                rows.extend(np.concatenate([aa, bb, aa, bb]))
                cols.extend(np.concatenate([aa, bb, bb, aa]))
                vals.extend(np.concatenate([np.full(len(aa), g_sq)] * 2 + [np.full(len(aa), -g_sq)] * 2))

        def edge(a, b, g):
            rows.extend([a, b, a, b]); cols.extend([a, b, b, a]); vals.extend([g, g, -g, -g])

        self.vias = []
        for v in pcb['vias']:
            if v['net'] != 'GND':
                continue
            pairs = self.ring(v['x'], v['y'], v['size'] / 2)
            assert pairs, v
            for a, b in pairs:
                edge(a, b, 1 / B.R_VIA / len(pairs))
            self.vias.append((v['x'], v['y'], pairs, 1 / B.R_VIA))
        for name in PLATED_PINS:
            p = self.pads[name]
            pairs = self.ring(p['x'], p['y'], min(p['w'], p['h']) / 2 * 0.8)
            g = 1 / B.R_J3 if name == 'J3.1' else 1 / B.R_THT
            for a, b in pairs:
                edge(a, b, g / len(pairs))
            self.vias.append((p['x'], p['y'], pairs, g))
        self.G = sparse.csr_matrix((np.array(vals, float), (np.array(rows), np.array(cols))), shape=(n, n))

    def ring(self, x, y, r):
        """Cell pairs (top, bottom) within radius r of (x, y) that are GND copper on both layers."""
        i0, j0 = B.ij(x, y)
        rr = int(math.ceil(r * B.S))
        out = []
        for di in range(-rr, rr + 1):
            for dj in range(-rr, rr + 1):
                if di * di + dj * dj <= (r * B.S) ** 2:
                    i, j = i0 + di, j0 + dj
                    if self.idx['F'][i, j] >= 0 and self.idx['B'][i, j] >= 0:
                        out.append((self.idx['F'][i, j], self.idx['B'][i, j]))
        return out

    def pad_nodes(self, name, layer='F'):
        p = self.pads[name]
        w, h = (p['w'], p['h']) if round(p['rot']) % 180 == 0 else (p['h'], p['w'])
        i0, j0 = B.ij(p['x'] - w / 2, p['y'] - h / 2)
        i1, j1 = B.ij(p['x'] + w / 2, p['y'] + h / 2)
        m = self.idx[layer][i0:i1 + 1, j0:j1 + 1]
        return m[m >= 0]

    def solve(self, inj, dirichlet):
        """inj: {pad: amps}; dirichlet: pads held at 0 V."""
        I = np.zeros(self.n)
        for name, amps in inj.items():
            nodes = self.pad_nodes(name)
            I[nodes] += amps / len(nodes)
        fixed = np.zeros(self.n, bool)
        for name in dirichlet:
            fixed[self.pad_nodes(name)] = True
        free = ~fixed
        V = np.zeros(self.n)
        V[free] = spsolve(self.G[free][:, free].tocsc(), I[free])
        return V

    def unpack(self, V):
        out = {}
        for L in 'FB':
            m = self.idx[L]
            a = np.full(m.shape, np.nan)
            a[m >= 0] = V[m[m >= 0]]
            out[L] = a
        return out

    def pad_v(self, Vl, name, layer='F'):
        return float(np.nanmean(Vl[layer][np.isin(self.idx[layer], self.pad_nodes(name, layer))]))


def solve_source(src, pcb):
    net = Network(src, pcb)
    print(f'[{src}] {net.n} nodes')
    res = {}
    for k, (inj, dirichlet) in SCENARIOS.items():
        V = net.solve(inj, dirichlet)
        Vl = net.unpack(V)
        np.savez_compressed(B.work_path(f'V_{src}_{k}.npz'), F=Vl['F'], B=Vl['B'])
        res[k] = {p: net.pad_v(Vl, p) for p in PROBES}
        # current changing layer at each via / plated pin (+ = top -> bottom)
        res[k + '_vias'] = [(x, y, sum((V[a] - V[b]) * g / len(pairs) for a, b in pairs))
                            for x, y, pairs, g in net.vias]
        r = net.G @ V
        res[k + '_sinks'] = {name: -float(r[net.pad_nodes(name)].sum()) for name in dirichlet}
        print(f'  {k:5}', ' '.join(f'{p}={v * 1e3:.3f}mV' for p, v in res[k].items()))
        print(f'        sinks (A): {res[k + "_sinks"]}')
    B.work_path(f'solve_{src}.json').write_text(json.dumps(res, indent=1))
    return res


def main():
    pcb = B.load_pcb()
    for src in ('gerber', 'kicad'):
        solve_source(src, pcb)


if __name__ == '__main__':
    main()
