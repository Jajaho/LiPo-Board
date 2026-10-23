# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "scipy>=1.13", "pillow>=10", "contourpy>=1.2"]
# ///
"""AC (R + jwL) current-spreading solve: the resistive sheet network of solve.py with inductance.

At DC the current spreads by resistance alone. Above R/(2 pi L) -- about 0.6 MHz for the switching
loop -- inductance takes over and the return current crowds under its forward path. This solve is a
quasi-static PEEC model on the same 0.05 mm raster:

  - every cell edge of the solved nets' copper (both layers) is a current element: R = one square of
    1 oz copper, partial inductance to every other parallel element (Neumann formula for parallel
    filaments, self term regularised by the cross-section's geometric mean distance), applied by FFT;
  - vias and plated pins are vertical elements: R as in solve.py, partial self inductance of a tube;
    perpendicular to the sheet currents, they couple to none of them (mutual between vias neglected);
  - the parts are straight line currents between their pad centres, 0.5 mm above the top copper:
    the IC switch and L1 carry a prescribed 1 A, capacitors and D1 are solved branches with their
    own impedance. Part ESL beyond that straight path is not included.

Scenarios:
  input : inductor-ripple loop at the switching frequency (1.2 MHz): C1/C4 -> BAT+ -> L1 -> SW ->
          switch -> IC1 pin 2 -> GND -> C1-/C4-. The C1/C4 split is solved, not imposed.
  hot   : switching loop at 100 MHz, the edges: IC1 pin 2 -> switch -> SW -> D1 -> 5V -> C3 -> GND.
          Its loop inductance is the cross-check against the openEMS full-wave solve (1.35 nH).

Writes work/Jac_<src>_<scenario>.npz (complex sheet current density per cell, A/mm, NaN off-copper)
and work/solve_ac_<src>.json. Usage: solve_ac.py [scenario ...] [--f scenario=Hz ...] [--src gerber|kicad]
"""
import json
import math
import sys
import time

import numpy as np
from scipy import fft, sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, gmres, splu

import board as B
from board import H, W, S

MU0 = 4e-7 * math.pi
T_CU = 35e-6
DZ = 1.535e-3                       # top to bottom copper, centre to centre
Z_PART = 0.5e-3                     # parts' line currents above the top copper
L_EL = 1e-3 / S                     # element length (one cell)
GMD = 0.2235 * (L_EL + T_CU)        # geometric mean distance of a cell's cross-section to itself

# (C, ESR) of the capacitors; D1 conducting is a short
CAPS = {'C1': (22e-6, 5e-3), 'C4': (10e-6, 10e-3), 'C3': (22e-6, 5e-3)}

# name: (frequency, nets solved, prescribed 1 A branches, solved branches); a branch is (from pad, to pad)
SCENARIOS = {
    'input': (1.2e6, ('GND', 'BAT+', 'N$1'),
              [('L1.1', 'L1.2'), ('IC1.1', 'IC1.2')],
              [('C1.2', 'C1.1'), ('C4.2', 'C4.1')]),
    'hot': (100e6, ('GND', '5V', 'N$1'),
            [('IC1.2', 'IC1.1')],
            [('D1.A', 'D1.C'), ('C3.1', 'C3.2')]),
}


def kernel(dz, along):
    """Mutual partial inductance between parallel cell elements, as an FFT-ready (2H, 2W) array.

    along = 'x' or 'y' is the elements' direction; offsets (di, dj) are in cells."""
    di = np.fft.fftfreq(2 * H, 1 / (2 * H)).astype(int)[:, None]
    dj = np.fft.fftfreq(2 * W, 1 / (2 * W)).astype(int)[None, :]
    s, p = (dj, di) if along == 'x' else (di, dj)
    s = s * L_EL
    d = np.sqrt((p * L_EL) ** 2 + dz ** 2 + GMD ** 2)

    def g(x):
        return x * np.arcsinh(x / d) - np.sqrt(x * x + d * d)

    m = MU0 / (4 * math.pi) * (g(s + L_EL) - 2 * g(s) + g(s - L_EL))
    r = np.sqrt(s * s + d * d)
    far = r > 20 * L_EL
    m[far] = (MU0 / (4 * math.pi) * L_EL ** 2 / r)[far]
    return fft.fft2(m, workers=-1)


class Kernels:
    def __init__(self):
        self.k = {(a, dz): kernel(dz, a) for a in 'xy' for dz in (0.0, DZ, Z_PART, DZ + Z_PART)}
        self.self_l = float(fft.ifft2(self.k['x', 0.0], workers=-1)[0, 0].real)

    @staticmethod
    def fwd(src):
        return fft.fft2(src, s=(2 * H, 2 * W), workers=-1)

    def conv(self, f, a, dz):
        """f: FFT of (H, W) element currents (fwd); returns the flux they link per element."""
        return fft.ifft2(f * self.k[a, dz], workers=-1)[:H, :W]


class AcNetwork:
    def __init__(self, src, pcb, nets, kern):
        self.kern = kern
        d = np.load(B.work_path(f'raster_{src}.npz'))
        lab = {'F': d['labF'], 'B': d['labB']}
        comp_net = json.loads(B.work_path(f'comp_net_{src}.json').read_text())
        self.cu = {L: np.zeros((H, W), bool) for L in 'FB'}
        for k, v in comp_net.items():
            if len(v) == 1 and v[0] in nets:
                self.cu[k[0]] |= lab[k[0]] == int(k[1:])
        self.pads = B.pad_index(pcb)

        self.idx, n = {}, 0
        for L in 'FB':
            m = np.full((H, W), -1, int)
            m[self.cu[L]] = np.arange(n, n + self.cu[L].sum())
            self.idx[L] = m
            n += self.cu[L].sum()
        self.n = n

        # sheet elements: x then y on each layer; current + along +x (+j) or +y (+i)
        rows, cols, vals = [], [], []
        self.emask, ne = {}, 0
        for L in 'FB':
            m = self.idx[L]
            for a, (aa, bb) in (('x', (m[:, :-1], m[:, 1:])), ('y', (m[:-1, :], m[1:, :]))):
                ok = (aa >= 0) & (bb >= 0)
                mask = np.zeros((H, W), bool)
                mask[:ok.shape[0], :ok.shape[1]] = ok
                k = ok.sum()
                e = np.arange(ne, ne + k)
                rows += [aa[ok], bb[ok]]; cols += [e, e]; vals += [np.ones(k), -np.ones(k)]
                self.emask[L, a] = (mask, slice(ne, ne + k))
                ne += k
        self.n_sheet = ne
        z = [np.full(ne, B.RS + 0j)]
        lself = [np.full(ne, kern.self_l)]

        # vias and plated pins: (top, bottom) cell pairs sharing the barrel
        self.vias = []
        barrels = [(v['x'], v['y'], v['size'] / 2, v['drill'] / 2, B.R_VIA, v['net']) for v in pcb['vias'] if v['net'] in nets]
        for fp in pcb['footprints']:
            for p in fp['pads']:
                if p['type'] == 'thru_hole' and p['net'] in nets:
                    r = min(p['w'], p['h']) / 2
                    barrels.append((p['x'], p['y'], r * 0.8, r * 0.8,
                                    B.R_J3 if f"{fp['ref']}.{p['num']}" == 'J3.1' else B.R_THT, p['net']))
        for x, y, r_ring, r_tube, rv, vnet in barrels:
            pairs = self.ring(x, y, r_ring)
            if not pairs:
                continue
            l_via = MU0 / (2 * math.pi) * DZ * (math.log(2 * DZ / (r_tube * 1e-3)) - 1)
            e = np.arange(ne, ne + len(pairs))
            rows += [np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])]
            cols += [e, e]; vals += [np.ones(len(pairs)), -np.ones(len(pairs))]
            z.append(np.full(len(pairs), rv * len(pairs) + 0j))
            lself.append(np.full(len(pairs), l_via * len(pairs)))
            self.vias.append((x, y, vnet, e))
            ne += len(pairs)
        self.ne = ne
        self.D = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, ne))
        self.z_el = np.concatenate(z)
        self.l_el = np.concatenate(lself)

    def ring(self, x, y, r):
        i0, j0 = B.ij(x, y)
        rr = int(math.ceil(r * S))
        out = []
        for di in range(-rr, rr + 1):
            for dj in range(-rr, rr + 1):
                if di * di + dj * dj <= (r * S) ** 2:
                    i, j = i0 + di, j0 + dj
                    if self.idx['F'][i, j] >= 0 and self.idx['B'][i, j] >= 0:
                        out.append((self.idx['F'][i, j], self.idx['B'][i, j]))
        return out

    def pad_nodes(self, name):
        p = self.pads[name]
        w, h = (p['w'], p['h']) if round(p['rot']) % 180 == 0 else (p['h'], p['w'])
        i0, j0 = B.ij(p['x'] - w / 2, p['y'] - h / 2)
        i1, j1 = B.ij(p['x'] + w / 2, p['y'] + h / 2)
        m = self.idx['F'][i0:i1 + 1, j0:j1 + 1]
        nodes = m[m >= 0]
        assert len(nodes), name
        return nodes

    def branch_pattern(self, a, b):
        """A part as a line current pad a -> pad b, spread over the narrower pad's width:
        (Px, Py) element-length fractions on the x and y lattices, per amp."""
        pa, pb = self.pads[a], self.pads[b]
        x0, y0, x1, y1 = pa['x'], pa['y'], pb['x'], pb['y']
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy)
        ux, uy = dx / ln, dy / ln
        wid = min(min(pa['w'], pa['h']), min(pb['w'], pb['h']))
        nw, nl = max(1, round(wid * S)), max(1, math.ceil(ln * S * 2))
        px, py = np.zeros((H, W)), np.zeros((H, W))
        for k in range(nw):
            off = (k + 0.5) / nw * wid - wid / 2
            for t in (np.arange(nl) + 0.5) / nl:
                x, y = x0 + dx * t - uy * off, y0 + dy * t + ux * off
                px[round((y - B.Y0) * S), round((x - B.X0) * S - 0.5)] += dx * S / nl / nw
                py[round((y - B.Y0) * S - 0.5), round((x - B.X0) * S)] += dy * S / nl / nw
        return px, py

    def gather(self, grids):
        """{(L, a): (H, W) array} -> element vector (sheet elements only, vias zero)."""
        v = np.zeros(self.ne, complex)
        for (L, a), (mask, sl) in self.emask.items():
            v[sl] = grids[L, a][mask]
        return v

    def scatter(self, vec):
        out = {}
        for (L, a), (mask, sl) in self.emask.items():
            g = np.zeros((H, W), complex)
            g[mask] = vec[sl]
            out[L, a] = g
        return out

    def flux_sheet(self, vec):
        """Flux linked per sheet element by the sheet currents vec (complex)."""
        g = self.scatter(vec)
        out = {}
        k = self.kern
        for a in 'xy':
            fF, fB = k.fwd(g['F', a]), k.fwd(g['B', a])
            out['F', a] = fft.ifft2(fF * k.k[a, 0.0] + fB * k.k[a, DZ], workers=-1)[:H, :W]
            out['B', a] = fft.ifft2(fB * k.k[a, 0.0] + fF * k.k[a, DZ], workers=-1)[:H, :W]
        return self.gather(out)

    def flux_from_part(self, pat):
        """Flux linked per sheet element, and the pattern's FFTs, for a part's unit line current."""
        px, py = pat
        grids = {}
        for a, g in (('x', px), ('y', py)):
            f = self.kern.fwd(g)
            grids['F', a] = self.kern.conv(f, a, Z_PART)
            grids['B', a] = self.kern.conv(f, a, DZ + Z_PART)
        return self.gather(grids)

    def part_mutual(self, p1, p2):
        return float(sum(np.sum(p1[k] * self.kern.conv(self.kern.fwd(p2[k]), a, 0.0).real) for k, a in ((0, 'x'), (1, 'y'))))


def solve_scenario(net, f, prescribed, branches):
    w = 2 * math.pi * f
    ne, nb, n = net.ne, len(branches), net.n
    pats_s = [net.branch_pattern(*p) for p in prescribed]
    pats_b = [net.branch_pattern(*p) for p in branches]
    a_src = sum(net.flux_from_part(p) for p in pats_s)                 # flux from the prescribed 1 A
    a_b = [net.flux_from_part(p) for p in pats_b]
    l_bb = np.array([[net.part_mutual(p, q) for q in pats_b] for p in pats_b])
    l_bs = np.array([sum(net.part_mutual(p, q) for q in pats_s) for p in pats_b])
    l_ss = sum(net.part_mutual(p, q) for p in pats_s for q in pats_s)
    z_b = np.array([0j if a.startswith('D') else CAPS[a.split('.')[0]][1] + 1 / (1j * w * CAPS[a.split('.')[0]][0])
                    for a, _ in branches])

    # incidence of the solved branches, and the prescribed currents' injections
    rows, cols, vals = [], [], []
    for k, (a, b) in enumerate(branches):
        na, nbb = net.pad_nodes(a), net.pad_nodes(b)
        rows += [na, nbb]; cols += [np.full(len(na), k), np.full(len(nbb), k)]
        vals += [np.full(len(na), 1 / len(na)), np.full(len(nbb), -1 / len(nbb))]
    Db = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, nb))
    s = np.zeros(n)
    for a, b in prescribed:
        na, nbb = net.pad_nodes(a), net.pad_nodes(b)
        s[na] -= 1 / len(na)
        s[nbb] += 1 / len(nbb)

    # one reference node per connected piece (pieces joined only by prescribed parts float)
    Dall = sparse.hstack([net.D, Db]).tocsr()
    adj = (abs(Dall) @ abs(Dall).T).tocsr()
    ncomp, comp = connected_components(adj, directed=False)
    ref = np.zeros(n, bool)
    ref[np.unique(comp, return_index=True)[1]] = True
    free = ~ref
    Df = Dall[free]
    nf = free.sum()

    z0 = np.concatenate([net.z_el + 1j * w * net.l_el, z_b + 1j * w * np.diag(l_bb)])
    S0 = (Df @ sparse.diags(1 / z0) @ Df.T).tocsc()
    lu = splu(S0)
    ns = net.n_sheet
    DfT_e, DfT_b, Df_e, Df_b = Df[:, :ne].T.tocsr(), Df[:, ne:].T.tocsr(), Df[:, :ne].tocsr(), Df[:, ne:].tocsr()

    def matvec(x):
        i_e, i_b, phi = x[:ne], x[ne:ne + nb], x[ne + nb:]
        ve = net.z_el * i_e
        ve[:ns] += 1j * w * net.flux_sheet(i_e[:ns])[:ns]
        ve[ns:] += 1j * w * net.l_el[ns:] * i_e[ns:]
        for a, ib in zip(a_b, i_b):
            ve += 1j * w * a * ib
        vb = z_b * i_b + 1j * w * (np.array([a @ i_e for a in a_b]) + l_bb @ i_b)
        return np.concatenate([ve - DfT_e @ phi, vb - DfT_b @ phi, Df_e @ i_e + Df_b @ i_b])

    def precond(r):
        r1, r2 = r[:ne + nb], r[ne + nb:]
        y = lu.solve(r2 - Df @ (r1 / z0))
        return np.concatenate([(r1 + Df.T @ y) / z0, y])

    N = ne + nb + nf
    rhs = np.concatenate([-1j * w * a_src, -1j * w * l_bs, s[free]])
    it = [0]
    t0 = time.time()
    x, info = gmres(LinearOperator((N, N), matvec, dtype=complex), rhs.astype(complex),
                    M=LinearOperator((N, N), precond, dtype=complex),
                    rtol=1e-7, restart=60, maxiter=40, callback=lambda _: it.__setitem__(0, it[0] + 1),
                    callback_type='pr_norm')
    res = np.linalg.norm(matvec(x) - rhs) / np.linalg.norm(rhs)
    print(f'    gmres: {it[0]} iterations, {time.time() - t0:.0f} s, residual {res:.1e}, info {info}')
    i_e, i_b = x[:ne], x[ne:ne + nb]

    # loop impedance by Tellegen: power the 1 A sources deliver into everything else
    r_cu = float(np.sum(net.z_el.real * abs(i_e) ** 2))
    r_loop = r_cu + float(np.sum(z_b.real * abs(i_b) ** 2))
    flux = net.flux_sheet(i_e[:ns])[:ns]
    l_loop = (np.vdot(i_e[:ns], flux) + np.sum(net.l_el[ns:] * abs(i_e[ns:]) ** 2)
              + 2 * np.real(np.vdot(i_e, a_src) + sum(np.vdot(i_e, a) * ib for a, ib in zip(a_b, i_b)))
              + np.vdot(i_b, l_bb @ i_b) + 2 * np.real(np.vdot(i_b, l_bs)) + l_ss)
    return i_e, i_b, r_cu, r_loop, float(l_loop.real)


def current_density(net, i_e):
    g = net.scatter(i_e)
    out = {}
    for L in 'FB':
        jx, jy = np.zeros((H, W), complex), np.zeros((H, W), complex)
        ex, ey = g[L, 'x'], g[L, 'y']
        jx[:, :-1] += ex[:, :-1] / 2; jx[:, 1:] += ex[:, :-1] / 2
        jy[:-1, :] += ey[:-1, :] / 2; jy[1:, :] += ey[:-1, :] / 2
        jx *= S; jy *= S                                  # A per mm of copper width
        jx[~net.cu[L]] = np.nan; jy[~net.cu[L]] = np.nan
        out[L] = (jx, jy)
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    freq = dict(kv.split('=') for a, kv in zip(sys.argv[1:], sys.argv[2:]) if a == '--f')
    names = [a for a in args if a in SCENARIOS] or list(SCENARIOS)
    src = next((v for k, v in zip(sys.argv[1:], sys.argv[2:]) if k == '--src'), 'gerber')
    pcb = B.load_pcb()
    kern = Kernels()
    out = {}
    for name in names:
        f, nets, prescribed, branches = SCENARIOS[name]
        f = float(freq.get(name, f))
        net = AcNetwork(src, pcb, nets, kern)
        print(f'[{src}] {name} @ {f / 1e6:g} MHz: {net.n} nodes, {net.ne} elements, nets {nets}')
        i_e, i_b, r_cu, r_loop, l_loop = solve_scenario(net, f, prescribed, branches)
        J = current_density(net, i_e)
        np.savez_compressed(B.work_path(f'Jac_{src}_{name}.npz'),
                            FX=J['F'][0], FY=J['F'][1], BX=J['B'][0], BY=J['B'][1], f=f)
        vias = [(x, y, vnet, complex(i_e[e].sum())) for x, y, vnet, e in net.vias]
        out[name] = {'f': f, 'R_copper_mohm': r_cu * 1e3, 'R_loop_mohm': r_loop * 1e3, 'L_loop_nH': l_loop * 1e9,
                     'branches': {f'{a}->{b}': [abs(ib), math.degrees(np.angle(ib))] for (a, b), ib in zip(branches, i_b)},
                     # current changing layer (+ = top -> bottom), complex against the 1 A source
                     'vias': [(x, y, vnet, c.real, c.imag) for x, y, vnet, c in vias if abs(c) >= 0.04]}
        print(f'    loop: R {r_cu * 1e3:.2f} mOhm copper ({r_loop * 1e3:.2f} with the caps\' ESR), L {l_loop * 1e9:.3f} nH')
        for (a, b), ib in zip(branches, i_b):
            print(f'    {a}->{b}: {abs(ib):.3f} A at {math.degrees(np.angle(ib)):+.0f} deg')
    B.work_path(f'solve_ac_{src}.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
