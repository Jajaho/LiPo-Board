# Boost converter return-path analysis

Layout review of the MT3608 boost stage on the 2019 board: where its currents flow in the ground copper, how large the switching loop is, and how much of the return-path voltage reaches the feedback divider.

Open `boost-return-paths.html` in a browser for the interactive result: animated current flow and current-density maps per layer for the switching loop, input loop and DC power path, the ground potential seen by the feedback divider, the top-layer ground islands, and the findings with fixes. `boost-return-paths-ac.html` is the same page with the switching-loop and input-loop maps from the AC solve (below) instead of the DC one.

## Run it

```sh
uv run analysis/boost-return-paths-python/run_all.py
```

The run takes about 4 min (the AC solve is most of it), needs numpy, scipy, pillow and contourpy (declared inline, so `uv` installs them), and writes intermediate arrays to `work/` (ignored by git). Each step can also be run on its own in this order: `connectivity.py`, `solve.py`, `solve_ac.py`, `loop_area.py`, `gen_viz.py`.

| File | Does |
|---|---|
| `board.py` | Paths, constants, KiCad s-expression reader, EAGLE Gerber parser, rasterisers |
| `connectivity.py` | Splits each copper layer into connected pieces and assigns nets, for the Gerbers and for the KiCad fill |
| `solve.py` | Resistive current-spreading solve in the ground copper of both layers for three scenarios |
| `solve_ac.py` | R + jωL solve of the switching loop (100 MHz) and input loop (1.2 MHz): current crowding, loop inductance, capacitor split |
| `loop_area.py` | Effective switching-loop area from the solved current |
| `gen_viz.py` | Heatmaps, flow field and geometry, merged into `template.html` to produce `boost-return-paths.html` and `boost-return-paths-ac.html` |

## Inputs

- **Copper as fabricated:** `outputs/LiPo-BoardV1_2019-01-18.zip` (EAGLE 9.2.2 production Gerbers), read directly from the zip.
- **Nets, pads, tracks, component outlines:** `src/kicad-2019/LiPo-BoardV1-2019.kicad_pcb`. Its tracks match the Gerbers.
- **Cross-check:** the same solve runs on the zone fill stored in the KiCad file.

## Model

- Both copper layers are rasterised at 0.05 mm (900 × 320 cells per layer) and only the ground copper is solved, as a sheet-resistor network of about 395,000 nodes.
- 35 µm copper (0.49 mΩ/□). Vias: 0.35 mm drill, 20 µm plating, 1.25 mΩ each. Plated pins are larger barrels; J3 is treated as soldered through (0.2 mΩ).
- Currents are injected and removed over the pad areas:

| Scenario | In | Out |
|---|---|---|
| `hot`: switching-loop return | 1 A at C3 pin 2 | IC1 pin 2 |
| `input`: inductor-ripple return | 1 A at IC1 pin 2 | C1 pin 2 and C4 pin 2, both held at 0 V |
| `dc`: 1 A load, ideal 3.7 V → 5.1 V | 1 A at J3, 0.5 A at IC1 pin 2 | 1.5 A at Q1 pin 3 |

- Loop area is the magnetic moment of the forward path (traces as line currents) plus the solved return current, divided by the loop current.

### AC solve (`solve_ac.py`)

Above about R/(2πL) ≈ 0.6 MHz the switching loop is inductive: the return no longer spreads by resistance but crowds under its forward path. `solve_ac.py` solves that regime on the same raster as a quasi-static PEEC model:

- Every cell edge of the solved nets' copper (both layers) is a current element with one square of resistance and a partial inductance to every parallel element (Neumann formula, self term regularised by the cross-section's geometric mean distance), applied by FFT. Vias are vertical elements with their own partial inductance.
- The forward copper is solved too (SW, 5V or BAT+), not drawn as a line.
- Parts are straight line currents between their pad centres, 0.5 mm above the top copper. The IC switch and L1 carry a prescribed 1 A; C1, C4, C3 and D1 are solved branches (22 µF / 5 mΩ, 10 µF / 10 mΩ, 22 µF / 5 mΩ, short).
- The kernel reproduces the analytic partial inductance of straight strips to within 1 %.

| Scenario | Frequency | Loop | Solved |
|---|---|---|---|
| `hot` | 100 MHz (edges) | IC1 pin 2 → switch → SW → D1 → 5V → C3 → GND | GND, 5V, SW copper |
| `input` | 1.2 MHz (switching frequency) | C1/C4 → BAT+ → L1 → SW → switch → IC1 pin 2 → GND | GND, BAT+, SW copper |

In `boost-return-paths-ac.html` the switching-loop and input-loop maps and via shares come from this solve (amplitude for the heatmap, the real part for the flow direction); the DC path and the feedback-potential view stay resistive. Its prose and fixed figures are still the DC page's.

## Results

| | As fabricated (Gerbers) | KiCad-2019 fill |
|---|---|---|
| Top-layer ground | 3 islands: IC1 GND + C4 (81 mm²), main pour (330 mm²), under SW (29 mm²) | joined by thin edge slivers |
| Return resistance C3− → IC1 pin 2 | 4.4 mΩ | 3.8 mΩ |
| R3 ground, share of that drop | 68 % | 73 % |
| Switching-loop area, DC spread | 179 mm² | 180 mm² |
| R3 ground above IC1 ground at 1 A load | 0.44 mV (−3.3 mV at the output) | 0.69 mV |

For comparison, the tightest possible return route through the J3 vias gives 93 mm² and a direct top-layer connection from C3− to IC1 pin 2 gives 49 mm².

AC solve, as fabricated (KiCad fill in brackets):

| | |
|---|---|
| Switching-loop inductance at 100 MHz | 13.6 nH (14.1 nH) |
| Switching-loop copper resistance at 100 MHz, current crowded | 38 mΩ |
| Switching return dropping to the bottom near J3 / rising at IC1 pin 2 | 69 % / 91 % (DC: 58 % / 86 %) |
| Input-loop inductance at 1.2 MHz | 5.9 nH |
| Input ripple through C1 / C4 | 0.44 A / 0.60 A per 1 A, 31° apart |

## Limits

- The DC solve overstates spreading at switching-edge frequencies; the AC solve covers that, but it is quasi-static (no radiation or dielectric) and part ESL beyond the pad-to-pad line is not included.
- An openEMS full-wave cross-check (`../boost-return-paths-circuitrf/hot-loop-3d`) is not yet valid: its first run shorted SW to the ground pour under D1 and its grid has one cell through the FR4.
- The example FB spike in the page (0.18 V) uses an assumed 2 nH of shared return and a 1.5 A edge in 10 ns; it is not computed.
- Not benchmarked against a commercial field solver. A probe measurement on SW and FB is the real test.
- The prose and key figures in `template.html` were written by hand from this output; rerunning updates the data and the computed figures in the page, not the prose.
