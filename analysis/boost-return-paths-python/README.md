# Boost converter return-path analysis

Layout review of the MT3608 boost stage on the 2019 board: where its currents flow in the ground copper, how large the switching loop is, and how much of the return-path voltage reaches the feedback divider.

Open `boost-return-paths.html` in a browser for the interactive result: animated current flow and current-density maps per layer for the switching loop, input loop and DC power path, the ground potential seen by the feedback divider, the top-layer ground islands, and the findings with fixes.

## Run it

```sh
uv run analysis/boost-return-paths-python/run_all.py
```

The run takes about 20 s, needs numpy, scipy, pillow and contourpy (declared inline, so `uv` installs them), and writes intermediate arrays to `work/` (ignored by git). Each step can also be run on its own in this order: `connectivity.py`, `solve.py`, `loop_area.py`, `gen_viz.py`.

| File | Does |
|---|---|
| `board.py` | Paths, constants, KiCad s-expression reader, EAGLE Gerber parser, rasterisers |
| `connectivity.py` | Splits each copper layer into connected pieces and assigns nets, for the Gerbers and for the KiCad fill |
| `solve.py` | Resistive current-spreading solve in the ground copper of both layers for three scenarios |
| `loop_area.py` | Effective switching-loop area from the solved current |
| `gen_viz.py` | Heatmaps, flow field and geometry, merged into `template.html` to produce `boost-return-paths.html` |

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

## Results

| | As fabricated (Gerbers) | KiCad-2019 fill |
|---|---|---|
| Top-layer ground | 3 islands: IC1 GND + C4 (81 mm²), main pour (330 mm²), under SW (29 mm²) | joined by thin edge slivers |
| Return resistance C3− → IC1 pin 2 | 4.4 mΩ | 3.8 mΩ |
| R3 ground, share of that drop | 68 % | 73 % |
| Switching-loop area, DC spread | 179 mm² | 180 mm² |
| R3 ground above IC1 ground at 1 A load | 0.44 mV (−3.3 mV at the output) | 0.69 mV |

For comparison, the tightest possible return route through the J3 vias gives 93 mm² and a direct top-layer connection from C3− to IC1 pin 2 gives 49 mm².

## Limits

- This is a DC (resistive) solution. At switching-edge frequencies the return current crowds closer to the forward path, so the field overstates spreading; it still has to pass through the same islands and vias, which is why the loop area is quoted as 93–180 mm².
- The example FB spike in the page (0.18 V) uses an assumed 2 nH of shared return and a 1.5 A edge in 10 ns; it is not computed.
- Not benchmarked against a commercial field solver. A probe measurement on SW and FB is the real test.
- The prose and key figures in `template.html` were written by hand from this output; rerunning updates the data and the computed figures in the page, not the prose.
