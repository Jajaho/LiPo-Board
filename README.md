# Battery Management and Boost Board

This PCB combines a single cell 3.7 V LiPo-battery charger and protection circuitry with a 5 V boost converter to deliver up to 1 A of current for small projects.
The main components are:
- **TP4056** 1 A standalone linear Li-Ion battery charger with thermal protection in SOP-8 package
- **MT3608** High efficiency 1.2 MHz 2 A boost converter

You can check out the post on my website [here](https://jajaho.github.io//projects/lipo_booster_charger) if you like.

## Repository Contents

The project was originally designed and manufactured in 01/2019 using Eagle ECAD. Later they have been converted to KiCad to preserve them. 
The KiCad design files represent the original files as closely as possible.

- `src/eagle-2019` - The original design files which produced the production files of the manufactured PCB are in
- `src/kicad-2019` - Converted design files which produced the production files of the manufactured PCB are in

The original production files are archived in `outputs/LiPo-BoardV1_2019-01-18.zip`.

The faulty wiring of the enable switch was discovered only after the board has been manufactured. Consequently this mistake has been corrected in the schematic but the board has not been rerouted.

- `src/eagle` - A later state of the design files, with incomplete routing after the DPST switch was rewired.
- `src/kicad` - A later state of the converted design files, with incomplete routing after the DPST switch was rewired.

Schematic PDFs of both KiCad projects are available in the generated `schematic-pdfs` artifact.
