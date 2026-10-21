# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy>=2", "scipy>=1.13", "pillow>=10", "contourpy>=1.2"]
# ///
"""Run the whole boost return-path analysis: uv run analysis/boost-return-paths-python/run_all.py"""
import connectivity
import gen_viz
import loop_area
import solve

if __name__ == '__main__':
    connectivity.main()
    solve.main()
    loop_area.main()
    gen_viz.main()
