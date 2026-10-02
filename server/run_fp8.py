"""python run_fp8.py <script.py> [args...]: run a script with the FP8 fallback installed (see fp8_fallback.py)."""
import runpy, sys
import fp8_fallback  # noqa: F401  (patches transformers' FP8 linear)
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
