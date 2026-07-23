import os
import sys
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["POLARS_SKIP_CPU_CHECK"] = "1"

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

modules = [
    "scraper",
    "intelligence",
    "market_pipeline",
    "features",
    "ml_pipeline",
    "calibration",
    "monitoring",
    "analytics",
    "backtest",
]

results = []
for mod in modules:
    try:
        __import__(mod)
        results.append((mod, "OK"))
    except Exception as e:
        import traceback
        results.append((mod, f"FAIL: {e}\n{traceback.format_exc()}"))

print("=" * 60)
print("MODULE IMPORT VERIFICATION RESULTS")
print("=" * 60)
for mod, status in results:
    print(f"  {mod:25s} -> {status}")

failed = [r for r in results if not r[1].startswith("OK")]
if failed:
    print(f"\n{len(failed)} module(s) failed to import.")
    sys.exit(1)
else:
    print(f"\nAll {len(results)} modules imported successfully.")
    sys.exit(0)
