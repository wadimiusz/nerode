"""scripts/belief_by_gap.py for the one-layer models (24.11.2026): same measurement, the loader
swapped for object_form.load (one layer, 100k steps, fresh data)."""
import sys

sys.path.insert(0, "scripts")
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import belief_by_gap  # noqa: E402
from object_form import load  # noqa: E402

belief_by_gap.load_models = lambda kind, seed, steps, batch, min_sep: load(kind, seed)
if __name__ == "__main__":
    belief_by_gap.main()
