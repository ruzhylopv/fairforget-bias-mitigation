"""Find the layers with the largest rotation part of the forget-loss drop and largest total drop.

Reads `sweep.csv` from the adaptive RMU layer sweep (see 06.ipynb). With β = 1 the relative
forget-loss drop splits as 1 − L/L₀ ≈ shrinkage + rotation, so "rotation" is reported both as the
absolute term k·cosθ and as its share of shrinkage + rotation.

    uv run python find_max_rotation_and_loss_drop_layers.py [path/to/sweep.csv]
"""

import sys
from pathlib import Path

import pandas as pd

csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sweep.csv"
sweep = pd.read_csv(csv_path).sort_values("layer")
sweep["forget_rotation_share"] = sweep["forget_rotation"] / (
    sweep["forget_shrinkage"] + sweep["forget_rotation"]
)

best = {
    "largest rotation term": sweep.loc[sweep["forget_rotation"].idxmax()],
    "largest rotation share": sweep.loc[sweep["forget_rotation_share"].idxmax()],
    "largest overall loss drop": sweep.loc[sweep["forget_drop"].idxmax()],
}
columns = ["layer", "forget_drop", "forget_shrinkage", "forget_rotation", "forget_rotation_share"]
table = pd.DataFrame(best).T[columns]
table["layer"] = table["layer"].astype(int)

print(f"Source: {csv_path}\n")
print(table.round(4).to_string())
