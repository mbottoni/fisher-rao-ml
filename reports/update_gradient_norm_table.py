"""Print updated Table 6 LaTeX rows from gradient_norm_full.csv.

Run after gradient_norm_analysis.py --seeds 3 completes. Outputs the midrule
rows for Table 6 (tab:grad_ratio) with means across all available seeds.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path("reports/results")
EPOCHS = [0, 10, 20, 30, 40, 50, 59]
OBJ_ORDER = ["kl", "fisher_rao", "hellinger", "gce", "mae", "sce"]
OBJ_LABELS = {
    "kl": "KL (CE)    ", "fisher_rao": "Fisher--Rao", "hellinger": "Hellinger  ",
    "gce": "GCE        ", "mae": "MAE        ", "sce": "SCE        ",
}
ANNOTATIONS = {
    "kl": r" $\uparrow$",
    "fisher_rao": r" $\approx 1$",
    "hellinger": r" $\approx 1$",
    "gce": r" $\downarrow$",
    "mae": r" erratic",
    "sce": r" erratic",
}


def main() -> None:
    path = RESULTS / "gradient_norm_full.csv"
    rows = list(csv.DictReader(path.open()))

    # data[obj][seed][epoch][sample_type] = grad_norm
    data: dict[str, dict[int, dict[int, dict[str, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for r in rows:
        data[r["objective"]][int(r["seed"])][int(r["epoch"])][r["sample_type"]] = float(
            r["mean_grad_norm"]
        )

    seeds_available = sorted({int(r["seed"]) for r in rows})
    n_seeds = len(seeds_available)
    print(f"% Seeds available: {seeds_available} (n={n_seeds})")
    print()
    print("% Table 6 rows (replace \\midrule...\\bottomrule content):")
    print()

    for obj in OBJ_ORDER:
        if obj not in data:
            print(f"    % {obj}: no data")
            continue

        cells = []
        for ep in EPOCHS:
            ratios = []
            for seed in seeds_available:
                if ep not in data[obj][seed]:
                    continue
                ep_data = data[obj][seed][ep]
                if "clean" in ep_data and "noisy" in ep_data:
                    ratio = ep_data["noisy"] / ep_data["clean"]
                    ratios.append(ratio)
            if ratios:
                mean_ratio = np.mean(ratios)
                cells.append(f"{mean_ratio:.2f}")
            else:
                cells.append("--")

        final_cell = cells[-1] if cells else "--"
        annotation = ANNOTATIONS.get(obj, "")
        cells[-1] = f"\\textbf{{{final_cell}}}{annotation}"
        label = OBJ_LABELS.get(obj, obj)
        row = f"    {label} & " + " & ".join(cells) + r" \\"
        print(row)

    print()
    print(f"% Caption update: change 'seed\\,0' to '{n_seeds} seeds' if n_seeds > 1")
    print(f"% Conclusion update: change 'seed\\,0' to '{n_seeds} seeds' in §5 and §3.4")


if __name__ == "__main__":
    main()
