"""Reading spectra directories and writing the canonical outputs."""
import csv
import json
from pathlib import Path

import numpy as np

from . import model

RESULT_COLUMNS = ["spectrum_id", "component_id", "center_nm", "nu", "width_nm", "area", "fit_status"]
DIAG_COLUMNS = ["spectrum_id", "sse", "n_iter", "converged", "best_start", "background", "background_slope_per_nm"]


def read_directory(input_dir):
    paths = sorted(Path(input_dir).glob("*.txt"))
    if not paths:
        raise ValueError(f"no *.txt spectra in {input_dir}")
    ids, xs, ys = [], [], []
    for path in paths:
        x, y = model.load_spectrum(path)
        ids.append(path.stem); xs.append(x); ys.append(y)
    return ids, xs, ys


def write_results(res, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    areas = res.areas
    with (out / "fit_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(RESULT_COLUMNS)
        for s, sid in enumerate(res.ids):
            pk = res.params[s, res.nbg:].reshape(res.K, 4)
            for k in range(res.K):
                w.writerow([sid, k + 1, f"{pk[k, 1]:.10g}", f"{pk[k, 2]:.10g}", f"{pk[k, 3]:.10g}",
                            f"{areas[s, k]:.10g}", "ok"])
    with (out / "diagnostics.csv").open("w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(DIAG_COLUMNS)
        for s, sid in enumerate(res.ids):
            w.writerow([sid, f"{res.sse[s]:.10g}", int(res.n_iter[s]), int(res.converged[s]),
                        res.best_start[s], f"{res.params[s, 0]:.10g}",
                        f"{(res.params[s, 1] / model.X_SCALE if res.nbg == 2 else 0.0):.10g}"])


def write_runtime(output_dir, payload):
    (Path(output_dir) / "runtime.json").write_text(json.dumps(payload, indent=2) + "\n")
