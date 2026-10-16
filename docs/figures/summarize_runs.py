# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Summarize finished `fastdecov fit` runs of one map into a small JSON for make_figures.py.

    python docs/figures/summarize_runs.py OUT.json NAME=RUN_DIR [NAME=RUN_DIR ...]

Each RUN_DIR is an OUTPUT_DIR of `fastdecov fit` (it needs basis.json, run.json and emitters.tsv). The JSON keeps only
what the README needs: emitter centres (and whether each is patch-confined), how many spectra each emitter is
present in (zero-phonon-line peak SNR >= 5), the fit quality and the run time. It holds no spectra.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def summarize(run):
    run = Path(run)
    basis = json.loads((run / "basis.json").read_text())
    meta = json.loads((run / "run.json").read_text())
    q = meta["fit_quality"]
    present = defaultdict(int)
    with open(run / "emitters.tsv") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if float(r["zpl_peak_snr"] or 0) >= 5:
                present[int(r["emitter"])] += 1
    centres = basis.get("reference_centers") or basis["centers"]
    local = basis.get("local") or [False] * basis["K"]
    return dict(method=basis["method"], K=basis["K"], n_spectra=meta["n_spectra"], threads=meta["threads"],
                local_emitters=meta.get("local_emitters"), total_s=round(meta["total_s"], 1), fit_s=round(meta["fit_s"], 1),
                pass_rate=round(q["pass_rate"], 3), rchi2_median=round(q["rchi2_median"], 2),
                residual_limit_pass=round(q["residual_limit"]["pass_fraction"], 3),
                unexplained_peak_spectra=q["audit"]["n_spectra"], min_sep_nm=basis.get("min_sep_nm"),
                window_nm=basis.get("window_nm"),
                emitters=[dict(center_nm=round(float(c), 2), local=bool(l), spectra=present.get(k + 1, 0))
                          for k, (c, l) in enumerate(zip(centres, local))])


def main(argv):
    out, runs = argv[1], [a.split("=", 1) for a in argv[2:]]
    Path(out).write_text(json.dumps({name: summarize(d) for name, d in runs}, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv)
