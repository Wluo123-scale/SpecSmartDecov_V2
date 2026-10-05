"""Minimal SDK example: discover emitters on a map and list the g2-priority candidates."""
import sys

from fastdecov import api

if __name__ == "__main__":
    m = api.load_map(sys.argv[1])
    res = api.fit_map(m, method="v3")
    out = api.save(res, sys.argv[2] if len(sys.argv) > 2 else "out_sdk")
    for r in out["ranking"]:
        print(f"{r['center_nm']:8.2f} nm  nu {r['nu_median']:.2f}  FWHM {r['fwhm_nm']:.2f} nm  sideband {r['phonon_sideband']}  "
              f"g2 priority {r['g2_priority']}")
