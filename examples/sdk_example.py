# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""Minimal SDK example: screen a map, then print the follow-up shortlist (lines to re-measure at higher resolution)."""
import sys

from fastdecov import api

if __name__ == "__main__":
    m = api.load_map(sys.argv[1])
    print("map shape:", {k: v for k, v in api.detect_grid(m).items() if k in ("rows", "cols", "snake", "method", "reason")})
    res = api.fit_map(m, method="v3")
    out = api.save(res, sys.argv[2] if len(sys.argv) > 2 else "out_sdk", grid="auto")
    for r in out["ranking"]:
        ls = r.get("line_shape") or {}
        print(f"{r['center_nm']:8.2f} nm  {r['kind']:13s} nu {r['nu_median']:.2f} -> intrinsic {ls.get('nu_intrinsic', '-')}  "
              f"FWHM {r['fwhm_nm']:.2f} nm  g2 priority {r['g2_priority']}")
    print("\nfollow-up shortlist:")
    for it in out["shortlist"]:
        print(f"  {it['rank']}. tier {it['tier']}  {it['center_nm']} nm  grating {it['grating']}  "
              f"pixels {', '.join(p['spectrum_id'] for p in it['best_pixels'])}  ({it['why']})")
