# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
"""fastdecov: fast emitter discovery and deconvolution of photoluminescence spectrum maps (SpecSmartDecov V2).

CLI:  fastdecov fit SPECTRA_DIR OUT_DIR [--method v3|v2|legacy] [--grid 11x11] [--figures]
SDK:  from fastdecov import api; res = api.fit_map(api.load_map(SPECTRA_DIR)); api.save(res, OUT_DIR)
"""
__version__ = "2.5.0"
