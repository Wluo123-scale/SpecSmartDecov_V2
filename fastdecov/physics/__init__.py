"""Sample/instrument physics read from the manuscript (one JSON per sample family), not hardcoded in the fitter."""
import json
from pathlib import Path

HC_EV_NM = 1239.841984
DEFAULT = Path(__file__).with_name("gase_pillar_4K.json")


def load(path=None):
    return json.loads(Path(path or DEFAULT).read_text())


def instrument_fwhm_nm(wavelength_nm, phys=None):
    """Instrument FWHM in nm at a wavelength, from a resolution given in energy: dlambda = lambda^2 dE / hc."""
    p = phys or load()
    return wavelength_nm ** 2 * p["instrument_resolution_meV"] * 1e-3 / HC_EV_NM
