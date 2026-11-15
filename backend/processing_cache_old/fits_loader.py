from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


# ============================================================
# EXTENSION IDENTIFICATION
# ============================================================

SCIENCE_NAMES = {
    "IMAGE",
    "SCI",
    "SCIENCE",
}

VARIANCE_NAMES = {
    "VARIANCE",
    "VAR",
}

FLAG_NAMES = {
    "FLAGS",
    "FLAG",
    "DQ",
    "MASK",
}


def identify_hdus(hdul):
    """
    Identify science, variance and quality flag HDUs.
    """

    science = None
    variance = None
    flags = None

    image_candidates = []

    for index, hdu in enumerate(hdul):

        if hdu.data is None:
            continue

        name = hdu.name.upper()

        if name in VARIANCE_NAMES:
            variance = index
            continue

        if name in FLAG_NAMES:
            flags = index
            continue

        if name in SCIENCE_NAMES:
            science = index
            continue

        if isinstance(hdu.data, np.ndarray):
            if hdu.data.ndim == 2:
                image_candidates.append(index)

    # Prefer an explicitly named science extension.
    # Otherwise use the largest 2D image extension.
    if science is None and image_candidates:

        science = max(
            image_candidates,
            key=lambda index: hdul[index].data.size
        )

    # Also support a 2D image in the primary HDU.
    if science is None:
        if hdul[0].data is not None:
            if hdul[0].data.ndim == 2:
                science = 0

    if science is None:
        raise ValueError(
            "Could not identify a 2D science image."
        )

    return science, variance, flags


# ============================================================
# LOAD FITS FILE
# ============================================================

def load_fits(path):
    """
    Load a SPHEREx FITS observation.

    Returns:
        Dictionary containing:
        science
        variance
        flags
        wcs
        header
        metadata
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".part":
        raise ValueError("Incomplete download cannot be loaded.")

    with fits.open(
        path,
        memmap=False,
        lazy_load_hdus=False
    ) as hdul:

        hdul.verify("exception")

        science_index, variance_index, flags_index = (
            identify_hdus(hdul)
        )

        science_hdu = hdul[science_index]

        science = np.asarray(
            science_hdu.data,
            dtype=np.float32
        ).copy()

        if science.ndim != 2:
            raise ValueError(
                f"Science image must be 2D, got {science.shape}"
            )

        header = science_hdu.header.copy()

        # Find celestial WCS.
        wcs = WCS(header).celestial

        if not wcs.has_celestial:
            raise ValueError(
                "Science image has no celestial WCS."
            )

        variance = None

        if variance_index is not None:

            variance = np.asarray(
                hdul[variance_index].data,
                dtype=np.float32
            ).copy()

            if variance.shape != science.shape:
                raise ValueError(
                    "Variance dimensions do not match science image."
                )

        flags = None

        if flags_index is not None:

            flags = np.asarray(
                hdul[flags_index].data
            ).copy()

            if flags.shape != science.shape:
                raise ValueError(
                    "Flag dimensions do not match science image."
                )

        metadata = {
            "filename": path.name,
            "science_hdu": science_index,
            "variance_hdu": variance_index,
            "flags_hdu": flags_index,
            "shape": list(science.shape),
            "bunit": header.get("BUNIT"),
            "date_obs": header.get("DATE-OBS"),
            "mjd_obs": header.get("MJD-OBS"),
            "exptime": header.get("EXPTIME"),
            "instrument": header.get("INSTRUME"),
            "telescope": header.get("TELESCOP"),
            "object": header.get("OBJECT"),
        }

    return {
        "science": science,
        "variance": variance,
        "flags": flags,
        "wcs": wcs,
        "header": header,
        "metadata": metadata,
    }


# ============================================================
# QUALITY MASK
# ============================================================

def create_quality_mask(
    science,
    variance=None,
    flags=None,
    excluded_flag_bits=None
):
    """
    Create a Boolean mask.

    True  = invalid pixel
    False = usable pixel

    Excluded flag bits must be selected using the
    actual SPHEREx flag definitions.
    """

    invalid = ~np.isfinite(science)

    if variance is not None:

        invalid |= ~np.isfinite(variance)
        invalid |= variance <= 0

    if flags is not None and excluded_flag_bits:

        for bit in excluded_flag_bits:

            invalid |= (
                (flags.astype(np.uint64) & (1 << bit)) != 0
            )

    return invalid


if __name__ == "__main__":

    print("NightLock FITS Loader")
    print("Import load_fits() from this module.")