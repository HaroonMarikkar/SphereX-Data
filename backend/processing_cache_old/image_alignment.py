from pathlib import Path
import argparse
import json

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy import units as u

from reproject import reproject_interp

from .fits_loader import load_fits


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DOWNLOAD_DIR = (
    ROOT / "data_pipeline"
    / "downloads"
    / "challenge_targets"
)

OUTPUT_DIR = (
    ROOT / "data_pipeline"
    / "processed"
    / "aligned"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Target coordinates are initial reference positions.
# For high proper-motion targets, use the propagated
# position for the selected observation epoch.

TARGET_COORDINATES = {
    "barnards_star": (269.452077, 4.693365),
    "epsilon_indi_b": (331.044146, -56.782844),
    "wise_0855": (133.795129, -7.245147),
    "rw_aurigae": (76.956526, 30.401438),
    "r_draconis": (248.167611, 66.754979),
}


# ============================================================
# BUILD REFERENCE GRID
# ============================================================

def build_reference_grid(
    reference_path,
    center,
    size_arcmin=6,
    pixel_scale_arcsec=None
):

    observation = load_fits(reference_path)

    image = observation["science"]
    wcs = observation["wcs"]

    center_coord = SkyCoord(
        center[0] * u.deg,
        center[1] * u.deg,
        frame="icrs"
    )

    if pixel_scale_arcsec is None:

raw_scales = wcs.proj_plane_pixel_scales()

scales = []

for index, scale in enumerate(raw_scales):

    if hasattr(scale, "to_value"):
        value = scale.to_value(u.arcsec)

    else:
        unit = wcs.wcs.cunit[index]

        if not unit:
            unit = u.deg

        value = (float(scale) * unit).to_value(u.arcsec)

    scales.append(abs(float(value)))

pixel_scale_arcsec = float(np.mean(scales))

    pixel_count = int(
        size_arcmin * 60 / pixel_scale_arcsec
    )

    shape = (
        pixel_count,
        pixel_count
    )

    reference_cutout = Cutout2D(
        image,
        position=center_coord,
        size=(
            size_arcmin * u.arcmin,
            size_arcmin * u.arcmin
        ),
        wcs=wcs,
        mode="partial",
        fill_value=np.nan
    )

    reference_wcs = reference_cutout.wcs

    return reference_wcs, shape, pixel_scale_arcsec


# ============================================================
# ALIGN ONE IMAGE
# ============================================================

def align_image(
    input_path,
    reference_wcs,
    output_shape,
    output_path
):

    observation = load_fits(input_path)

    science = observation["science"]
    variance = observation["variance"]
    flags = observation["flags"]
    source_wcs = observation["wcs"]

    # Reproject science image.
    aligned_science, footprint = reproject_interp(
        (science, source_wcs),
        reference_wcs,
        shape_out=output_shape,
        order="bilinear",
        return_footprint=True
    )

    aligned_variance = None

    if variance is not None:

        aligned_variance, _ = reproject_interp(
            (variance, source_wcs),
            reference_wcs,
            shape_out=output_shape,
            order="nearest-neighbor",
            return_footprint=True
        )

    aligned_flags = None

    if flags is not None:

        aligned_flags, _ = reproject_interp(
            (flags.astype(np.float32), source_wcs),
            reference_wcs,
            shape_out=output_shape,
            order="nearest-neighbor",
            return_footprint=True
        )

        aligned_flags = np.nan_to_num(
            aligned_flags,
            nan=0
        ).astype(np.uint32)

    # Pixels outside the input image are invalid.
    invalid = (
        footprint <= 0
    ) | ~np.isfinite(aligned_science)

    aligned_science[invalid] = np.nan

    if aligned_variance is not None:
        aligned_variance[invalid] = np.nan

    # Save output FITS.
    primary = fits.PrimaryHDU()

    science_hdu = fits.ImageHDU(
        data=aligned_science.astype(np.float32),
        header=reference_wcs.to_header(),
        name="SCIENCE"
    )

    science_hdu.header["BUNIT"] = (
        observation["metadata"].get("bunit") or "UNKNOWN"
    )

    hdus = [
        primary,
        science_hdu
    ]

    if aligned_variance is not None:

        hdus.append(
            fits.ImageHDU(
                data=aligned_variance.astype(np.float32),
                name="VARIANCE"
            )
        )

    if aligned_flags is not None:

        hdus.append(
            fits.ImageHDU(
                data=aligned_flags,
                name="FLAGS"
            )
        )

    hdus.append(
        fits.ImageHDU(
            data=footprint.astype(np.float32),
            name="FOOTPRINT"
        )
    )

    hdus[0].header["ORIGIN"] = "NIGHTLOCK"
    hdus[0].header["INFILE"] = input_path.name

    fits.HDUList(hdus).writeto(
        output_path,
        overwrite=True
    )

    return {
        "input": str(input_path),
        "output": str(output_path),
        "shape": list(output_shape),
        "valid_pixel_fraction": float(
            np.mean(~invalid)
        ),
    }


# ============================================================
# ALIGN TARGET OBSERVATIONS
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target",
        required=True,
        choices=TARGET_COORDINATES.keys()
    )

    parser.add_argument(
        "--band",
        required=True
    )

    parser.add_argument(
        "--size",
        type=float,
        default=6
    )

    args = parser.parse_args()

    target = args.target
    band = args.band

    input_dir = DOWNLOAD_DIR / target / band

    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    files = sorted(
        path for path in input_dir.glob("*.fits")
        if path.is_file()
    )

    if not files:
        raise RuntimeError(
            f"No FITS files found in {input_dir}"
        )

    center = TARGET_COORDINATES[target]

    # The first observation establishes the reference grid.
    reference_path = files[0]

    reference_wcs, output_shape, pixel_scale = (
        build_reference_grid(
            reference_path,
            center,
            size_arcmin=args.size
        )
    )

    destination = OUTPUT_DIR / target / band

    destination.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Target: {target}")
    print(f"Band: {band}")
    print(f"Observations: {len(files)}")
    print(f"Output grid: {output_shape}")
    print(f"Pixel scale: {pixel_scale:.4f} arcsec/pixel")

    results = []

    for index, path in enumerate(files, start=1):

        output_path = (
            destination / f"aligned_{path.name}"
        )

        print(
            f"[{index}/{len(files)}] "
            f"Aligning {path.name}"
        )

        try:

            result = align_image(
                path,
                reference_wcs,
                output_shape,
                output_path
            )

            results.append(result)

            print(
                "Valid pixels:",
                f"{result['valid_pixel_fraction']:.3f}"
            )

        except Exception as error:

            print(
                f"FAILED {path.name}: {error}"
            )

    report_path = destination / "alignment_report.json"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print("\nALIGNMENT COMPLETE")
    print(f"Successful: {len(results)}/{len(files)}")
    print(f"Output: {destination}")


if __name__ == "__main__":
    main()