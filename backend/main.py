import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from astropy.stats import SigmaClip

from photutils.background import (
    Background2D,
    MedianBackground
)


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = (
    PROJECT_ROOT
    / "data_pipeline"
    / "downloads"
    / "qr3"
    / "2026W34_2A"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_pipeline"
    / "processed"
    / "2026W34_2A"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# INPUT FITS FILE
# --------------------------------------------------

fits_files = sorted(INPUT_DIR.glob("*.fits"))

if not fits_files:
    raise FileNotFoundError(
        f"No FITS files found in {INPUT_DIR}"
    )

FITS_PATH = fits_files[0]


# --------------------------------------------------
# FLAG DEFINITIONS
# --------------------------------------------------

def extract_flag_definitions(header):
    """Read SPHEREx bit-mask definitions from the supplied FITS header."""
    definitions = []

    for key in header:
        if key.startswith("MSKM"):
            suffix = key[4:]
            name = str(
                header.get("MSKN" + suffix, "UNKNOWN")
            ).strip()
            description = str(
                header.get("MSKD" + suffix, "")
            ).strip()

            definitions.append({
                "name": name,
                "mask_value": int(header[key]),
                "description": description
            })

    return definitions


# --------------------------------------------------
# SCIENCE IMAGE PREVIEW
# --------------------------------------------------


def create_image_preview(image, output_path):
    finite = np.isfinite(image)

    if not finite.any():
        raise ValueError("Science image contains no finite pixels.")

    valid = image[finite]
    lower = float(np.percentile(valid, 1))
    upper = float(np.percentile(valid, 99.5))

    if lower >= upper:
        raise ValueError("Image has insufficient intensity variation.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 9))

    try:
        display = ax.imshow(
            image,
            origin="lower",
            cmap="gray",
            vmin=lower,
            vmax=upper,
            interpolation="nearest"
        )

        ax.set_title("SPHEREx QR3 Level 2 Science Image")
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        colorbar = fig.colorbar(display, ax=ax)
        colorbar.set_label("Surface brightness (MJy / sr)")

        fig.tight_layout()

        # Open the destination ourselves and pass the file
        # handle to Matplotlib instead of passing the path.
        with open(output_path, "wb") as output_file:
            fig.savefig(
                output_file,
                format="png",
                dpi=180
            )

    finally:
        plt.close(fig)


def create_candidate_overlay(image, candidates, output_path):
    finite = np.isfinite(image)
    valid = image[finite]

    lower = float(np.percentile(valid, 1))
    upper = float(np.percentile(valid, 99.5))

    fig, ax = plt.subplots(figsize=(12, 10))

    try:
        ax.imshow(
            image,
            origin="lower",
            cmap="gray",
            vmin=lower,
            vmax=upper,
            interpolation="nearest"
        )

        if candidates:
            xs = [item["pixel_x"] for item in candidates]
            ys = [item["pixel_y"] for item in candidates]

            scatter = ax.scatter(
                xs,
                ys,
                s=12,
                facecolors="none",
                edgecolors="red",
                linewidths=0.6,
                alpha=0.65
            )

        ax.set_title(
            f"Project NightLock - Candidate Detection Overlay "
            f"({len(candidates)} candidates)"
        )
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")

        fig.tight_layout()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "wb") as output_file:
            fig.savefig(output_file, format="png", dpi=180)

    finally:
        plt.close(fig)

# --------------------------------------------------
# PRELIMINARY SOURCE CANDIDATE DETECTION
# --------------------------------------------------

def estimate_epsf_fwhm(epsf_table):
    """
    Estimate a representative detector-pixel FWHM from the supplied SPHEREx
    EPSF stamps. Returns (fwhm_pixels, method, number_of_stamps_used).
    """
    widths = []

    if epsf_table is None:
        return 2.5, "fallback: EPSF table unavailable", 0

    # Astropy Table exposes .colnames; a FITS_rec (from hdul["EPSF"].data)
    # exposes .names / dtype.names. Support both representations.
    column_names = getattr(epsf_table, "colnames", None)
    if column_names is None:
        column_names = getattr(epsf_table, "names", None)
    if column_names is None:
        column_names = getattr(getattr(epsf_table, "dtype", None), "names", None)
    if not column_names or "EPSF" not in column_names:
        return 2.5, "fallback: EPSF column not found in supplied table", 0

    def crossing_distance(profile, peak_index, half_level, direction):
        i = peak_index
        while 0 <= i + direction < len(profile):
            j = i + direction
            if profile[j] <= half_level:
                inner = profile[i]
                outer = profile[j]
                if inner == outer:
                    return float(abs(j - peak_index))
                fraction = (half_level - inner) / (outer - inner)
                return float(abs((i + fraction * direction) - peak_index))
            i = j
        return None

    for stamp in epsf_table["EPSF"]:
        arr = np.asarray(stamp, dtype=float)
        if arr.ndim != 2 or min(arr.shape) < 5 or not np.isfinite(arr).any():
            continue

        arr = np.where(np.isfinite(arr), arr, np.nan)
        border = np.concatenate((
            arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]
        ))
        baseline = float(np.nanmedian(border))
        profile_image = arr - baseline
        if not np.isfinite(profile_image).any():
            continue

        py, px = np.unravel_index(np.nanargmax(profile_image), profile_image.shape)
        peak = float(profile_image[py, px])
        if peak <= 0:
            continue
        half = peak / 2.0

        xprof = profile_image[py, :]
        yprof = profile_image[:, px]
        xl = crossing_distance(xprof, px, half, -1)
        xr = crossing_distance(xprof, px, half, 1)
        yl = crossing_distance(yprof, py, half, -1)
        yr = crossing_distance(yprof, py, half, 1)

        if None not in (xl, xr, yl, yr):
            fx = xl + xr
            fy = yl + yr
            f = (fx + fy) / 2.0
            if np.isfinite(f) and 0.8 <= f <= 10.0:
                widths.append(float(f))

    if len(widths) < 5:
        return 2.5, "fallback: insufficient valid EPSF stamps", len(widths)

    return float(np.median(widths)), "median FWHM measured from EPSF stamps", len(widths)


def detect_sources(
    image,
    variance,
    flags,
    wcs,
    definitions,
    epsf_table=None,
    threshold=5.0,
    max_candidates=None
):
    """
    Preliminary PSF-like detection using a spatially varying background-RMS
    map and the observation's own EPSF stamps.

    The candidate list is not capped by default. A catalogue still requires
    validation against an external/reference catalogue.
    """
    from photutils.detection import DAOStarFinder
    from astropy.stats import sigma_clipped_stats

    height, width = image.shape

    valid = (
        np.isfinite(image)
        & np.isfinite(variance)
        & (variance > 0)
    )
    if not valid.any():
        raise ValueError("No valid pixels available for detection.")

    bad_flag_names = {
        "TRANSIENT", "OVERFLOW", "SUR_ERROR", "NONFUNC",
        "DICHROIC", "MISSING_DATA", "HOT", "COLD",
        "PHANMISS", "NONLINEAR", "PERSIST", "OUTLIER",
        "CROSSTALK", "GHOST", "GHOST_FPA", "GHOST_EXT",
        "STREAK", "BLOOM", "SNOWBALL", "HALO"
    }

    bad_flag_mask = np.zeros(image.shape, dtype=bool)
    flags_int = flags.astype(np.int64, copy=False)
    for definition in definitions:
        if definition["name"].strip().upper() in bad_flag_names:
            mask_value = int(definition["mask_value"])
            bad_flag_mask |= (flags_int & mask_value) != 0

    detection_mask = ~valid | bad_flag_mask

    sigma_clip = SigmaClip(sigma=3.0, maxiters=5)
    background_method = "Photutils Background2D with local RMS map"
    try:
        background_model = Background2D(
            image,
            box_size=(64, 64),
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=MedianBackground(),
            mask=detection_mask,
            exclude_percentile=50
        )
        background_map = np.asarray(background_model.background, dtype=np.float32)
        rms_map = np.asarray(background_model.background_rms, dtype=np.float32)
    except Exception as exc:
        usable = image[~detection_mask]
        if usable.size == 0:
            raise ValueError(
                "No unmasked pixels available for background estimation."
            ) from exc
        _, global_median, global_rms = sigma_clipped_stats(
            usable, sigma=3.0, maxiters=5
        )
        if not np.isfinite(global_rms) or global_rms <= 0:
            raise ValueError("Could not estimate a positive fallback noise level.")
        background_map = np.full(image.shape, float(global_median), dtype=np.float32)
        rms_map = np.full(image.shape, float(global_rms), dtype=np.float32)
        background_method = (
            f"Global sigma-clipped fallback (Background2D failed: {type(exc).__name__})"
        )

    residual = image - background_map
    finite_rms = np.isfinite(rms_map) & (rms_map > 0) & ~detection_mask
    if not finite_rms.any():
        raise ValueError("Background RMS map contains no usable positive values.")

    rms_floor = float(np.nanpercentile(rms_map[finite_rms], 10))
    if not np.isfinite(rms_floor) or rms_floor <= 0:
        rms_floor = float(np.nanmedian(rms_map[finite_rms]))
    safe_rms_map = np.where(
        np.isfinite(rms_map) & (rms_map > 0),
        np.maximum(rms_map, rms_floor * 0.25),
        rms_floor
    ).astype(np.float32)

    # Normalize by the local noise so a single global RMS does not make
    # low-noise regions dominate detections or inflate significance.
    significance_image = (residual / safe_rms_map).astype(np.float32)
    significance_image[detection_mask] = 0.0

    fwhm_pixels, fwhm_method, epsf_stamps_used = estimate_epsf_fwhm(epsf_table)
    finder = DAOStarFinder(
        fwhm=fwhm_pixels,
        threshold=float(threshold),
        exclude_border=True
    )
    sources = finder(significance_image, mask=detection_mask)

    candidates = []
    total_detections = int(len(sources)) if sources is not None else 0
    if sources is not None and len(sources) > 0:
        sources.sort("peak")
        sources.reverse()

        source_columns = set(sources.colnames)
        x_column = "x_centroid" if "x_centroid" in source_columns else "xcentroid"
        y_column = "y_centroid" if "y_centroid" in source_columns else "ycentroid"

        for source in sources:
            x_float = float(source[x_column])
            y_float = float(source[y_column])
            x = int(round(x_float))
            y = int(round(y_float))

            if not (0 <= x < width and 0 <= y < height):
                continue
            if detection_mask[y, x]:
                continue

            cy0, cy1 = max(0, y - 1), min(height, y + 2)
            cx0, cx1 = max(0, x - 1), min(width, x + 2)
            if np.any(detection_mask[cy0:cy1, cx0:cx1]):
                continue

            ay0, ay1 = max(0, y - 2), min(height, y + 3)
            ax0, ax1 = max(0, x - 2), min(width, x + 3)
            aperture_valid = ~detection_mask[ay0:ay1, ax0:ax1]
            aperture = residual[ay0:ay1, ax0:ax1]
            aperture = np.where(
                aperture_valid & np.isfinite(aperture),
                np.maximum(aperture, 0.0),
                0.0
            )
            aperture_sum = float(np.sum(aperture))

            ra, dec = wcs.pixel_to_world_values(x_float, y_float)
            pixel_flags = int(flags_int[y, x])
            flag_names = [
                definition["name"]
                for definition in definitions
                if (pixel_flags & int(definition["mask_value"])) != 0
            ]

            local_noise = float(safe_rms_map[y, x])
            peak_excess = float(source["peak"])
            candidates.append({
                "candidate_id": f"NL-{len(candidates) + 1:05d}",
                "pixel_x": x,
                "pixel_y": y,
                "centroid_x": round(x_float, 3),
                "centroid_y": round(y_float, 3),
                "ra_degrees": float(ra),
                "dec_degrees": float(dec),
                "snr": round(peak_excess, 3),
                "local_noise_mjy_sr": local_noise,
                "peak_intensity_mjy_sr": float(image[y, x]),
                "aperture_sum_mjy_sr_pixel": aperture_sum,
                "central_pixel_flags": pixel_flags,
                "central_pixel_flag_names": flag_names,
                "has_source_mapping_flag": bool(pixel_flags & 2097152),
                "sharpness": float(source["sharpness"]),
                "roundness1": float(source["roundness1"]),
                "detection_method": "DAOStarFinder on locally noise-normalized residual"
            })

    total_accepted_before_limit = len(candidates)
    if max_candidates is not None and len(candidates) > int(max_candidates):
        candidates = candidates[:int(max_candidates)]

    detection_info = {
        "method": (
            "Photutils DAOStarFinder on Background2D-subtracted image "
            "normalized by spatially varying background RMS"
        ),
        "snr_threshold": float(threshold),
        "noise_estimator": "spatially varying Background2D.background_rms",
        "background_method": background_method,
        "background_box_size": [64, 64],
        "psf_fwhm_pixels": fwhm_pixels,
        "psf_fwhm_method": fwhm_method,
        "epsf_stamps_used": epsf_stamps_used,
        "candidate_count": len(candidates),
        "total_detections_before_quality_checks": total_detections,
        "total_accepted_before_limit": total_accepted_before_limit,
        "maximum_candidates_saved": max_candidates,
        "bad_quality_pixels": int(np.count_nonzero(bad_flag_mask)),
        "valid_unmasked_pixels": int(np.count_nonzero(valid & ~bad_flag_mask)),
        "variance_plane_used_for_detection_snr": False,
        "variance_plane_note": (
            "Variance plane is used to exclude invalid/non-positive pixels; "
            "local image RMS is used for detection significance pending "
            "verification of variance calibration and units."
        ),
        "scientific_status": (
            "Preliminary candidates. Validate completeness and reliability "
            "against SPHEREx EPSF metadata and an external/reference catalogue."
        )
    }
    return candidates, detection_info


# --------------------------------------------------
# MAIN OBSERVATION PROCESSOR
# --------------------------------------------------

def process_observation():
    print("=" * 60)
    print("PROJECT NIGHTLOCK - SPHEREx PROCESSOR")
    print("=" * 60)

    print("\nInput:", FITS_PATH.name)
    print("Output:", OUTPUT_DIR)

    with fits.open(
        FITS_PATH,
        memmap=True
    ) as hdul:
        image = np.array(
            hdul["IMAGE"].data,
            dtype=np.float32,
            copy=True
        )

        variance = np.array(
            hdul["VARIANCE"].data,
            dtype=np.float32,
            copy=True
        )

        flags = np.array(
            hdul["FLAGS"].data,
            copy=True
        )

        if not (
            image.shape
            == variance.shape
            == flags.shape
        ):
            raise ValueError(
                "Image, variance and flags have different shapes."
            )

        # Image statistics.
        finite_image = np.isfinite(image)
        valid_image = image[finite_image]

        statistics = {
            "minimum": float(np.min(valid_image)),
            "maximum": float(np.max(valid_image)),
            "mean": float(np.mean(valid_image)),
            "median": float(np.median(valid_image)),
            "standard_deviation": float(
                np.std(valid_image)
            ),
            "finite_pixels": int(
                np.count_nonzero(finite_image)
            ),
            "non_finite_pixels": int(
                np.count_nonzero(~finite_image)
            )
        }

        # Variance statistics.
        finite_variance = np.isfinite(variance)

        if not finite_variance.any():
            raise ValueError(
                "Variance image contains no finite pixels."
            )

        variance_values = variance[finite_variance]

        variance_statistics = {
            "minimum": float(
                np.min(variance_values)
            ),
            "maximum": float(
                np.max(variance_values)
            ),
            "median": float(
                np.median(variance_values)
            ),
            "non_finite_pixels": int(
                np.count_nonzero(~finite_variance)
            )
        }

        # Celestial coordinate system.
        wcs = WCS(
            hdul["IMAGE"].header
        ).celestial

        height, width = image.shape

        center_x = (width - 1) / 2
        center_y = (height - 1) / 2

        center_ra, center_dec = (
            wcs.pixel_to_world_values(
                center_x,
                center_y
            )
        )

        # Decode flag definitions.
        # SPHEREx QR3 bit definitions are stored in the primary
        # header for this product, not the FLAGS extension header.
        definitions = extract_flag_definitions(hdul[0].header)
        if not definitions:
            definitions = extract_flag_definitions(hdul["FLAGS"].header)
        if not definitions:
            definitions = extract_flag_definitions(hdul["IMAGE"].header)
        if not definitions:
            raise ValueError("No SPHEREx MSKM flag definitions found in FITS headers.")


        flag_results = []
        total_pixels = flags.size

        for definition in definitions:
            mask_value = definition["mask_value"]

            affected = (
                np.bitwise_and(
                    flags,
                    mask_value
                ) != 0
            )

            count = int(
                np.count_nonzero(affected)
            )

            flag_results.append({
                "name": definition["name"],
                "mask_value": mask_value,
                "description": definition["description"],
                "pixel_count": count,
                "percentage": round(
                    count / total_pixels * 100,
                    6
                )
            })

        # Run preliminary source-candidate detection.
        candidate_sources, detection_info = detect_sources(
            image=image,
            variance=variance,
            flags=flags,
            wcs=wcs,
            definitions=definitions,
            epsf_table=hdul["EPSF"].data if "EPSF" in hdul else None,
            threshold=5.0,
            max_candidates=None
        )

        observation = {
            "project": "Project NightLock",
            "source_file": FITS_PATH.name,
            "pipeline_version": hdul[0].header.get(
                "VERSION",
                "Unknown"
            ),
            "observation_date": hdul["IMAGE"].header.get(
                "DATE-OBS",
                None
            ),
            "mjd_observation": hdul["IMAGE"].header.get(
                "MJD-OBS",
                None
            ),
            "image_dimensions": {
                "width": width,
                "height": height
            },
            "image_unit": hdul["IMAGE"].header.get(
                "BUNIT",
                None
            ),
            "coordinate_system": hdul["IMAGE"].header.get(
                "RADESYS",
                None
            ),
            "image_center": {
                "ra_degrees": float(center_ra),
                "dec_degrees": float(center_dec)
            },
            "image_statistics": statistics,
            "variance_unit": hdul["VARIANCE"].header.get(
                "BUNIT",
                None
            ),
            "variance_statistics": variance_statistics,
            "candidate_detection": detection_info
        }

    # --------------------------------------------------
    # EXPORT SCIENCE IMAGE PREVIEW
    # --------------------------------------------------

    preview_path = (
        OUTPUT_DIR / "science_preview.png"
    )

    create_image_preview(
        image,
        preview_path
    )

    overlay_path = (
        OUTPUT_DIR / "candidate_overlay.png"
    )

    create_candidate_overlay(
        image,
        candidate_sources,
        overlay_path
    )

    

    # --------------------------------------------------
    # EXPORT OBSERVATION METADATA
    # --------------------------------------------------

    metadata_path = (
        OUTPUT_DIR / "observation.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            observation,
            file,
            indent=4
        )

    # --------------------------------------------------
    # EXPORT FLAG SUMMARY CSV
    # --------------------------------------------------

    csv_path = (
        OUTPUT_DIR / "flag_summary.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "name",
                "mask_value",
                "description",
                "pixel_count",
                "percentage"
            ]
        )

        writer.writeheader()
        writer.writerows(flag_results)

    # --------------------------------------------------
    # EXPORT CANDIDATE JSON
    # --------------------------------------------------

    candidate_json_path = (
        OUTPUT_DIR / "candidate_sources.json"
    )

    with open(
        candidate_json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "observation": FITS_PATH.name,
                "detection": detection_info,
                "candidates": candidate_sources
            },
            file,
            indent=4
        )

    # --------------------------------------------------
    # EXPORT CANDIDATE CSV
    # --------------------------------------------------

    candidate_csv_path = (
        OUTPUT_DIR / "candidate_sources.csv"
    )

    candidate_fields = [
        "candidate_id",
        "pixel_x",
        "pixel_y",
        "centroid_x",
        "centroid_y",
        "ra_degrees",
        "dec_degrees",
        "snr",
        "local_noise_mjy_sr",
        "peak_intensity_mjy_sr",
        "aperture_sum_mjy_sr_pixel",
        "central_pixel_flags",
        "central_pixel_flag_names",
        "has_source_mapping_flag",
        "sharpness",
        "roundness1",
        "detection_method"
    ]

    with open(
        candidate_csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=candidate_fields
        )

        writer.writeheader()

        for candidate in candidate_sources:
            row = candidate.copy()

            row["central_pixel_flag_names"] = ";".join(
                row["central_pixel_flag_names"]
            )

            writer.writerow(row)

    # --------------------------------------------------
    # PROCESSING REPORT
    # --------------------------------------------------

    report = {
        "status": "success",
        "processed_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "input_file": FITS_PATH.name,
        "outputs": [
            "science_preview.png",
            "observation.json",
            "flag_summary.csv",
            "candidate_sources.json",
            "candidate_overlay.png",
            "candidate_sources.csv"
        ],
        "quality_handling": {
            "flags_decoded": bool(definitions),
            "bad_quality_pixels_excluded_from_detection": (
                detection_info["bad_quality_pixels"]
            ),
            "exclusion_policy": (
                "Excluded pixels carrying selected detector, data-integrity, "
                "persistence, outlier, crosstalk, ghost, streak, bloom, "
                "snowball, or halo flags. SOURCE mapping flag is retained."
            ),
            "source_mapping_flag_excluded": False,
            "original_fits_preserved": True
        }
    }

    report_path = (
        OUTPUT_DIR / "processing_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------------------
    # FINAL TERMINAL OUTPUT
    # --------------------------------------------------

    print("\nPROCESSING SUCCESSFUL")

    print("\nImage statistics:")
    for key, value in statistics.items():
        print(f"{key}: {value}")

    print("\nSky coordinates:")
    print("RA:", center_ra)
    print("Dec:", center_dec)

    print("\nDecoded flags:", len(flag_results))

    print("\nGenerated files:")
    for path in [
        preview_path,
        metadata_path,
        csv_path,
        candidate_json_path,
        candidate_csv_path,
        overlay_path,
        report_path
    ]:
        print(path)

    print(
        "\nCandidate sources detected:",
        len(candidate_sources)
    )
    print(
        "Detection threshold:",
        detection_info["snr_threshold"],
        "SNR"
    )
    print("Candidate JSON:", candidate_json_path)
    print("Candidate CSV:", candidate_csv_path)

    print("\nNightLock processing complete.")


if __name__ == "__main__":
    process_observation()
