from pathlib import Path
import csv
import json
import math

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT / "data_pipeline"
    / "downloads"
    / "challenge_targets"
)

OUTPUT_DIR = (
    ROOT / "data_pipeline"
    / "processed"
    / "challenge_dataset"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = OUTPUT_DIR / "dataset_inventory.csv"
JSON_OUTPUT = OUTPUT_DIR / "dataset_inventory.json"
REPORT_OUTPUT = OUTPUT_DIR / "dataset_quality_report.json"


# ============================================================
# HEADER HELPERS
# ============================================================

def first_header_value(headers, keys):
    for header in headers:
        for key in keys:
            value = header.get(key)
            if value is not None:
                return str(value)
    return None


def get_image_hdu(hdul):
    for index, hdu in enumerate(hdul):
        data = hdu.data

        if (
            data is not None
            and isinstance(data, np.ndarray)
            and data.ndim >= 2
        ):
            return index, hdu

    return None, None


# ============================================================
# INSPECT ONE FITS FILE
# ============================================================

def inspect_fits(path):

    relative = path.relative_to(DATA_DIR)
    parts = relative.parts

    target = parts[0] if len(parts) > 0 else "unknown"
    folder_band = parts[1] if len(parts) > 1 else "unknown"

    result = {
        "file": path.name,
        "path": str(relative),
        "target": target,
        "folder_band": folder_band,
        "file_size_mb": round(path.stat().st_size / (1024 ** 2), 3),
        "status": "unknown",
        "error": "",
    }

    try:
        with fits.open(
            path,
            memmap=True,
            lazy_load_hdus=True,
        ) as hdul:

            headers = [hdu.header for hdu in hdul]

            image_index, image_hdu = get_image_hdu(hdul)

            result["hdu_count"] = len(hdul)

            result["observation_id"] = first_header_value(
                headers,
                ["OBS_ID", "OBSID", "PLANEID", "FILEID"]
            )

            result["observation_date"] = first_header_value(
                headers,
                [
                    "DATE-OBS",
                    "DATE_OBS",
                    "MJD-OBS",
                    "MJD_OBS",
                    "MJD-BEG",
                ]
            )

            result["band"] = first_header_value(
                headers,
                [
                    "BAND",
                    "BANDPASS",
                    "FILTER",
                    "WAVEBAND",
                ]
            ) or folder_band

            if image_hdu is None:
                result["status"] = "INVALID_NO_IMAGE"
                result["error"] = "No 2D image HDU found"
                return result

            data = image_hdu.data

            result["image_hdu"] = image_index
            result["image_shape"] = list(data.shape)
            result["image_dtype"] = str(data.dtype)

            # Inspect a subsample to avoid unnecessarily
            # loading large images into memory.
            if data.ndim > 2:
                image = data.reshape(-1, *data.shape[-2:])[0]
            else:
                image = data

            step_y = max(1, image.shape[0] // 256)
            step_x = max(1, image.shape[1] // 256)

            sample = np.asarray(
                image[::step_y, ::step_x],
                dtype=np.float64,
            )

            finite = np.isfinite(sample)

            result["finite_sample_fraction"] = round(
                float(finite.mean()), 5
            )

            if finite.any():
                valid = sample[finite]

                result["sample_min"] = float(np.min(valid))
                result["sample_max"] = float(np.max(valid))
                result["sample_median"] = float(
                    np.median(valid)
                )

            # Check for a usable celestial WCS.
            try:
                wcs = WCS(image_hdu.header)

                if wcs.has_celestial:
                    celestial = wcs.celestial

                    result["has_celestial_wcs"] = True
                    result["wcs_ctype"] = list(
                        celestial.wcs.ctype
                    )

                    ny, nx = image.shape[-2:]

                    corners = celestial.pixel_to_world(
                        [0, nx - 1, 0, nx - 1],
                        [0, 0, ny - 1, ny - 1],
                    )

                    result["footprint_ra"] = [
                        float(v) for v in corners.ra.deg
                    ]

                    result["footprint_dec"] = [
                        float(v) for v in corners.dec.deg
                    ]

                else:
                    result["has_celestial_wcs"] = False

            except Exception as error:
                result["has_celestial_wcs"] = False
                result["wcs_error"] = str(error)

            # Record available quality-related HDUs.
            result["available_hdus"] = [
                hdu.name for hdu in hdul
            ]

            result["has_flags"] = any(
                hdu.name.upper() in ("FLAGS", "DQ", "MASK")
                for hdu in hdul
            )

            result["has_variance"] = any(
                hdu.name.upper() in (
                    "VARIANCE",
                    "VAR",
                    "ERROR",
                )
                for hdu in hdul
            )

            result["status"] = (
                "VALID"
                if result.get("has_celestial_wcs")
                else "VALID_WCS_MISSING"
            )

    except Exception as error:
        result["status"] = "INVALID"
        result["error"] = str(error)

    return result


# ============================================================
# MAIN DATASET INSPECTION
# ============================================================

def main():

    print("=" * 60)
    print("NIGHTLOCK: SPHEREx DATASET INSPECTOR")
    print("=" * 60)

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    fits_files = sorted(
        path for path in DATA_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in (".fits", ".fit", ".fts")
    )

    print(f"\nFITS files found: {len(fits_files)}")

    if not fits_files:
        print("No completed FITS files found yet.")
        return

    inventory = []

    for number, path in enumerate(fits_files, start=1):

        print(
            f"[{number}/{len(fits_files)}] "
            f"Inspecting {path.name}"
        )

        result = inspect_fits(path)
        inventory.append(result)

    # Save CSV inventory.
    columns = sorted({
        key
        for row in inventory
        for key in row
    })

    with open(
        CSV_OUTPUT,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=columns,
        )

        writer.writeheader()

        for row in inventory:
            writer.writerow({
                key: json.dumps(value)
                if isinstance(value, (list, dict))
                else value
                for key, value in row.items()
            })

    # Save JSON inventory.
    with open(
        JSON_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            inventory,
            file,
            indent=2,
        )

    # Build quality report.
    statuses = {}

    for row in inventory:
        status = row["status"]
        statuses[status] = statuses.get(status, 0) + 1

    targets = {}

    for row in inventory:
        target = row["target"]

        if target not in targets:
            targets[target] = {
                "total": 0,
                "bands": {},
                "statuses": {},
            }

        item = targets[target]
        item["total"] += 1

        band = row["band"]
        item["bands"][band] = (
            item["bands"].get(band, 0) + 1
        )

        status = row["status"]
        item["statuses"][status] = (
            item["statuses"].get(status, 0) + 1
        )

    report = {
        "total_files": len(inventory),
        "status_counts": statuses,
        "targets": targets,
        "files_without_wcs": [
            row["path"]
            for row in inventory
            if not row.get("has_celestial_wcs", False)
        ],
        "invalid_files": [
            row["path"]
            for row in inventory
            if row["status"].startswith("INVALID")
        ],
    }

    with open(
        REPORT_OUTPUT,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(report, file, indent=2)

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 60)

    print(f"Total FITS files: {len(inventory)}")

    for status, count in statuses.items():
        print(f"{status}: {count}")

    print("\nTARGET SUMMARY")

    for target, info in targets.items():
        print(f"\n{target}: {info['total']} files")

        for band, count in info["bands"].items():
            print(f"  {band}: {count}")

    print("\nOUTPUT FILES")
    print(CSV_OUTPUT)
    print(JSON_OUTPUT)
    print(REPORT_OUTPUT)


if __name__ == "__main__":
    main()
