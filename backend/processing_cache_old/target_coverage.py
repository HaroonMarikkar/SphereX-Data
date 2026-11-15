from pathlib import Path
import csv
import json

import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits

from .fits_loader import load_fits


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    ROOT / "data_pipeline"
    / "downloads"
    / "challenge_targets"
)

RESULTS_CSV = DATA_DIR / "download_results.csv"

OUTPUT_DIR = (
    ROOT / "data_pipeline"
    / "processed"
    / "challenge_dataset"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "target_coverage.csv"
OUTPUT_JSON = OUTPUT_DIR / "target_coverage.json"


# J2000 coordinates and approximate proper motions.
# pmra is mu_alpha*cos(dec), in mas/year.
#
# These are working catalogue values, not certified
# target ephemerides. Verify against SIMBAD/Gaia before
# making scientific claims.

TARGETS = {
    "barnards_star": {
        "ra": 269.452077,
        "dec": 4.693365,
        "pmra": -801.551,
        "pmdec": 10362.394,
        "epoch": 2000.0,
    },

    "epsilon_indi_b": {
        "ra": 331.044146,
        "dec": -56.782844,
        "pmra": 3966.661,
        "pmdec": -2536.192,
        "epoch": 2000.0,
        "note": "Approximate motion adopted from the primary.",
    },

    "wise_0855": {
        "ra": 133.795129,
        "dec": -7.245147,
        "pmra": 0.0,
        "pmdec": 0.0,
        "epoch": 2000.0,
        "note": "Proper motion and coordinate epoch require verification.",
    },

    "rw_aurigae": {
        "ra": 76.956526,
        "dec": 30.401438,
        "pmra": 7.23,
        "pmdec": -20.33,
        "epoch": 2000.0,
    },

    "r_draconis": {
        "ra": 248.167611,
        "dec": 66.754979,
        "pmra": -20.643,
        "pmdec": 6.149,
        "epoch": 2000.0,
    },
}


# ============================================================
# PROPAGATE TARGET POSITION
# ============================================================

def get_target_position(name, observation_date):

    target = TARGETS[name]

    position = SkyCoord(
        ra=target["ra"] * u.deg,
        dec=target["dec"] * u.deg,
        pm_ra_cosdec=target["pmra"] * u.mas / u.yr,
        pm_dec=target["pmdec"] * u.mas / u.yr,
        obstime=Time(target["epoch"], format="jyear"),
        frame="icrs",
    )

    epoch = Time(observation_date)

    propagated = position.apply_space_motion(
        new_obstime=epoch
    )

    return propagated


# ============================================================
# CHECK TARGET COVERAGE
# ============================================================

def check_coverage(path, target_name, observation_date):

    observation = load_fits(path)

    image = observation["science"]
    wcs = observation["wcs"]

    target_position = get_target_position(
        target_name,
        observation_date
    )

    x, y = wcs.world_to_pixel(target_position)

    height, width = image.shape

    inside = (
        np.isfinite(x)
        and np.isfinite(y)
        and 0 <= x < width
        and 0 <= y < height
    )

    # Calculate distance to nearest image edge.
    edge_distance = None

    if inside:

        edge_distance = min(
            x,
            y,
            width - 1 - x,
            height - 1 - y
        )

    return {
        "target": target_name,
        "file": str(path),
        "observation_date": str(observation_date),
        "target_ra": float(target_position.ra.deg),
        "target_dec": float(target_position.dec.deg),
        "pixel_x": float(x),
        "pixel_y": float(y),
        "inside_image": bool(inside),
        "edge_distance_pixels": (
            float(edge_distance)
            if edge_distance is not None
            else None
        ),
        "target_astrometry_note": TARGETS[target_name].get(
            "note", ""
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Missing download manifest: {RESULTS_CSV}"
        )

    df = pd.read_csv(RESULTS_CSV)

    # Only process completed downloads.
    df = df[
        df["status"].isin([
            "downloaded",
            "already_downloaded"
        ])
    ]

    results = []

    for _, row in df.iterrows():

        target = row["target"]

        if target not in TARGETS:
            continue

        path = Path(row["local_file"])

        if not path.is_absolute():
            path = ROOT / path

        if not path.exists() or path.suffix.lower() == ".part":
            continue

        try:

            result = check_coverage(
                path,
                target,
                row["observation_time"]
            )

            result["bandpass"] = row["bandpass"]
            result["plane_id"] = row["plane_id"]

            results.append(result)

            print(
                f"{target}: "
                f"{result['inside_image']} "
                f"({path.name})"
            )

        except Exception as error:

            print(
                f"ERROR {path.name}: {error}"
            )

            results.append({
                "target": target,
                "file": str(path),
                "inside_image": False,
                "error": str(error),
            })

    if not results:
        print("No completed FITS observations to inspect.")
        return

    output = pd.DataFrame(results)

    output.to_csv(
        OUTPUT_CSV,
        index=False
    )

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2
        )

    print("\nTARGET COVERAGE SUMMARY")

    for target, group in output.groupby("target"):

        successful = (
            group["inside_image"] == True
        ).sum()

        print(
            f"{target}: "
            f"{successful}/{len(group)} observations cover target"
        )

    print("\nSaved:", OUTPUT_CSV)
    print("Saved:", OUTPUT_JSON)


if __name__ == "__main__":
    main()