from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import csv
import time

import numpy as np
import pyvo
import requests

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.io import fits


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

OUTPUT = ROOT / "data_pipeline" / "downloads" / "challenge_targets"
OUTPUT.mkdir(parents=True, exist_ok=True)

TAP_URL = "https://irsa.ipac.caltech.edu/TAP"

CUTOUT_SIZE = 0.15  # degrees; 9 arcminutes

BANDPASSES = [
    f"SPHEREx-D{i}" for i in range(1, 7)
]

MAX_WORKERS = 5

# SIMBAD J2000 coordinates.
# Proper motions are in mas/year:
# pm_ra_cosdec, pm_dec

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
        # Approximate comoving motion of Epsilon Indi A.
        "pmra": 3966.661,
        "pmdec": -2536.192,
        "epoch": 2000.0,
    },

    "wise_0855": {
        "ra": 133.795129,
        "dec": -7.245147,
        # Supplied challenge coordinate.
        # Proper motion must be checked against its reference epoch.
        "pmra": 0.0,
        "pmdec": 0.0,
        "epoch": 2000.0,
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
# ASTROMETRY
# ============================================================

def target_position(target):
    """
    Propagate the catalogue coordinate to a common
    reference epoch for the initial archive search.
    """

    position = SkyCoord(
        ra=target["ra"] * u.deg,
        dec=target["dec"] * u.deg,
        pm_ra_cosdec=target["pmra"] * u.mas / u.yr,
        pm_dec=target["pmdec"] * u.mas / u.yr,
        obstime=Time(target["epoch"], format="jyear"),
        frame="icrs",
    )

    propagated = position.apply_space_motion(
        new_obstime=Time("2026-01-01")
    )

    return propagated


# ============================================================
# QUERY IRSA TAP
# ============================================================

service = pyvo.dal.TAPService(TAP_URL)


def query_target(target_name, target):
    position = target_position(target)

    ra = float(position.ra.deg)
    dec = float(position.dec.deg)

    print(
        f"\nSearching {target_name}: "
        f"RA={ra:.7f}, Dec={dec:.7f}"
    )

    all_rows = []

    for bandpass in BANDPASSES:

        query = f"""
        SELECT
            'https://irsa.ipac.caltech.edu/' ||
            a.uri ||
            '?center={ra:.8f},{dec:.8f}d&size={CUTOUT_SIZE}'
            AS cutout_url,

            p.time_bounds_lower AS observation_time,
            p.energy_bandpassname AS bandpass,
            p.planeid AS plane_id

        FROM spherex.artifact a

        JOIN spherex.plane p
        ON a.planeid = p.planeid

        WHERE 1 = CONTAINS(
            POINT('ICRS', {ra:.8f}, {dec:.8f}),
            p.poly
        )

        AND p.energy_bandpassname = '{bandpass}'

        ORDER BY p.time_bounds_lower
        """

        try:
            result = service.search(query)

            print(
                f"  {bandpass}: {len(result)} observations"
            )

            for row in result:
                all_rows.append({
                    "target": target_name,
                    "url": str(row["cutout_url"]),
                    "observation_time": str(
                        row["observation_time"]
                    ),
                    "bandpass": str(row["bandpass"]),
                    "plane_id": str(row["plane_id"]),
                })

        except Exception as error:
            print(
                f"  QUERY ERROR {bandpass}: {error}"
            )

    return all_rows


# ============================================================
# DOWNLOAD AND VALIDATE FITS
# ============================================================

def download_one(row):

    target_folder = OUTPUT / row["target"] / row["bandpass"]

    target_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    plane_id = row["plane_id"]

    filename = f"{plane_id}.fits"

    destination = target_folder / filename

    if destination.exists():
        try:
            with fits.open(destination) as hdul:
                if hdul[1].data is not None:
                    return {
                        **row,
                        "status": "already_downloaded",
                        "local_file": str(destination),
                    }
        except Exception:
            destination.unlink(missing_ok=True)

    for attempt in range(3):

        try:
            response = requests.get(
                row["url"],
                timeout=(30, 240),
                stream=True
            )

            response.raise_for_status()

            temporary = destination.with_suffix(".part")

            with open(temporary, "wb") as output_file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        output_file.write(chunk)

            # Validate FITS structure before accepting.
            with fits.open(temporary) as hdul:
                hdul.verify("exception")

                image_hdu = next(
                    hdu for hdu in hdul
                    if hdu.data is not None
                    and getattr(hdu.data, "ndim", 0) >= 2
                )

                if image_hdu.data.size == 0:
                    raise ValueError("Empty FITS image")

            temporary.replace(destination)

            return {
                **row,
                "status": "downloaded",
                "local_file": str(destination),
            }

        except Exception as error:

            print(
                f"DOWNLOAD ERROR {filename}, "
                f"attempt {attempt + 1}: {error}"
            )

            if "temporary" in locals():
                temporary.unlink(missing_ok=True)

            time.sleep(2 ** attempt)

    return {
        **row,
        "status": "failed",
        "local_file": "",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    all_observations = []

    for name, target in TARGETS.items():

        observations = query_target(name, target)

        print(
            f"\n{name}: "
            f"{len(observations)} total observations"
        )

        all_observations.extend(observations)

    if not all_observations:
        raise RuntimeError(
            "IRSA returned no observations. "
            "Check TAP access and target coordinates."
        )

    # Save archive results before downloading.
    manifest_path = OUTPUT / "observation_manifest.csv"

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target",
                "url",
                "observation_time",
                "bandpass",
                "plane_id",
            ]
        )

        writer.writeheader()
        writer.writerows(all_observations)

    print(
        f"\nTotal observations found: "
        f"{len(all_observations)}"
    )

    print("\nStarting FITS downloads...")

    results = []

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = [
            executor.submit(download_one, row)
            for row in all_observations
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            print(
                result["target"],
                result["bandpass"],
                result["status"],
            )

    # Save download results.
    result_path = OUTPUT / "download_results.csv"

    with open(
        result_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "target",
                "url",
                "observation_time",
                "bandpass",
                "plane_id",
                "status",
                "local_file",
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    successful = sum(
        item["status"] in (
            "downloaded",
            "already_downloaded"
        )
        for item in results
    )

    failed = len(results) - successful

    print("\n================================")
    print("NIGHTLOCK DOWNLOAD SUMMARY")
    print("================================")
    print(f"Observations found: {len(all_observations)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()