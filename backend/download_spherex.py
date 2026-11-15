
import requests
from pathlib import Path

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BUCKET_URL = (
    "https://nasa-irsa-spherex.s3.us-east-1.amazonaws.com/"
)

S3_KEY = (
    "qr3/level2/2026W34_2A/"
    "l2b-v27-2026-243/1/"
    "level2_2026W34_2A_0004_1D1_"
    "spx_l2b-v27-2026-243.fits"
)

# Repository root = parent of backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOWNLOAD_DIR = (
    PROJECT_ROOT
    / "data_pipeline"
    / "downloads"
    / "qr3"
    / "2026W34_2A"
)

# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

def download_fits():

    DOWNLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = S3_KEY.split("/")[-1]

    destination = DOWNLOAD_DIR / filename
    temporary_file = DOWNLOAD_DIR / (filename + ".part")

    if destination.exists():
        print("File already exists:", destination)
        print("Size:", round(destination.stat().st_size / 1024**2, 2), "MB")
        return destination

    url = BUCKET_URL + S3_KEY

    print("Connecting to SPHEREx AWS...")
    print("Source:", url)
    print("Destination:", destination)

    downloaded = 0

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(15, 120)
        ) as response:

            response.raise_for_status()

            expected_size = int(
                response.headers.get("Content-Length", 0)
            )

            print(
                "Expected size:",
                round(expected_size / 1024**2, 2),
                "MB"
            )

            with open(temporary_file, "wb") as file:

                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):

                    if chunk:
                        file.write(chunk)
                        downloaded += len(chunk)

                        print(
                            f"\rDownloaded: "
                            f"{downloaded / 1024**2:.2f} MB",
                            end=""
                        )

        print()

        if expected_size and downloaded != expected_size:
            raise ValueError(
                f"Size mismatch! Expected {expected_size}, "
                f"received {downloaded}"
            )

        if downloaded == 0:
            raise ValueError("Downloaded file is empty.")

        temporary_file.replace(destination)

        print("\nDOWNLOAD SUCCESSFUL")
        print("Saved to:", destination)
        print(
            "Final size:",
            round(destination.stat().st_size / 1024**2, 2),
            "MB"
        )

        return destination

    except Exception as error:

        if temporary_file.exists():
            temporary_file.unlink()

        print("\nDOWNLOAD FAILED")
        print("Reason:", error)

        raise


if __name__ == "__main__":
    download_fits()
