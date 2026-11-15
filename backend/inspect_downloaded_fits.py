
from pathlib import Path
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FITS_DIR = (
    PROJECT_ROOT
    / "data_pipeline"
    / "downloads"
    / "qr3"
    / "2026W34_2A"
)

fits_files = list(FITS_DIR.glob("*.fits"))

if not fits_files:
    raise FileNotFoundError("No FITS files found.")

fits_path = fits_files[0]

print("Inspecting:", fits_path.name)

with fits.open(fits_path, checksum=True) as hdul:

    print("\nFITS VALIDATION")
    hdul.verify("exception")

    print("\nHEADER AND HDU INFORMATION")

    for index, hdu in enumerate(hdul):

        print("\n" + "=" * 45)
        print("HDU:", index)
        print("Name:", hdu.name)
        print("Type:", type(hdu).__name__)
        print("Shape:", getattr(hdu.data, "shape", None))

        if hdu.data is not None:
            print("Data type:", hdu.data.dtype)

        if index == 0:
            print("\nPRIMARY HEADER")

            for key in [
                "OBJECT",
                "DATE-OBS",
                "MJD-OBS",
                "EXPTIME",
                "TELESCOP",
                "INSTRUME",
                "BUNIT",
                "FILTER"
            ]:
                print(key, ":", hdu.header.get(key, "Not present"))

print("\nFITS inspection complete.")
