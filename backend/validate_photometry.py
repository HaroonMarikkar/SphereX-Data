"""NightLock: diagnose Barnard's Star aperture photometry on aligned FITS files.

Run from backend after replacing analysis.py with analysis_fixed.py:
    python validate_photometry.py --target barnards_star --band SPHEREx-D5

Writes CSV, JSON, and diagnostic PNGs under data_pipeline/processed/validation/.
It does not alter FITS files or existing analysis outputs.
"""
from __future__ import annotations
import argparse, csv, json, logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import processing
import analysis


def main():
    parser = argparse.ArgumentParser(description="Diagnose target aperture photometry")
    parser.add_argument("--target", default="barnards_star", choices=sorted(processing.TARGETS))
    parser.add_argument("--band", default="SPHEREx-D5")
    parser.add_argument("--samples", type=int, default=3, help="Number of diagnostic PNGs: earliest, middle, latest")
    parser.add_argument("--aperture", type=float, default=analysis.DEFAULT_APERTURE_PIX)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    band_dir = processing.ALIGNED_DIR / args.target / args.band
    files = sorted(p for p in band_dir.glob("*.fits") if p.is_file())
    if not files:
        raise SystemExit(f"No aligned FITS files found in {band_dir}")

    manifest = processing.read_manifest()
    rows, loaded_items = [], []
    for path in files:
        try:
            data = analysis.read_aligned(path)
            meta = analysis.observation_metadata(path, data, manifest)
            epoch = meta["epoch"]
            x, y = analysis.target_pixel(args.target, epoch, data["wcs"])
            phot = analysis.aperture_photometry(
                data["science"], x, y, args.aperture, data["bad"], data["variance"]
            )
            sci = data["science"]
            finite = np.isfinite(sci)
            invalid_fraction = float(1.0 - finite.mean())
            nearest = None
            try:
                sources = analysis.detect_sources(sci, data["bad"], threshold_sigma=4.0)
                if sources:
                    candidate = min(sources, key=lambda s: (s["x"]-x)**2 + (s["y"]-y)**2)
                    distance = float(np.hypot(candidate["x"]-x, candidate["y"]-y))
                    nearest = {"x": candidate["x"], "y": candidate["y"],
                               "snr_peak": candidate["snr_peak"], "distance_from_predicted_px": distance}
            except Exception:
                pass
            row = {
                "file": path.name,
                "observation_time": epoch.isot if epoch else None,
                "mjd": float(epoch.mjd) if epoch else None,
                "image_width": int(sci.shape[1]), "image_height": int(sci.shape[0]),
                "predicted_x": x, "predicted_y": y,
                "finite_pixel_fraction": float(finite.mean()),
                "invalid_pixel_fraction": invalid_fraction,
                "bad_mask_fraction": float(np.asarray(data["bad"], dtype=bool).mean()),
                "aperture_valid_fraction": phot["aperture_valid_fraction"],
                "background_pixels": phot["n_background_pixels"],
                "background": phot["background"],
                "background_sigma": phot["background_sigma"],
                "flux_after_nan_safe_sum": phot["flux"],
                "flux_error": phot["flux_error"],
                "photometry_quality": phot["quality"],
                "nearest_detected_source": json.dumps(nearest) if nearest else None,
            }
            rows.append(row)
            loaded_items.append((path, data, epoch, x, y, phot, nearest))
        except Exception as exc:
            rows.append({"file": path.name, "error": str(exc)})
            logging.exception("Could not diagnose %s", path.name)

    outdir = processing.PROCESSED_DIR / "validation" / args.target / args.band
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "photometry_diagnostic.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (outdir / "photometry_diagnostic.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if loaded_items:
        count = min(max(args.samples, 1), len(loaded_items))
        indices = np.unique(np.linspace(0, len(loaded_items)-1, count).round().astype(int))
        for rank, idx in enumerate(indices, 1):
            path, data, epoch, x, y, phot, nearest = loaded_items[int(idx)]
            image = np.asarray(data["science"], dtype=float)
            valid = np.isfinite(image) & ~np.asarray(data["bad"], dtype=bool)
            vals = image[valid]
            if vals.size:
                lo, hi = np.percentile(vals, [2, 99.5])
                if not np.isfinite(hi) or hi <= lo: hi = lo + 1.0
                display = np.arcsinh((image - lo) / max((hi-lo)/8.0, 1e-9))
            else:
                display = image
            fig, ax = plt.subplots(figsize=(8, 7))
            ax.imshow(display, origin="lower", cmap="gray", interpolation="nearest")
            ax.add_patch(Circle((x, y), args.aperture, fill=False, linewidth=1.5, label="Photometry aperture"))
            ax.add_patch(Circle((x, y), args.aperture+2, fill=False, linestyle="--", linewidth=1, label="Background annulus inner"))
            ax.add_patch(Circle((x, y), args.aperture+5, fill=False, linestyle=":", linewidth=1, label="Background annulus outer"))
            ax.plot(x, y, marker="+", markersize=12, markeredgewidth=2, label="Predicted target position")
            if nearest:
                ax.plot(nearest["x"], nearest["y"], marker="x", markersize=9, label="Nearest detected source")
            title_time = epoch.isot if epoch else "time unavailable"
            ax.set_title(f"{args.target} {args.band}\n{title_time}\nflux={phot['flux']} | aperture-valid={phot['aperture_valid_fraction']:.2f}")
            ax.set_xlabel("Pixel X"); ax.set_ylabel("Pixel Y")
            ax.legend(loc="upper right", fontsize=7)
            fig.tight_layout()
            fig.savefig(outdir / f"diagnostic_{rank:02d}_{path.stem}.png", dpi=180)
            plt.close(fig)

    valid_flux = [r for r in rows if isinstance(r.get("flux_after_nan_safe_sum"), (int, float)) and np.isfinite(r["flux_after_nan_safe_sum"])]
    summary = {
        "target": args.target, "band": args.band, "files_found": len(files),
        "files_read": len(loaded_items), "rows_with_finite_recomputed_flux": len(valid_flux),
        "median_recomputed_flux": float(np.median([r["flux_after_nan_safe_sum"] for r in valid_flux])) if valid_flux else None,
        "output_directory": str(outdir),
        "note": "Diagnostic aperture sums use valid pixels only. Flux remains in native image units and is not absolutely calibrated. Target coordinates and WCS remain provisional; nearest detected peak is only a diagnostic, not a confirmed source association."
    }
    (outdir / "validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nOpen diagnostic images and: {csv_path}")

if __name__ == "__main__":
    main()
