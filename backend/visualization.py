"""Project NightLock: create visual comparisons from sky-aligned SPHEREx FITS.

Run from backend after processing.py --target TARGET --mode align:
  python visualization.py --target barnards_star
  python visualization.py --target wise_0855 --band SPHEREx-D1
  python visualization.py --target all --max-frames 80

Outputs are diagnostic visualisations, not calibrated science products.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from astropy.visualization import ImageNormalize, AsinhStretch
from PIL import Image


import processing

OUTPUT_ROOT = processing.PROCESSED_DIR / "visualization"
ALIGNED_ROOT = processing.ALIGNED_DIR


def read_epoch(path: Path, header, sidecar: dict):
    """Read observation time from the aligned FITS header or metadata sidecar."""
    candidates = [
        (header.get("MJD-OBS"), "mjd"),
        (header.get("DATE-OBS"), None),
        (sidecar.get("observation_time"), None),
    ]
    for value, fmt in candidates:
        if value in (None, ""):
            continue
        try:
            if fmt == "mjd":
                return Time(float(value), format="mjd")
            try:
                numeric = float(value)
                if numeric > 30000:
                    return Time(numeric, format="mjd")
            except (TypeError, ValueError):
                pass
            return Time(value)
        except Exception:
            continue
    return None


def load_aligned(path: Path):
    """Load SCIENCE data and WCS from one aligned FITS; ignore incomplete files."""
    sidecar_path = path.with_suffix(".meta.json")
    sidecar = {}
    if sidecar_path.exists():
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logging.warning("Could not read sidecar %s: %s", sidecar_path, exc)

    with fits.open(path, memmap=False) as hdul:
        science_hdu = next((h for h in hdul if str(h.name).upper() in ("SCIENCE", "SCI", "IMAGE") and h.data is not None), None)
        if science_hdu is None:
            # Support a simple PrimaryHDU image too, but prefer named SCIENCE.
            science_hdu = next((h for h in hdul if h.data is not None and np.ndim(h.data) == 2), None)
        if science_hdu is None or np.ndim(science_hdu.data) != 2:
            raise ValueError("No two-dimensional science image HDU")
        image = np.asarray(science_hdu.data, dtype=np.float32).copy()
        header = science_hdu.header.copy()
        wcs = WCS(header, relax=True).celestial

    if not wcs.has_celestial:
        raise ValueError("Image has no celestial WCS")
    epoch = read_epoch(path, header, sidecar)
    return {"path": path, "image": image, "wcs": wcs, "header": header,
            "sidecar": sidecar, "epoch": epoch}


def load_band_images(folder: Path):
    """Load valid FITS images in one aligned target/band directory."""
    paths = sorted(p for p in folder.glob("*.fits") if p.is_file())
    records = []
    for path in paths:
        try:
            records.append(load_aligned(path))
        except Exception as exc:
            logging.warning("Skipping %s: %s", path.name, exc)
    # Chronological order where timestamps exist; missing dates follow, in filename order.
    records.sort(key=lambda r: (r["epoch"] is None,
                                r["epoch"].mjd if r["epoch"] is not None else 0.0,
                                r["path"].name.lower()))
    return records


def shared_limits(records, low=1.0, high=99.5):
    """Calculate one brightness stretch for every epoch, so blinking is fair."""
    samples = []
    for rec in records:
        a = rec["image"]
        good = a[np.isfinite(a)]
        if good.size:
            # Limit memory use for large images while retaining a representative sample.
            if good.size > 100000:
                good = good[::max(1, good.size // 100000)]
            samples.append(good)
    if not samples:
        raise ValueError("No finite image pixels available")
    values = np.concatenate(samples)
    vmin, vmax = np.nanpercentile(values, [low, high])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("Could not calculate a useful shared image stretch")
    return float(vmin), float(vmax)


def display_date(rec, index):
    epoch = rec["epoch"]
    return epoch.isot[:19].replace("T", " ") if epoch is not None else f"Date unavailable (image {index + 1})"


def target_marker(target, rec):
    """Return target's predicted pixel position for that epoch, if calculable."""
    if rec["epoch"] is None:
        return None
    try:
        coord = processing.target_position(target, rec["epoch"])
        x, y = rec["wcs"].world_to_pixel(coord)
        if np.isfinite(x) and np.isfinite(y):
            return float(x), float(y)
    except Exception as exc:
        logging.debug("Could not calculate target marker: %s", exc)
    return None


def draw_image(ax, rec, target, vmin, vmax, title):
    """Render a smoother, consistently stretched display image."""

    norm = ImageNormalize(
        vmin=vmin,
        vmax=vmax,
        stretch=AsinhStretch(a=0.1),
        clip=True
    )

    ax.imshow(
        rec["image"],
        origin="lower",
        cmap="gray",
        norm=norm,
        interpolation="lanczos"
    )

    marker = target_marker(target, rec)

    if marker is not None:
        ax.plot(
            marker[0],
            marker[1],
            marker="+",
            markersize=12,
            markeredgewidth=1.8,
            color="red",
            label="Catalogue-predicted target position"
        )

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def sampled_indices(n, maximum):
    if n <= maximum:
        return list(range(n))
    return sorted(set(np.linspace(0, n - 1, maximum, dtype=int).tolist()))


def make_contact_sheet(records, target, band, limits, output, max_tiles=16):
    indices = sampled_indices(len(records), max_tiles)
    cols = min(4, len(indices))
    rows = math.ceil(len(indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, idx in zip(axes.flat, indices):
        rec = records[idx]
        draw_image(ax, rec, target, *limits, f"{idx + 1}/{len(records)} · {display_date(rec, idx)}")
        ax.axis("on")
    fig.suptitle(f"{target} | {band} | chronological sample ({len(records)} observations)")
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_blink_gif(records, target, band, limits, output, max_frames=80, duration_ms=450):
    """Create a GIF using the same pixel grid and same brightness stretch per frame."""
    frames = []
    indices = sampled_indices(len(records), max_frames)
    for idx in indices:
        rec = records[idx]
        fig, ax = plt.subplots(figsize=(6, 6))
        draw_image(ax, rec, target, *limits, f"{target} | {band}\n{display_date(rec, idx)} · epoch {idx + 1}/{len(records)}")
        ax.set_xlabel("Aligned image pixels; red + = predicted target position", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        # Convert the rendered canvas to a compact RGB PIL frame.
        rgba = np.asarray(fig.canvas.buffer_rgba())
        frame = Image.fromarray(rgba[:, :, :3].copy(), mode="RGB")
        frames.append(frame.copy())
        plt.close(fig)
    if not frames:
        raise ValueError("No frames generated")
    frames[0].save(output, save_all=True, append_images=frames[1:], duration=duration_ms,
                   loop=0, optimize=True)


def make_first_last(records, target, band, limits, output):
    first, last = records[0], records[-1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    draw_image(axes[0], first, target, *limits, f"Earliest · {display_date(first, 0)}")
    draw_image(axes[1], last, target, *limits, f"Latest · {display_date(last, len(records)-1)}")
    fig.suptitle(f"{target} | {band} | first vs last (same stretch)")
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def make_raw_difference(records, limits, output):
    """Save a raw last-minus-first diagnostic difference; not PSF/flux matched."""
    a = records[0]["image"].astype(float)
    b = records[-1]["image"].astype(float)
    if a.shape != b.shape:
        raise ValueError("First and last image dimensions differ; cannot subtract")
    valid = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(valid) < 10:
        raise ValueError("Too few overlapping finite pixels for difference image")
    diff = np.full(a.shape, np.nan, dtype=np.float32)
    diff[valid] = (b[valid] - a[valid]).astype(np.float32)
    scale = float(np.nanpercentile(np.abs(diff[valid]), 99.0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(diff, origin="lower", cmap="coolwarm", norm=SymLogNorm(
        linthresh=max(scale * 0.01, 1e-8), vmin=-scale, vmax=scale), interpolation="nearest")
    ax.set_title("Raw difference: latest − earliest\nDiagnostic only; no PSF matching or photometric scaling")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(ax.images[0], ax=ax, label="Native image-unit difference")
    fig.tight_layout(); fig.savefig(output, dpi=160, bbox_inches="tight"); plt.close(fig)


def process_band(target, band_dir, max_frames, duration_ms):
    records = load_band_images(band_dir)
    if len(records) < 2:
        logging.warning("%s: need at least two readable FITS images; found %d", band_dir, len(records))
        return None
    shapes = {r["image"].shape for r in records}
    if len(shapes) != 1:
        logging.warning("%s: image shapes differ; skipping band to avoid invalid blink comparisons", band_dir)
        return None
    limits = shared_limits(records)
    outdir = OUTPUT_ROOT / target / band_dir.name
    outdir.mkdir(parents=True, exist_ok=True)
    make_contact_sheet(records, target, band_dir.name, limits, outdir / "contact_sheet.png")
    make_blink_gif(records, target, band_dir.name, limits, outdir / "blink.gif", max_frames, duration_ms)
    make_first_last(records, target, band_dir.name, limits, outdir / "first_vs_last.png")
    try:
        make_raw_difference(records, limits, outdir / "raw_difference_latest_minus_earliest.png")
    except Exception as exc:
        logging.warning("Could not create raw difference for %s: %s", band_dir, exc)

    report = {
        "target": target,
        "band": band_dir.name,
        "readable_aligned_images": len(records),
        "first_observation": records[0]["epoch"].isot if records[0]["epoch"] is not None else None,
        "last_observation": records[-1]["epoch"].isot if records[-1]["epoch"] is not None else None,
        "images_without_observation_time": sum(r["epoch"] is None for r in records),
        "shared_display_limits_percentile_1_99_5": list(limits),
        "gif_frames": len(sampled_indices(len(records), max_frames)),
        "outputs": ["contact_sheet.png", "blink.gif", "first_vs_last.png", "raw_difference_latest_minus_earliest.png"],
        "important_caveats": [
            "The same linear display stretch is used for every epoch in a band.",
            "Only images in the same aligned target/band directory are compared.",
            "The red plus is a catalogue-predicted position based on provisional target astrometry.",
            "The raw difference is not PSF-matched or photometrically scaled; bright/dark residuals are not automatically real variability.",
            "A blink can reveal apparent shifts, but proper motion requires validated WCS, astrometry, and source association.",
        ],
        "output_directory": str(outdir),
    }
    (outdir / "visualization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logging.info("Created visual products for %s / %s (%d epochs): %s", target, band_dir.name, len(records), outdir)
    return report


def main():
    parser = argparse.ArgumentParser(description="Create NightLock contact sheets, blink GIFs and epoch comparisons")
    parser.add_argument("--target", required=True, choices=["all", *sorted(processing.TARGETS)])
    parser.add_argument("--band", default=None, help="Optional aligned band directory name, e.g. SPHEREx-D1")
    parser.add_argument("--max-frames", type=int, default=80, help="Maximum chronological frames in each GIF")
    parser.add_argument("--duration-ms", type=int, default=450, help="Display duration per GIF frame in milliseconds")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    if args.max_frames < 2 or args.duration_ms < 50:
        parser.error("--max-frames must be >= 2 and --duration-ms must be >= 50")
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(levelname)s: %(message)s")

    targets = sorted(processing.TARGETS) if args.target == "all" else [args.target]
    reports = []
    for target in targets:
        root = ALIGNED_ROOT / target
        if not root.exists():
            logging.warning("No aligned directory for %s: %s", target, root)
            continue
        if args.band:
            band_dirs = [root / args.band]
        else:
            band_dirs = sorted(p for p in root.iterdir() if p.is_dir())
        for band_dir in band_dirs:
            if not band_dir.exists():
                logging.warning("Band directory does not exist: %s", band_dir)
                continue
            report = process_band(target, band_dir, args.max_frames, args.duration_ms)
            if report:
                reports.append(report)
    if not reports:
        raise SystemExit("No visual products created. Check that processing.py --mode align produced aligned FITS files.")
    logging.info("Finished: %d target/band visualisation sets.", len(reports))


if __name__ == "__main__":
    main()
