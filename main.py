import argparse
import math
import os
from typing import Optional

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


DEFAULT_SRC = r"I:\Raster\HyperspectralPixxel2026\v1\FF02_20260106_00501045_0000004144_L2A.tif"


def _sanitize_profile_for_small_geotiffs(profile: dict) -> dict:
    """
    Keep source settings where safe, but remove tiling block sizes that can
    break when writing very small rasters (e.g., 32x32).
    """
    p = profile.copy()

    # These are common troublemakers for tiny outputs.
    for k in ("blockxsize", "blockysize"):
        p.pop(k, None)

    # If the source is tiled, writing tiny tiles with original block sizes can fail.
    # We keep compression, predictor, etc., but disable tiling for simplicity/robustness.
    if p.get("tiled", None) is True:
        p["tiled"] = False

    # BigTIFF not needed for tiny outputs; leaving it can be harmless, but some
    # drivers/versions may complain if it is a non-boolean string.
    p.pop("BIGTIFF", None)
    p.pop("bigtiff", None)

    return p


def split_to_patches(
    src_path: str,
    patch_size: int = 32,
    out_dir: Optional[str] = None,
    log_every: int = 5000,
) -> None:
    print(f"[1/6] Opening source GeoTIFF:\n  {src_path}")

    with rasterio.open(src_path) as src:
        h, w = src.height, src.width
        bands = src.count

        print(
            "[2/6] Source info:"
            f" width={w}, height={h}, bands={bands}, dtype={src.dtypes[0]}, crs={src.crs}"
        )
        print(f"      nodata={src.nodata} (mask will be used to detect NoData tiles)")

        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(src_path), "patches")

        print(f"[3/6] Ensuring output folder exists:\n  {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

        n_cols = math.ceil(w / patch_size)
        n_rows = math.ceil(h / patch_size)
        total_cells = n_cols * n_rows

        print(f"[4/6] Planning grid: patch_size={patch_size} px")
        print(f"      grid_cols={n_cols} (x=0..{n_cols - 1}), grid_rows={n_rows} (y=0..{n_rows - 1})")
        print(f"      total grid cells to evaluate: {total_cells}")

        base_profile = _sanitize_profile_for_small_geotiffs(src.profile)

        processed = 0
        written = 0
        discarded = 0

        print("[5/6] Processing windows (skipping fully-NoData tiles using the raster mask)...")

        for y in range(n_rows):
            row_off = y * patch_size
            win_h = min(patch_size, h - row_off)

            for x in range(n_cols):
                col_off = x * patch_size
                win_w = min(patch_size, w - col_off)

                window = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)

                # Uses the dataset’s internal mask / nodata handling.
                # If max == 0, there are no valid pixels in this window across all bands.
                mask = src.dataset_mask(window=window)
                if mask.max() == 0:
                    discarded += 1
                    processed += 1
                    if log_every and (processed % log_every == 0):
                        print(
                            f"  - Progress: processed={processed}/{total_cells}, "
                            f"written={written}, discarded={discarded}"
                        )
                    continue

                data = src.read(window=window)  # (bands, win_h, win_w)

                out_profile = base_profile.copy()
                out_profile.update(
                    height=win_h,
                    width=win_w,
                    transform=window_transform(window, src.transform),
                )

                out_path = os.path.join(out_dir, f"{x}_{y}.tif")

                if log_every and (processed % log_every == 0) and processed != 0:
                    print(
                        f"  - Progress: processed={processed}/{total_cells}, "
                        f"written={written}, discarded={discarded}"
                    )

                # Write patch, preserving georeferencing and the per-pixel validity mask.
                with rasterio.open(out_path, "w", **out_profile) as dst:
                    dst.write(data)
                    dst.write_mask(mask)

                written += 1
                processed += 1

        print("[6/6] Done.")
        print("Summary:")
        print(f"  - Total grid cells evaluated: {total_cells}")
        print(f"  - Written patches: {written}")
        print(f"  - Discarded (fully NoData) cells: {discarded}")
        print(f"  - Output folder: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split a GeoTIFF into 32x32 pixel patches, skipping fully-NoData patches, preserving georeferencing."
    )
    parser.add_argument(
        "--src",
        default=DEFAULT_SRC,
        help=f"Path to source GeoTIFF (default: {DEFAULT_SRC})",
    )
    parser.add_argument("--patch-size", type=int, default=32, help="Patch size in pixels (default: 32)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <src_dir>/patches)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=5000,
        help="Print progress every N grid cells (0 disables periodic progress logs).",
    )

    args = parser.parse_args()
    split_to_patches(
        src_path=args.src,
        patch_size=args.patch_size,
        out_dir=args.out_dir,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
