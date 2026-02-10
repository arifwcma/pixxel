import argparse
import math
import os
from typing import Optional

import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


DEFAULT_SRC = r"I:\Raster\HyperspectralPixxel2026\v1\FF02_20260106_00501045_0000004144_L2A.tif"
DEFAULT_GRID = 32


def sanitize_profile(profile: dict) -> dict:
    p = profile.copy()
    for k in ("blockxsize", "blockysize"):
        p.pop(k, None)
    if p.get("tiled", None) is True:
        p["tiled"] = False
    p.pop("BIGTIFF", None)
    p.pop("bigtiff", None)
    return p


def split_to_patches(
    src_path: str,
    grid_size: int = DEFAULT_GRID,
    out_dir: Optional[str] = None,
    log_every: int = 100,
) -> None:
    print(f"[1/6] Opening source GeoTIFF:\n  {src_path}")

    with rasterio.open(src_path) as src:
        h, w = src.height, src.width
        bands = src.count

        patch_w = math.ceil(w / grid_size)
        patch_h = math.ceil(h / grid_size)
        total_cells = grid_size * grid_size

        print(
            f"[2/6] Source info:"
            f" width={w}, height={h}, bands={bands}, dtype={src.dtypes[0]}, crs={src.crs}"
        )
        print(f"      nodata={src.nodata}")
        print(f"      grid={grid_size}x{grid_size} = {total_cells} cells")
        print(f"      patch pixel size={patch_w}x{patch_h}")

        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(src_path), "patches")

        print(f"[3/6] Ensuring output folder exists:\n  {out_dir}")
        os.makedirs(out_dir, exist_ok=True)

        base_profile = sanitize_profile(src.profile)

        processed = 0
        written = 0
        discarded = 0

        print(f"[4/6] Processing {total_cells} grid cells (skipping fully-NoData cells)...")

        for y in range(grid_size):
            row_off = y * patch_h
            win_h = min(patch_h, h - row_off)
            if win_h <= 0:
                discarded += grid_size - y * grid_size
                break

            for x in range(grid_size):
                col_off = x * patch_w
                win_w = min(patch_w, w - col_off)
                if win_w <= 0:
                    discarded += 1
                    processed += 1
                    continue

                window = Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)

                mask = src.dataset_mask(window=window)
                if mask.max() == 0:
                    discarded += 1
                    processed += 1
                    if log_every and (processed % log_every == 0):
                        print(
                            f"  processed={processed}/{total_cells}  "
                            f"written={written}  discarded={discarded}"
                        )
                    continue

                data = src.read(window=window)

                out_profile = base_profile.copy()
                out_profile.update(
                    height=win_h,
                    width=win_w,
                    transform=window_transform(window, src.transform),
                )

                out_path = os.path.join(out_dir, f"{x}_{y}.tif")

                with rasterio.open(out_path, "w", **out_profile) as dst:
                    dst.write(data)
                    dst.write_mask(mask)

                written += 1
                processed += 1

                if log_every and (processed % log_every == 0):
                    print(
                        f"  processed={processed}/{total_cells}  "
                        f"written={written}  discarded={discarded}"
                    )

        print("[5/6] Done.")
        print(f"[6/6] Summary:")
        print(f"  Total grid cells: {total_cells}")
        print(f"  Written patches:  {written}")
        print(f"  Discarded (fully NoData): {discarded}")
        print(f"  Output folder: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--grid-size", type=int, default=DEFAULT_GRID)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--log-every", type=int, default=100)

    args = parser.parse_args()
    split_to_patches(
        src_path=args.src,
        grid_size=args.grid_size,
        out_dir=args.out_dir,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
