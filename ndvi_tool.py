#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rasterio",
#     "mercantile",
#     "numpy",
# ]
# ///
"""NDVI computation tool for multi-band GeoTIFF satellite imagery.

Reads GeoTIFF files, reprojects them into WebMercator (EPSG:3857) on a
slippy-map tile grid at a chosen zoom level, computes NDVI per tile in
parallel, and joins the results into a single output GeoTIFF.  Individual
tiles are saved as a byproduct.

In plain terms: the tool reads satellite images, cuts each one into
512x512-pixel squares on a shared world grid (the same grid web maps
use), computes a vegetation index per square in parallel, saves each
square as a separate file, and stitches them into one output image.
The shared grid guarantees that tiles from different input files cover
the same geographic areas -- no explicit alignment logic is needed.

Band order (PlanetScope PSScene4Band, 1-based):
    1 = blue, 2 = green, 3 = red, 4 = near-infrared
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import os
import pathlib
import sys

import mercantile
import numpy
import rasterio
import rasterio.enums
import rasterio.errors
import rasterio.transform
import rasterio.warp
import rasterio.windows

TILE_SIZE = 512
WEBMERCATOR_CRS = "EPSG:3857"  # Coordinate system used by web maps (meters)
BAND_RED = 3  # 1-based band index
BAND_NIR = 4
NODATA = 0  # Sentinel value marking "no data"/NULL


@dataclasses.dataclass(frozen=True, slots=True)
class NdviTile:
    """An NDVI tile: grid coordinates plus the computed NDVI array.

    Carries the tile's (x, y) position on the slippy-map grid and the
    computed NDVI array so the orchestrator can place it in the mosaic.
    """

    x: int
    y: int
    ndvi: numpy.ndarray


def tile_transform(tile: mercantile.Tile) -> rasterio.transform.Affine:
    """Build the affine transform for a TILE_SIZExTILE_SIZE WebMercator tile.

    An affine transform maps pixel coordinates (column, row) to world
    coordinates (meters in WebMercator).  It encodes the pixel size and
    the real-world location of the tile's upper-left corner, so any
    reader knows exactly where each pixel sits on Earth.
    """
    bounds = mercantile.xy_bounds(tile)

    return rasterio.transform.Affine(
        (bounds.right - bounds.left) / TILE_SIZE,
        0,
        bounds.left,
        0,
        -(bounds.top - bounds.bottom) / TILE_SIZE,
        bounds.top,
    )


def compute_ndvi(red: numpy.ndarray, nir: numpy.ndarray) -> numpy.ndarray | None:
    """Compute NDVI = (NIR - Red) / (NIR + Red) per pixel.

    NDVI ranges from -1 (water / bare ground) to +1 (dense vegetation).
    Pixels where either input band is NODATA are set to NODATA in the
    output, as are any non-finite results (division by zero).

    Returns None if the result is entirely NODATA (e.g. the tile does not
    overlap the source image).
    """
    red = red.astype(numpy.float32)
    nir = nir.astype(numpy.float32)
    valid = (red != NODATA) & (nir != NODATA)

    with numpy.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)

    ndvi[~valid | ~numpy.isfinite(ndvi)] = NODATA

    if numpy.all(ndvi == NODATA):
        return None

    return ndvi


def reproject_bands(
    src: rasterio.io.DatasetReader,
    tile: mercantile.Tile,
) -> tuple[numpy.ndarray, numpy.ndarray, rasterio.transform.Affine]:
    """Reproject (resample) red and NIR bands onto a WebMercator tile grid.

    "Reproject" means resampling pixel values from the source image's
    coordinate system into the tile's coordinate system — like resizing
    an image, but the scale factor varies across the image because the
    two coordinate systems flatten the Earth differently.  Bilinear
    interpolation (2-pixel average) is used for sub-pixel sampling.

    *src* must be an open dataset with at least 4 bands (PSScene4Band order).

    Returns (red, nir, transform) where red/nir are float32 arrays filled
    with NODATA where the tile does not overlap the source, and transform
    is the affine transform of the tile grid.
    """
    dst_transform = tile_transform(tile)
    red = numpy.full((TILE_SIZE, TILE_SIZE), NODATA, dtype=numpy.float32)
    nir = numpy.full((TILE_SIZE, TILE_SIZE), NODATA, dtype=numpy.float32)

    for band, dest in [(BAND_RED, red), (BAND_NIR, nir)]:
        rasterio.warp.reproject(
            source=rasterio.band(src, band),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=WEBMERCATOR_CRS,
            resampling=rasterio.enums.Resampling.bilinear,
            src_nodata=NODATA,
            dst_nodata=NODATA,
        )

    return red, nir, dst_transform


def process_tile(
    path: str | pathlib.Path,
    tile: mercantile.Tile,
    out_dir: pathlib.Path,
    scene_id: str,
) -> NdviTile | None:
    """Process a single tile: reproject → compute NDVI → save tile file.

    Returns the tile result (grid coordinates + NDVI array), or None if
    the tile is entirely NODATA (it falls outside the source image).
    """
    with rasterio.open(path) as src:
        red, nir, dst_transform = reproject_bands(src, tile)

    ndvi = compute_ndvi(red, nir)

    if ndvi is None:
        return None

    with rasterio.open(
        fp=out_dir / f"{scene_id}_{tile.z}_{tile.x}_{tile.y}.tif",
        mode="w",
        driver="GTiff",
        width=TILE_SIZE,
        height=TILE_SIZE,
        count=1,
        dtype="float32",
        crs=WEBMERCATOR_CRS,
        transform=dst_transform,
        nodata=NODATA,
    ) as dst:
        dst.write(arr=ndvi, indexes=1)

    return NdviTile(tile.x, tile.y, ndvi)


def process_file(
    path: str | pathlib.Path,
    out_dir: pathlib.Path,
    zoom: int,
    workers: int = 1,
) -> pathlib.Path:
    """Process one GeoTIFF end-to-end: enumerate tiles → NDVI → mosaic.

    Tiles are processed in parallel with a thread pool of *workers* threads
    (GDAL releases the GIL during reprojection, so threads give real
    parallelism).  Each finished tile is streamed directly into the mosaic
    at its grid offset, avoiding a second pass that would re-read tile
    files from disk.

    Returns the path to the output mosaic GeoTIFF.
    """
    scene_id = pathlib.Path(path).stem

    with rasterio.open(path) as src:
        bounds = rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    tiles = list(mercantile.tiles(*bounds, zooms=zoom))
    print(f"  {len(tiles)} tiles at zoom {zoom}")

    min_x = min(t.x for t in tiles)
    max_x = max(t.x for t in tiles)
    min_y = min(t.y for t in tiles)
    max_y = max(t.y for t in tiles)

    mosaic_path = out_dir / f"{scene_id}_ndvi.tif"
    origin_transform = tile_transform(mercantile.Tile(min_x, min_y, zoom))
    with (
        rasterio.open(
            fp=mosaic_path,
            mode="w",
            driver="GTiff",
            width=(max_x - min_x + 1) * TILE_SIZE,
            height=(max_y - min_y + 1) * TILE_SIZE,
            count=1,
            dtype="float32",
            crs=WEBMERCATOR_CRS,
            transform=origin_transform,
            nodata=NODATA,
            compress="lzw",
            tiled=True,
            blockxsize=TILE_SIZE,
            blockysize=TILE_SIZE,
        ) as mosaic,
        concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = {
            pool.submit(process_tile, path, tile, out_dir, scene_id): tile
            for tile in tiles
        }
        done = 0

        for future in concurrent.futures.as_completed(futures):
            result = future.result()

            if result is None:
                continue

            done += 1
            mosaic.write(
                arr=result.ndvi,
                indexes=1,
                window=rasterio.windows.Window(
                    col_off=(result.x - min_x) * TILE_SIZE,
                    row_off=(result.y - min_y) * TILE_SIZE,
                    width=TILE_SIZE,
                    height=TILE_SIZE,
                ),
            )

            print(
                f"  [{done}/{len(tiles)}] tile {result.x},{result.y}",
                end="\r",
                flush=True,
            )

    print(f"  {done}/{len(tiles)} tiles with data" + " " * 40)
    print(f"  Mosaic -> {mosaic_path}")
    return mosaic_path



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute NDVI from multi-band GeoTIFFs using WebMercator tiling.",
    )
    parser.add_argument("inputs", nargs="+", help="Input GeoTIFF file(s)")
    parser.add_argument("-o", "--output", default="output", help="Output directory")
    parser.add_argument(
        "-z",
        "--zoom",
        type=int,
        default=14,
        help="Slippy-map zoom level (default: 14, lower for fewer/larger tiles and testing)",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of parallel threads for tile processing (default: CPU count)",
    )
    args = parser.parse_args()

    for path in args.inputs:
        if not pathlib.Path(path).is_file():
            parser.error(f"Input file not found: {path}")

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in args.inputs:
        print(f"Processing {path}...")
        try:
            process_file(path, out_dir, args.zoom, workers=args.workers)
        except rasterio.errors.RasterioIOError as exc:
            print(f"  Error: cannot read {path}: {exc}", file=sys.stderr)
            sys.exit(1)
