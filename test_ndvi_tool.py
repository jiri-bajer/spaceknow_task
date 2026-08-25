# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest",
#     "rasterio",
#     "mercantile",
#     "numpy",
# ]
# ///
"""Test suite for ndvi_tool.

Unit tests cover the pure functions (compute_ndvi, tile_transform); E2E
tests build tiny synthetic 4-band GeoTIFFs and drive the CLI via subprocess.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import mercantile
import numpy
import pytest
import rasterio
import rasterio.transform

import ndvi_tool

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

# Footprint anchors in EPSG:32633 (UTM zone 33N, ~50°N).  Two 256×256 scenes
# at 10 m/px overlapping by ~1.3 km east / ~1.5 km north.  At zoom 14 this
# yields 6 tiles per scene with 4 shared tile coordinates.
_ANCHOR_A = (440_000.0, 5_559_000.0)
_ANCHOR_B = (440_000.0 + 128 * 10.0, 5_559_000.0 + 100 * 10.0)
_PIXEL = 10.0
_SIZE = 256
_ZOOM = 14


def _make_synthetic_geotiff(
    path: pathlib.Path,
    anchor: tuple[float, float],
    red: int,
    nir: int,
) -> pathlib.Path:
    """Write a tiny 4-band GeoTIFF (PSScene4Band order) in EPSG:32633.

    Bands 1-2 are filled with a neutral constant; bands 3 (red) and 4 (NIR)
    use the given values so NDVI is deterministic.  NODATA is 0.
    """
    west, south = anchor
    north = south + _SIZE * _PIXEL
    transform = rasterio.transform.from_origin(west, north, _PIXEL, _PIXEL)

    with rasterio.open(
        path,
        mode="w",
        driver="GTiff",
        width=_SIZE,
        height=_SIZE,
        count=4,
        dtype="uint16",
        crs="EPSG:32633",
        transform=transform,
        nodata=ndvi_tool.NODATA,
    ) as dst:
        for band in (1, 2):
            dst.write(numpy.full((_SIZE, _SIZE), 100, dtype=numpy.uint16), band)

        dst.write(numpy.full((_SIZE, _SIZE), red, dtype=numpy.uint16), 3)
        dst.write(numpy.full((_SIZE, _SIZE), nir, dtype=numpy.uint16), 4)

    return path


def _run_tool(
    out_dir: pathlib.Path,
    *inputs: pathlib.Path,
    zoom: int = _ZOOM,
) -> subprocess.CompletedProcess[str]:
    """Run ndvi_tool.py as a subprocess and return the completed process."""
    return subprocess.run(
        [
            sys.executable,
            str(pathlib.Path(ndvi_tool.__file__)),
            "-o",
            str(out_dir),
            "-z",
            str(zoom),
            *map(str, inputs),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )


def _tile_coords(
    out_dir: pathlib.Path, scene_id: str, zoom: int
) -> set[tuple[int, int]]:
    """Collect (x, y) tile coordinates from byproduct filenames in out_dir."""
    pattern = re.compile(rf"^{re.escape(scene_id)}_{zoom}_(\d+)_(\d+)\.tif$")
    coords: set[tuple[int, int]] = set()

    for p in out_dir.glob(f"{scene_id}_{zoom}_*.tif"):
        m = pattern.match(p.name)

        if m:
            coords.add((int(m.group(1)), int(m.group(2))))

    return coords


# --------------------------------------------------------------------------- #
# Unit tests — compute_ndvi
# --------------------------------------------------------------------------- #


class TestComputeNdvi:
    def test_basic_formula(self) -> None:
        # GIVEN red band is 100 and NIR band is 200 for all pixels
        red = numpy.full((4, 4), 100, dtype=numpy.float32)
        nir = numpy.full((4, 4), 200, dtype=numpy.float32)

        # WHEN NDVI is computed
        ndvi = ndvi_tool.compute_ndvi(red, nir)

        # THEN the result is (200-100)/(200+100) ≈ 0.333 everywhere
        assert ndvi is not None
        expected = numpy.float32((200 - 100) / (200 + 100))
        assert numpy.allclose(ndvi, expected)

    def test_negative_ndvi(self) -> None:
        # GIVEN red band is 200 and NIR band is 100 (more red than NIR)
        red = numpy.full((4, 4), 200, dtype=numpy.float32)
        nir = numpy.full((4, 4), 100, dtype=numpy.float32)

        # WHEN NDVI is computed
        ndvi = ndvi_tool.compute_ndvi(red, nir)

        # THEN the result is (100-200)/(100+200) ≈ -0.333 everywhere
        assert ndvi is not None
        expected = numpy.float32((100 - 200) / (100 + 200))
        assert numpy.allclose(ndvi, expected)

    def test_all_nodata_returns_none(self) -> None:
        # GIVEN both red and NIR bands are entirely NODATA (0)
        red = numpy.zeros((4, 4), dtype=numpy.float32)
        nir = numpy.zeros((4, 4), dtype=numpy.float32)

        # WHEN NDVI is computed
        # THEN None is returned because all pixels are NODATA
        assert ndvi_tool.compute_ndvi(red, nir) is None

    def test_red_band_all_nodata_returns_none(self) -> None:
        # GIVEN red band is entirely NODATA but NIR has valid data
        red = numpy.zeros((4, 4), dtype=numpy.float32)
        nir = numpy.full((4, 4), 200, dtype=numpy.float32)

        # WHEN NDVI is computed
        # THEN None is returned because every pixel is masked by red NODATA
        assert ndvi_tool.compute_ndvi(red, nir) is None

    def test_mixed_valid_and_invalid(self) -> None:
        # GIVEN a 2×2 array where column 0 has valid data and column 1 is
        # NODATA in the red band
        red = numpy.array(
            [[100, 0], [100, 0]],
            dtype=numpy.float32,
        )
        nir = numpy.array(
            [[200, 200], [200, 200]],
            dtype=numpy.float32,
        )

        # WHEN NDVI is computed
        ndvi = ndvi_tool.compute_ndvi(red, nir)

        # THEN NODATA pixels are masked to 0 and valid pixels hold the formula
        assert ndvi is not None
        expected_valid = numpy.float32((200 - 100) / (200 + 100))
        assert ndvi[0, 1] == ndvi_tool.NODATA
        assert ndvi[1, 1] == ndvi_tool.NODATA
        assert numpy.isclose(ndvi[0, 0], expected_valid)
        assert numpy.isclose(ndvi[1, 0], expected_valid)

    def test_dtype_is_float32(self) -> None:
        # GIVEN red and NIR bands are uint16
        red = numpy.full((2, 2), 100, dtype=numpy.uint16)
        nir = numpy.full((2, 2), 200, dtype=numpy.uint16)

        # WHEN NDVI is computed
        ndvi = ndvi_tool.compute_ndvi(red, nir)

        # THEN the output array is float32
        assert ndvi is not None
        assert ndvi.dtype == numpy.float32

    def test_division_by_zero_masked(self) -> None:
        # GIVEN nir + red == 0 but neither equals NODATA, producing inf/nan
        red = numpy.full((2, 2), -50.0, dtype=numpy.float32)
        nir = numpy.full((2, 2), 50.0, dtype=numpy.float32)

        # WHEN NDVI is computed
        # THEN the non-finite result is masked to NODATA everywhere → None
        assert ndvi_tool.compute_ndvi(red, nir) is None

    def test_extreme_ratio_stays_in_range(self) -> None:
        # GIVEN a very small red value and a very large NIR value
        red = numpy.full((2, 2), 1, dtype=numpy.float32)
        nir = numpy.full((2, 2), 10_000, dtype=numpy.float32)

        # WHEN NDVI is computed
        ndvi = ndvi_tool.compute_ndvi(red, nir)

        # THEN NDVI approaches but does not exceed 1.0
        assert ndvi is not None
        assert numpy.all(ndvi > 0.99)
        assert numpy.all(ndvi < 1.0)


# --------------------------------------------------------------------------- #
# Unit tests — tile_transform
# --------------------------------------------------------------------------- #


class TestTileTransform:
    def test_known_tile_affine(self) -> None:
        # GIVEN a known slippy-map tile at z=11, x=1104, y=692
        tile = mercantile.Tile(1104, 692, 11)
        b = mercantile.xy_bounds(tile)

        # WHEN the tile transform is computed
        expected = rasterio.transform.Affine(
            (b.right - b.left) / ndvi_tool.TILE_SIZE,
            0,
            b.left,
            0,
            -(b.top - b.bottom) / ndvi_tool.TILE_SIZE,
            b.top,
        )

        # THEN it matches the expected Affine derived from xy_bounds
        assert ndvi_tool.tile_transform(tile) == expected

    def test_pixel_size_matches_tile_size(self) -> None:
        # GIVEN a known slippy-map tile at z=11, x=1104, y=692
        tile = mercantile.Tile(1104, 692, 11)
        b = mercantile.xy_bounds(tile)

        # WHEN the tile transform is computed
        aff = ndvi_tool.tile_transform(tile)

        # THEN pixel dimensions and origin match the tile bounds / TILE_SIZE
        assert aff.a == pytest.approx((b.right - b.left) / ndvi_tool.TILE_SIZE)
        assert aff.e == pytest.approx(-(b.top - b.bottom) / ndvi_tool.TILE_SIZE)
        assert aff.c == pytest.approx(b.left)
        assert aff.f == pytest.approx(b.top)

    def test_transform_covers_exact_tile_bounds(self) -> None:
        # GIVEN a known slippy-map tile at z=11, x=1104, y=692
        tile = mercantile.Tile(1104, 692, 11)
        b = mercantile.xy_bounds(tile)

        # WHEN the tile transform is computed
        aff = ndvi_tool.tile_transform(tile)

        # THEN pixel (0,0) maps to the tile origin and (512,512) to the extent
        assert aff * (0, 0) == pytest.approx((b.left, b.top))
        assert aff * (ndvi_tool.TILE_SIZE, ndvi_tool.TILE_SIZE) == pytest.approx(
            (b.right, b.bottom)
        )


# --------------------------------------------------------------------------- #
# E2E tests — CLI via subprocess
# --------------------------------------------------------------------------- #


@pytest.fixture()
def scene_a(tmp_path: pathlib.Path) -> pathlib.Path:
    return _make_synthetic_geotiff(
        tmp_path / "scene_a.tif", _ANCHOR_A, red=100, nir=200
    )


@pytest.fixture()
def scene_b(tmp_path: pathlib.Path) -> pathlib.Path:
    return _make_synthetic_geotiff(
        tmp_path / "scene_b.tif", _ANCHOR_B, red=200, nir=100
    )


class TestEndToEnd:
    def test_mosaic_and_tiles_produced(
        self, tmp_path: pathlib.Path, scene_a: pathlib.Path, scene_b: pathlib.Path
    ) -> None:
        # GIVEN two synthetic 4-band GeoTIFFs with known band values
        # (provided by scene_a and scene_b fixtures)
        out_dir = tmp_path / "out"

        # WHEN the tool is run on both inputs at zoom 14
        result = _run_tool(out_dir, scene_a, scene_b)

        # THEN the CLI exits successfully and produces a mosaic + tiles per scene
        assert result.returncode == 0, result.stderr

        for scene in (scene_a, scene_b):
            scene_id = scene.stem
            mosaic = out_dir / f"{scene_id}_ndvi.tif"

            assert mosaic.is_file(), f"missing mosaic for {scene_id}"
            with rasterio.open(mosaic) as src:
                assert src.crs == "EPSG:3857"
                assert src.count == 1
                assert src.dtypes[0] == "float32"
                assert src.nodata == ndvi_tool.NODATA

                data = src.read(1)
                valid = data != ndvi_tool.NODATA
                assert valid.any(), f"mosaic for {scene_id} is entirely NODATA"
                assert data[valid].min() >= -1.0
                assert data[valid].max() <= 1.0

            coords = _tile_coords(out_dir, scene_id, _ZOOM)
            assert coords, f"no tile byproducts for {scene_id}"

    def test_overlapping_scenes_share_tile_coordinates(
        self, tmp_path: pathlib.Path, scene_a: pathlib.Path, scene_b: pathlib.Path
    ) -> None:
        # GIVEN two synthetic GeoTIFFs with overlapping footprints
        # (provided by scene_a and scene_b fixtures)
        out_dir = tmp_path / "out"

        # WHEN the tool is run on both inputs
        result = _run_tool(out_dir, scene_a, scene_b)

        # THEN the two scenes share at least one tile coordinate,
        # demonstrating slippy-map grid alignment
        assert result.returncode == 0, result.stderr

        coords_a = _tile_coords(out_dir, scene_a.stem, _ZOOM)
        coords_b = _tile_coords(out_dir, scene_b.stem, _ZOOM)
        assert coords_a & coords_b, "overlapping scenes share no tile coordinates"

    def test_file_not_found_exits_argparse_error(self, tmp_path: pathlib.Path) -> None:
        # GIVEN a path to a file that does not exist
        out_dir = tmp_path / "out"
        missing = tmp_path / "does_not_exist.tif"

        # WHEN the tool is invoked with that path
        result = _run_tool(out_dir, missing)

        # THEN argparse rejects it with exit code 2 and a "not found" message
        assert result.returncode == 2
        assert "not found" in result.stderr.lower()

    def test_corrupted_input_fails_gracefully(self, tmp_path: pathlib.Path) -> None:
        # GIVEN an empty file that is not a valid GeoTIFF
        out_dir = tmp_path / "out"
        bad = tmp_path / "corrupted.tif"
        bad.write_bytes(b"")

        # WHEN the tool is invoked with the corrupted file
        result = _run_tool(out_dir, bad)

        # THEN it terminates with a non-zero exit and a clean error message,
        # without a raw Python stack trace
        assert result.returncode != 0
        assert "Traceback" not in result.stderr
        assert "cannot read" in result.stderr.lower()
        assert out_dir.is_dir()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
