"""科学帧平场校正端到端测试：数值、坐标映射、失败路径。"""

from __future__ import annotations

import io

import numpy as np
import pytest
from astropy.io import fits

from app.calibrate import calibrate_science
from app.errors import PreparationError

MIN_COUNTS = 3


def write_master_flat(
    path,
    coeffs: np.ndarray,
    detsec: str | None,
    counts: np.ndarray | None = None,
    instrument: str = "CAM-1",
    filt: str = "V",
    nframes: int = 5,
):
    """写一个 /prepare 样式的成品：float32 系数主图 + 可选 COUNTS 扩展。"""
    primary = fits.PrimaryHDU(data=np.asarray(coeffs, dtype=np.float32))
    primary.header["IMAGTYPE"] = "MASTERFLAT"
    primary.header["NFRAMES"] = nframes
    primary.header["INSTRUME"] = instrument
    primary.header["FILTER"] = filt
    if detsec is not None:
        primary.header["DETSEC"] = detsec
    hdus = [primary]
    if counts is not None:
        hdus.append(fits.ImageHDU(data=np.asarray(counts, dtype=np.uint8), name="COUNTS"))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def write_science(path, raw: np.ndarray, detsec: str | None = "[1:8,1:6]", **header):
    """写一张无压缩 int16 科学帧。"""
    hdu = fits.PrimaryHDU(data=np.asarray(raw, dtype=np.int16))
    if detsec is not None:
        hdu.header["DETSEC"] = detsec
    for key, value in header.items():
        hdu.header[key] = value
    fits.HDUList([hdu]).writeto(path, overwrite=True)
    return path


def run(tmp_path, flat_path, science_path, name="sci.fits"):
    return calibrate_science(
        flat_path=flat_path,
        science_path=science_path,
        science_name=name,
        output_dir=tmp_path / "out",
        work_dir=tmp_path / "work",
    )


def test_identity_detsec_divides_by_matching_flat(tmp_path):
    coeffs = np.array([[1.0, 2.0], [4.0, 8.0]], dtype=np.float32)
    counts = np.full((2, 2), 5, dtype=np.uint8)
    flat = write_master_flat(tmp_path / "flat.fits", coeffs, "[1:2,1:2]", counts)
    raw = np.array([[100, 100], [100, 100]], dtype=np.int16)
    sci = write_science(tmp_path / "sci.fits", raw, "[1:2,1:2]",
                        INSTRUME="CAM-1", FILTER="V")
    product = run(tmp_path, flat, sci)
    with fits.open(product) as out:
        data = out[0].data
        assert out[0].header["BITPIX"] == -32
        assert out[0].header["IMAGTYPE"] == "CALIBRATED"
        np.testing.assert_allclose(
            data, np.array([[100.0, 50.0], [25.0, 12.5]], dtype=np.float32)
        )


def test_blank_and_bad_coefficients_become_nan(tmp_path):
    coeffs = np.array([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32)
    counts = np.array([[5, 5], [5, 2]], dtype=np.uint8)  # COUNTS=2 -> NaN
    flat = write_master_flat(tmp_path / "flat.fits", coeffs, "[1:2,1:2]", counts)
    raw = np.array([[-5, 0], [100, 100]], dtype=np.int16)  # BLANK = -5
    hdu = fits.PrimaryHDU(data=raw)
    hdu.header["DETSEC"] = "[1:2,1:2]"
    hdu.header["BLANK"] = -5
    hdu.header["INSTRUME"] = "CAM-1"
    hdu.header["FILTER"] = "V"
    sci = tmp_path / "sci.fits"
    fits.HDUList([hdu]).writeto(sci, overwrite=True)
    product = run(tmp_path, flat, sci)
    with fits.open(product) as out:
        data = out[0].data
    assert np.isnan(data).all()


def test_science_window_inside_flat_and_bscale_bzero(tmp_path):
    # 平场覆盖 4x4 探测器，科学帧只读取 [2:3,2:3] 的子窗。
    coeffs = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    counts = np.full((4, 4), 7, dtype=np.uint8)
    flat = write_master_flat(tmp_path / "flat.fits", coeffs, "[1:4,1:4]", counts)
    raw = np.full((2, 2), 2000, dtype=np.int16)
    sci = write_science(tmp_path / "sci.fits", raw, "[2:3,2:3]",
                        INSTRUME="CAM-1", FILTER="V",
                        BSCALE=0.5, BZERO=100.0)
    product = run(tmp_path, flat, sci)
    with fits.open(product) as out:
        data = out[0].data
    # 科学值 = 2000*0.5 + 100 = 1100；平场子窗 = coeffs[1:3, 1:3] = [[6,7],[10,11]]
    np.testing.assert_allclose(
        data, (1100.0 / np.array([[6.0, 7.0], [10.0, 11.0]])).astype(np.float32)
    )


def test_reversed_axes_map_to_flipped_flat(tmp_path):
    coeffs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    counts = np.full((2, 2), 9, dtype=np.uint8)
    flat = write_master_flat(tmp_path / "flat.fits", coeffs, "[1:2,1:2]", counts)
    raw = np.full((2, 2), 100, dtype=np.int16)
    # 科学帧两轴都反向：sci 像素 (0,0) 对应探测器 (2,2) -> 平场数组 (1,1)。
    sci = write_science(tmp_path / "sci.fits", raw, "[2:1,2:1]",
                        INSTRUME="CAM-1", FILTER="V")
    product = run(tmp_path, flat, sci)
    with fits.open(product) as out:
        data = out[0].data
    np.testing.assert_allclose(
        data, (100.0 / coeffs[::-1, ::-1]).astype(np.float32)
    )


def test_2x2_science_binning_averages_2x2_flat_coefficients(tmp_path):
    coeffs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 8.0], [7.0, 8.0]],
                      dtype=np.float32)  # 4 行 2 列
    counts = np.full((4, 2), 4, dtype=np.uint8)
    flat = write_master_flat(tmp_path / "flat.fits", coeffs, "[1:2,1:4]", counts)
    raw = np.full((2, 1), 100, dtype=np.int16)  # 科学帧 2x2 分箱 -> 2 行 1 列
    sci = write_science(tmp_path / "sci.fits", raw, "[1:2,1:4]",
                        INSTRUME="CAM-1", FILTER="V")
    product = run(tmp_path, flat, sci)
    with fits.open(product) as out:
        data = out[0].data
    # 行块均值：第 0 行 (1+2+3+4)/4 = 2.5；第 1 行 (5+8+7+8)/4 = 7.0
    np.testing.assert_allclose(data[:, 0], np.array([40.0, 100.0 / 7.0], np.float32))


def test_science_2x2_over_flat_1x1(tmp_path):
    coeffs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    counts = np.full((2, 2), 4, dtype=np.uint8)
    flat = write_master_flat(tmp_path / "flat.fits", coeffs, "[1:2,1:2]", counts)
    raw = np.full((1, 1), 90, dtype=np.int16)  # 2x2 分箱 -> 1 像素
    sci = write_science(tmp_path / "sci.fits", raw, "[1:2,1:2]",
                        INSTRUME="CAM-1", FILTER="V")
    product = run(tmp_path, flat, sci)
    with fits.open(product) as out:
        assert out[0].data[0, 0] == pytest.approx(90.0 / 2.5, rel=1e-6)


def test_wcs_header_preserved(tmp_path):
    coeffs = np.ones((2, 2), dtype=np.float32)
    counts = np.full((2, 2), 5, dtype=np.uint8)
    flat = write_master_flat(tmp_path / "flat.fits", coeffs, "[1:2,1:2]", counts)
    sci = write_science(
        tmp_path / "sci.fits", np.full((2, 2), 10, np.int16), "[1:2,1:2]",
        INSTRUME="CAM-1", FILTER="V",
        CRPIX1=2.0, CRPIX2=1.5, CRVAL1=150.25, CRVAL2=-30.5,
        CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CD1_1=-1e-4, CD1_2=0.0,
        CD2_1=0.0, CD2_2=1e-4, RADESYS="ICRS",
    )
    product = run(tmp_path, flat, sci)
    with fits.open(product) as out:
        header = out[0].header
        assert header["CRPIX1"] == 2.0
        assert header["CRVAL1"] == 150.25
        assert header["CTYPE1"] == "RA---TAN"
        assert header["CD1_1"] == -1e-4
        assert header["RADESYS"] == "ICRS"
        assert header["DETSEC"] == "[1:2,1:2]"
        # 像元顺序保留：数据形状与科学帧一致。
        assert out[0].data.shape == (2, 2)


def test_missing_instrument_rejected(tmp_path):
    flat = write_master_flat(
        tmp_path / "flat.fits", np.ones((2, 2), np.float32), "[1:2,1:2]",
        np.full((2, 2), 5, np.uint8), instrument="CAM-1", filt="V",
    )
    sci = write_science(tmp_path / "sci.fits", np.ones((2, 2), np.int16),
                        "[1:2,1:2]", FILTER="V")
    with pytest.raises(PreparationError, match="缺少匹配信息"):
        run(tmp_path, flat, sci)
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_filter_mismatch_rejected(tmp_path):
    flat = write_master_flat(
        tmp_path / "flat.fits", np.ones((2, 2), np.float32), "[1:2,1:2]",
        np.full((2, 2), 5, np.uint8), instrument="CAM-1", filt="V",
    )
    sci = write_science(tmp_path / "sci.fits", np.ones((2, 2), np.int16),
                        "[1:2,1:2]", INSTRUME="CAM-1", FILTER="R")
    with pytest.raises(PreparationError, match="不一致"):
        run(tmp_path, flat, sci)


def test_missing_detsec_rejected(tmp_path):
    flat = write_master_flat(
        tmp_path / "flat.fits", np.ones((2, 2), np.float32), "[1:2,1:2]",
        np.full((2, 2), 5, np.uint8),
    )
    sci = write_science(tmp_path / "sci.fits", np.ones((2, 2), np.int16), None,
                        INSTRUME="CAM-1", FILTER="V")
    with pytest.raises(PreparationError, match="缺少匹配信息"):
        run(tmp_path, flat, sci)


def test_science_window_outside_flat_rejected(tmp_path):
    flat = write_master_flat(
        tmp_path / "flat.fits", np.ones((2, 2), np.float32), "[1:2,1:2]",
        np.full((2, 2), 5, np.uint8),
    )
    sci = write_science(tmp_path / "sci.fits", np.ones((2, 2), np.int16),
                        "[2:3,1:2]", INSTRUME="CAM-1", FILTER="V")
    with pytest.raises(PreparationError, match="超出主平场覆盖范围"):
        run(tmp_path, flat, sci)


def test_truncated_science_rejected(tmp_path):
    flat = write_master_flat(
        tmp_path / "flat.fits", np.ones((64, 64), np.float32), "[1:64,1:64]",
        np.full((64, 64), 5, np.uint8),
    )
    sci = write_science(tmp_path / "sci.fits",
                        np.ones((64, 64), np.int16), "[1:64,1:64]",
                        INSTRUME="CAM-1", FILTER="V")
    blob = sci.read_bytes()
    sci.write_bytes(blob[: len(blob) // 2])
    with pytest.raises(PreparationError, match="截断|无法解析|损坏|读取失败"):
        run(tmp_path, flat, sci)
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_float_science_rejected(tmp_path):
    flat = write_master_flat(
        tmp_path / "flat.fits", np.ones((2, 2), np.float32), "[1:2,1:2]",
        np.full((2, 2), 5, np.uint8),
    )
    hdu = fits.PrimaryHDU(data=np.ones((2, 2), np.float32))
    hdu.header["DETSEC"] = "[1:2,1:2]"
    hdu.header["INSTRUME"] = "CAM-1"
    hdu.header["FILTER"] = "V"
    sci = tmp_path / "sci.fits"
    fits.HDUList([hdu]).writeto(sci, overwrite=True)
    with pytest.raises(PreparationError, match="BITPIX"):
        run(tmp_path, flat, sci)


def test_compressed_science_rejected(tmp_path):
    flat = write_master_flat(
        tmp_path / "flat.fits", np.ones((2, 2), np.float32), "[1:2,1:2]",
        np.full((2, 2), 5, np.uint8),
    )
    raw = np.ones((2, 2), np.int16)
    comp = fits.CompImageHDU(data=raw)
    comp.header["DETSEC"] = "[1:2,1:2]"
    comp.header["INSTRUME"] = "CAM-1"
    comp.header["FILTER"] = "V"
    sci = tmp_path / "sci.fits"
    fits.HDUList([fits.PrimaryHDU(), comp]).writeto(sci, overwrite=True)
    with pytest.raises(PreparationError, match="压缩"):
        run(tmp_path, flat, sci)


def test_unsupported_binning_rejected(tmp_path):
    # 科学帧跨度 4 倍于像素数 = 4x4 分箱，超出支持范围。
    flat = write_master_flat(
        tmp_path / "flat.fits", np.ones((8, 8), np.float32), "[1:8,1:8]",
        np.full((8, 8), 5, np.uint8),
    )
    sci = write_science(tmp_path / "sci.fits", np.ones((2, 8), np.int16),
                        "[1:8,1:8]", INSTRUME="CAM-1", FILTER="V")
    with pytest.raises(PreparationError, match="分箱"):
        run(tmp_path, flat, sci)


def test_write_failure_leaves_no_product(tmp_path):
    flat = write_master_flat(
        tmp_path / "flat.fits", np.ones((2, 2), np.float32), "[1:2,1:2]",
        np.full((2, 2), 5, np.uint8),
    )
    sci = write_science(tmp_path / "sci.fits", np.ones((2, 2), np.int16),
                        "[1:2,1:2]", INSTRUME="CAM-1", FILTER="V")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "child").write_text("not a directory")
    with pytest.raises(PreparationError) as info:
        calibrate_science(
            flat_path=flat, science_path=sci, science_name="sci.fits",
            output_dir=blocked / "child" / "out", work_dir=tmp_path / "work",
        )
    assert info.value.status_code == 500
    assert not list((tmp_path / "work").rglob("*.tmp"))
