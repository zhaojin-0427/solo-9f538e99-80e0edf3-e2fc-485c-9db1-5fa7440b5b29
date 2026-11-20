"""科学帧平场校正管线。

输入
====
* 一张主平场成品（``/prepare`` 产出，float32 系数，带 ``INSTRUME`` /
  ``FILTER`` / ``DETSEC``）；
* 一张已扣偏置与暗电流、**无压缩**、二维 **int16** 单幅科学 FITS。

流程
====
1. 校验两侧结构、相机/滤镜一致性与 DETSEC 合法性（1 起算、含端点）。
2. 由两侧 DETSEC 与像素数推得分箱因子，仅接受 1×1 / 2×2。
3. 按探测器坐标建立科学像元 -> 平场像元的仿射映射（两轴可反向）。
4. 科学像元按 ``BSCALE`` / ``BZERO`` 换算，再除以其覆盖的平场系数均值。
5. 科学像元切分为行带流式处理：常驻内存只与条带有关，保证双
   4096×4096 图像在 256 MiB 容器内完成；写出 float32 浮点 FITS。
6. 保留科学帧的像元顺序与 WCS；BLANK、非正/非有限系数或 COUNTS<3 处填 NaN。

任何不可预期输入（缺匹配信息、范围不合规、文件损坏、写出失败）都不产出成品。
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
from astropy.io import fits

from .detector import (
    AxisLayout,
    DetectorSection,
    axis_layout,
    band_flat_start,
    detsec_from_header,
)
from .errors import PreparationError, sanitize_ascii

MAX_DIM = 4096
MIN_COUNTS = 3
# 单次行带高度：行带内系数缓冲约 ROW_BAND × 2 × 4096 × 8 B ≈ 16 MiB。
ROW_BAND = 256
FITS_BLOCK = 2880

# 从科学帧复制到成品、属于 WCS/像元顺序语义的头部关键字。
_WCS_KEYS = (
    "WCSAXES", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2",
    "CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CROTA1", "CROTA2",
    "CD1_1", "CD1_2", "CD2_1", "CD2_2", "PC1_1", "PC1_2", "PC2_1", "PC2_2",
    "LONPOLE", "LATPOLE", "RADESYS", "EQUINOX", "EPOCH", "WCSNAME",
)


@dataclass(frozen=True)
class ScienceFrame:
    path: Path
    source_name: str
    shape: tuple[int, int]
    bscale: float
    bzero: float
    blank: int | None
    header: fits.Header
    detsec: DetectorSection | None


@dataclass(frozen=True)
class MasterFlat:
    path: Path
    shape: tuple[int, int]
    header: fits.Header
    detsec: DetectorSection | None


def _open_fits(path: Path, memmap: bool) -> fits.HDUList:
    try:
        return fits.open(
            path, do_not_scale_image_data=True, memmap=memmap, lazy_load_hdus=False
        )
    except (OSError, ValueError, TypeError, fits.VerifyError) as exc:
        raise PreparationError(f"FITS 文件截断或无法解析：{path.name}：{exc}") from exc


def _reject_compressed_or_table(hdul: fits.HDUList, name: str) -> None:
    for hdu in hdul:
        if isinstance(hdu, fits.CompImageHDU):
            raise PreparationError(f"文件 {name} 含压缩图像 HDU，仅接受无压缩 FITS")
        if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
            raise PreparationError(f"文件 {name} 含表 HDU，要求单幅二维图像")


def _read_scaling(header: fits.Header, name: str) -> tuple[float, float, int | None]:
    try:
        bscale = float(header.get("BSCALE", 1.0))
        bzero = float(header.get("BZERO", 0.0))
    except (TypeError, ValueError) as exc:
        raise PreparationError(f"帧 {name} 的 BSCALE/BZERO 关键字非法：{exc}") from exc
    if not (math.isfinite(bscale) and math.isfinite(bzero)) or bscale == 0.0:
        raise PreparationError(f"帧 {name} 的 BSCALE/BZERO 非有限值或 BSCALE 为 0")
    blank: int | None = None
    if "BLANK" in header:
        try:
            blank = int(header["BLANK"])
        except (TypeError, ValueError) as exc:
            raise PreparationError(f"帧 {name} 的 BLANK 关键字不是整数：{exc}") from exc
        if not (-32768 <= blank <= 32767):
            raise PreparationError(f"帧 {name} 的 BLANK={blank} 超出 16 位整数范围")
    return bscale, bzero, blank


def _describe_science(path: Path, source_name: str) -> ScienceFrame:
    """校验科学帧头部并返回元数据；数据本身在行带循环中再读取。"""
    try:
        hdul = _open_fits(path, memmap=False)
        try:
            _reject_compressed_or_table(hdul, path.name)
            if len(hdul) != 1 or not hdul[0].is_image:
                raise PreparationError(
                    f"帧 {path.name} 须为单幅二维主图 FITS，实际含 {len(hdul)} 个 HDU"
                )
            primary = hdul[0]
            header = primary.header.copy()
            bitpix = header.get("BITPIX")
            if bitpix != 16:
                raise PreparationError(
                    f"帧 {path.name} 的 BITPIX={bitpix}，要求无压缩 16 位整数（BITPIX=16）"
                )
            shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
            data = primary.data
            if data.ndim != 2:
                raise PreparationError(
                    f"帧 {path.name} 主图为 {data.ndim} 维，要求二维图像"
                )
            if data.dtype.kind != "i" or data.dtype.itemsize != 2:
                raise PreparationError(
                    f"帧 {path.name} 主图数据类型 {data.dtype} 不是 16 位整数"
                )
            height, width = shape
            if height == 0 or width == 0 or height > MAX_DIM or width > MAX_DIM:
                raise PreparationError(
                    f"帧 {path.name} 尺寸 {width}x{height} 非法，"
                    f"宽高均须在 1~{MAX_DIM} 之间"
                )
            # 抽样读取首尾像元，尽早发现截断/损坏（整图分块读取时再完整覆盖）。
            _probe_readable(data, path.name)
            bscale, bzero, blank = _read_scaling(header, path.name)
            detsec = detsec_from_header(header, path.name)
        finally:
            hdul.close()
    except PreparationError:
        raise
    except (OSError, ValueError, EOFError, TypeError, fits.VerifyError) as exc:
        raise PreparationError(f"FITS 文件截断或数据损坏：{path.name}：{exc}") from exc
    return ScienceFrame(
        path, source_name, shape, bscale, bzero, blank, header, detsec
    )


def _probe_readable(data: np.ndarray, name: str) -> None:
    """读取四个角，触发 memmap 上可能出现的截断异常。"""
    try:
        for corner in (data[:1, :1], data[:1, -1:], data[-1:, :1], data[-1:, -1:]):
            np.asarray(corner).ravel()[:1].sum()
    except (OSError, ValueError, EOFError) as exc:
        raise PreparationError(f"FITS 文件截断或数据损坏：{name}：{exc}") from exc


def _describe_master_flat(path: Path) -> MasterFlat:
    """校验主平场成品头部：浮点二维系数主图，可带 COUNTS 扩展。"""
    try:
        hdul = _open_fits(path, memmap=True)
        try:
            _reject_compressed_or_table(hdul, path.name)
            if not hdul or not hdul[0].is_image or hdul[0].data is None:
                raise PreparationError(f"主平场 {path.name} 缺少主图数据")
            primary = hdul[0]
            header = primary.header.copy()
            data = primary.data
            if header.get("BITPIX") not in (-32, -64):
                raise PreparationError(
                    f"主平场 {path.name} 的 BITPIX={header.get('BITPIX')}，"
                    "要求浮点系数主图（BITPIX=-32 或 -64）"
                )
            if data.ndim != 2:
                raise PreparationError(
                    f"主平场 {path.name} 主图为 {data.ndim} 维，要求二维图像"
                )
            height, width = (int(data.shape[0]), int(data.shape[1]))
            if height == 0 or width == 0 or height > MAX_DIM or width > MAX_DIM:
                raise PreparationError(
                    f"主平场 {path.name} 尺寸 {width}x{height} 非法，"
                    f"宽高均须在 1~{MAX_DIM} 之间"
                )
            if int(header.get("NFRAMES", MIN_COUNTS)) < MIN_COUNTS:
                raise PreparationError(
                    f"主平场 {path.name} 的 NFRAMES={header.get('NFRAMES')} "
                    f"小于 {MIN_COUNTS}"
                )
            if len(hdul) > 1:
                extension = hdul[1]
                if extension.is_image and extension.data is not None:
                    if hdul[1].header.get("EXTNAME", "").strip().upper() != "COUNTS":
                        raise PreparationError(
                            f"主平场 {path.name} 扩展名非法，仅支持 COUNTS 扩展"
                        )
                    if extension.data.shape != data.shape:
                        raise PreparationError(
                            f"主平场 {path.name} 的 COUNTS 尺寸与主图不一致"
                        )
            _probe_readable(data, path.name)
            detsec = detsec_from_header(header, path.name)
        finally:
            hdul.close()
    except PreparationError:
        raise
    except (OSError, ValueError, EOFError, TypeError, fits.VerifyError) as exc:
        raise PreparationError(f"FITS 文件截断或数据损坏：{path.name}：{exc}") from exc
    return MasterFlat(path, (height, width), header, detsec)


def _normalized_key(header: fits.Header, key: str) -> str | None:
    value = header.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_match(flat: MasterFlat, science: ScienceFrame) -> None:
    """相机与滤镜必须一致；任一侧缺失或两侧不同都拒绝校正。"""
    for key in ("INSTRUME", "FILTER"):
        flat_value = _normalized_key(flat.header, key)
        sci_value = _normalized_key(science.header, key)
        if flat_value is None or sci_value is None:
            raise PreparationError(
                f"缺少匹配信息：主平场 {key}={flat_value!r}，"
                f"科学帧 {key}={sci_value!r}；两侧须提供相同的相机与滤镜标识"
            )
        if flat_value != sci_value:
            raise PreparationError(
                f"相机/滤镜不一致：主平场 {key}={flat_value!r}，"
                f"科学帧 {key}={sci_value!r}"
            )


def _require_detsec(
    flat: MasterFlat, science: ScienceFrame
) -> tuple[DetectorSection, DetectorSection]:
    if flat.detsec is None or science.detsec is None:
        raise PreparationError(
            "缺少匹配信息：主平场与科学帧均须带 DETSEC 才能按探测器坐标取校正系数"
        )
    return flat.detsec, science.detsec


def _layouts(flat: MasterFlat, science: ScienceFrame) -> tuple[AxisLayout, AxisLayout]:
    flat_sec, sci_sec = _require_detsec(flat, science)
    x_layout = axis_layout(
        flat_sec, flat.shape[1], sci_sec, science.shape[1], "x", science.source_name
    )
    y_layout = axis_layout(
        flat_sec, flat.shape[0], sci_sec, science.shape[0], "y", science.source_name
    )
    return x_layout, y_layout


def _stack_band(
    flat_data: np.ndarray,
    counts_data: np.ndarray | None,
    y_layout: AxisLayout,
    x_layout: AxisLayout,
    row0: int,
    row1: int,
) -> np.ndarray:
    """取一个科学行带对应的平场系数均值。

    返回形状 ``(row1-row0, 科学宽度)``；覆盖范围内任一系数非正、非有限
    或 COUNTS<3 时该科学像元为 NaN（均值对 NaN 传播）。
    """
    height, _ = flat_data.shape
    block_y, block_x = y_layout.block, x_layout.block
    fy0 = band_flat_start(y_layout, row0, row1, height)
    fy1 = fy0 + (row1 - row0) * block_y
    fx0 = x_layout.flat_min
    fx1 = x_layout.flat_min + x_layout.flat_count

    try:
        coefficients = np.array(flat_data[fy0:fy1, fx0:fx1], dtype=np.float64, copy=True)
    except (OSError, ValueError, EOFError) as exc:
        raise PreparationError(f"主平场数据读取失败：{exc}") from exc

    bad = ~np.isfinite(coefficients) | (coefficients <= 0.0)
    if counts_data is not None:
        try:
            bad |= np.asarray(counts_data[fy0:fy1, fx0:fx1]) < MIN_COUNTS
        except (OSError, ValueError, EOFError) as exc:
            raise PreparationError(f"主平场 COUNTS 读取失败：{exc}") from exc
    coefficients[bad] = np.nan

    if y_layout.flip:
        coefficients = coefficients[::-1]
    if x_layout.flip:
        coefficients = coefficients[:, ::-1]

    n_rows = row1 - row0
    n_cols = x_layout.flat_count // block_x
    # 归一到 (科学行, y 块, 科学列, x 块) 后在块轴上求均值。
    shaped = coefficients.reshape(n_rows, block_y, n_cols, block_x)
    band = shaped.mean(axis=3).mean(axis=1)
    return band.astype(np.float32)


def _iter_bands(height: int) -> Iterator[tuple[int, int]]:
    for row0 in range(0, height, ROW_BAND):
        yield row0, min(row0 + ROW_BAND, height)


def _build_header(flat: MasterFlat, science: ScienceFrame) -> fits.Header:
    """成品主图头部：二维 float32 + 科学帧 WCS/像元顺序与匹配信息。"""
    header = fits.PrimaryHDU().header
    header["BITPIX"] = (-32, "array data type")
    header["NAXIS"] = (2, "number of array dimensions")
    header["NAXIS1"] = (science.shape[1], "length of data axis 1")
    header["NAXIS2"] = (science.shape[0], "length of data axis 2")
    header["EXTEND"] = (True, "FITS dataset may contain extensions")
    header["IMAGTYPE"] = ("CALIBRATED", "Flat-field corrected science frame")
    header["BUNIT"] = ("COUNTS", "Science counts divided by flat coefficient")
    header["FLATAPPL"] = (True, "Master flat applied")
    header["CREATED"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Calibration time (UTC)",
    )
    for key in ("INSTRUME", "FILTER", "DETSEC"):
        if key in science.header:
            header[key] = science.header[key]
    for key in _WCS_KEYS:
        if key in science.header and key not in header:
            header[key] = science.header[key]
    header.add_history(
        "Science pixels scaled by BSCALE/BZERO and divided by the mean master-flat "
        "coefficient over their detector footprint; BLANK, non-positive, "
        "non-finite, or under-sampled (COUNTS<3) coefficients yield NaN."
    )
    header.add_history(f"Science input: {sanitize_ascii(science.source_name)}")
    header.add_history(f"Master flat: {sanitize_ascii(flat.path.name)}")
    return header


def _write_float32(
    header: fits.Header,
    height: int,
    width: int,
    rows: Iterator[np.ndarray],
    path: Path,
) -> None:
    """流式写出单 HDU float32 FITS：头部 + 逐行带大端数据。"""
    written = 0
    with path.open("wb") as handle:
        handle.write(header.tostring(sep="", padding=True).encode("ascii"))
        for band in rows:
            if band.shape != (band.shape[0], width) or band.dtype != np.float32:
                raise PreparationError("校正成品行带形状非法", status_code=500)
            written += band.shape[0]
            np.ascontiguousarray(band, dtype=">f4").tofile(handle)
        if written != height:
            raise PreparationError(
                f"写出失败：行数 {written} 与预期 {height} 不符", status_code=500
            )
        pad = (FITS_BLOCK - written * width * 4 % FITS_BLOCK) % FITS_BLOCK
        if pad:
            handle.write(b"\0" * pad)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_output(path: Path, shape: tuple[int, int]) -> None:
    try:
        with fits.open(path, memmap=False) as hdul:
            if len(hdul) != 1 or not hdul[0].is_image:
                raise PreparationError("校正成品结构校验失败", status_code=500)
            hdu = hdul[0]
            if hdu.header.get("BITPIX") != -32:
                raise PreparationError("校正成品类型校验失败", status_code=500)
            if (int(hdu.header["NAXIS2"]), int(hdu.header["NAXIS1"])) != shape:
                raise PreparationError("校正成品尺寸校验失败", status_code=500)
            data = np.asarray(hdu.data)
            if data.shape != shape or data.dtype.kind != "f":
                raise PreparationError("校正成品数据校验失败", status_code=500)
    except PreparationError:
        raise
    except (OSError, ValueError, fits.VerifyError) as exc:
        raise PreparationError(f"校正成品校验失败：{exc}", status_code=500) from exc


def calibrate_science(
    flat_path: Path,
    science_path: Path,
    science_name: str,
    output_dir: Path,
    work_dir: Path,
) -> Path:
    """执行平场校正；成功返回成品路径，失败不留下任何半成品。"""
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        work_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PreparationError(
            f"工作目录不可用，校正失败：{exc}", status_code=500
        ) from exc

    flat = _describe_master_flat(flat_path)
    science = _describe_science(science_path, science_name)
    _require_match(flat, science)
    x_layout, y_layout = _layouts(flat, science)
    height, width = science.shape

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    token = os.urandom(3).hex()
    final_name = f"calibrated_{timestamp}_{token}.fits"
    tmp_path = work_dir / f".{final_name}.tmp"
    final_path = output_dir / final_name

    header = _build_header(flat, science)

    def bands(flat_hdul: fits.HDUList, science_hdul: fits.HDUList) -> Iterator[np.ndarray]:
        flat_data = flat_hdul[0].data
        science_data = science_hdul[0].data
        counts_data = None
        if len(flat_hdul) > 1 and flat_hdul[1].is_image and flat_hdul[1].data is not None:
            counts_data = flat_hdul[1].data
        for row0, row1 in _iter_bands(height):
            try:
                raw = np.asarray(science_data[row0:row1]).astype(np.int32, copy=False)
            except (OSError, ValueError, EOFError) as exc:
                raise PreparationError(
                    f"科学帧数据读取失败：{science.source_name}：{exc}"
                ) from exc
            scaled = raw.astype(np.float64) * science.bscale + science.bzero
            bad = np.zeros(scaled.shape, dtype=bool)
            if science.blank is not None:
                bad = raw == science.blank
            mean_flat = _stack_band(
                flat_data, counts_data, y_layout, x_layout, row0, row1
            )
            with np.errstate(invalid="ignore", divide="ignore"):
                corrected = scaled / mean_flat.astype(np.float64)
            bad |= ~np.isfinite(corrected)
            corrected[bad] = np.nan
            yield corrected.astype(np.float32)

    flat_hdul = _open_fits(flat_path, memmap=True)
    science_hdul = _open_fits(science_path, memmap=True)
    try:
        _write_float32(header, height, width, bands(flat_hdul, science_hdul), tmp_path)
        _verify_output(tmp_path, science.shape)
        os.replace(tmp_path, final_path)
    except PreparationError:
        tmp_path.unlink(missing_ok=True)
        raise
    except (OSError, ValueError, EOFError, fits.VerifyError) as exc:
        tmp_path.unlink(missing_ok=True)
        raise PreparationError(
            f"校正成品写出或数据读取失败：{exc}", status_code=500
        ) from exc
    finally:
        flat_hdul.close()
        science_hdul.close()
    return final_path
