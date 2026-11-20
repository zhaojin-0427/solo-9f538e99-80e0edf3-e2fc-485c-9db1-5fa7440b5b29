"""探测器坐标（DETSEC）解析与平场/科学帧几何映射。

DETSEC 约定（与 IRAF/IRTF 等仪器惯例一致）：

* 形如 ``[x1:x2,y1:y2]``，可带前导分区号 ``1:[x1:x2,y1:y2]``；
* 坐标从 1 起算、两端闭区间；
* 两个轴都允许反向（``x1 > x2`` 或 ``y1 > y2``），反向表示数据数组沿该轴
  从高坐标向低坐标排列；
* DETSEC 始终以**未分箱**探测器坐标表示；图像实际像素数决定分箱因子。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .errors import PreparationError

# 允许可选分区前缀与括号内四段整数；不接受浮点或表达式。
_DETSEC_RE = re.compile(
    r"^\s*(?:[+-]?\d+\s*:\s*)?"
    r"\[\s*([+-]?\d+)\s*:\s*([+-]?\d+)\s*,\s*"
    r"([+-]?\d+)\s*:\s*([+-]?\d+)\s*\]\s*$"
)

SUPPORTED_BINNING = (1, 2)


@dataclass(frozen=True)
class DetectorSection:
    """未分箱探测器坐标窗口。"""

    x1: int
    x2: int
    y1: int
    y2: int

    @property
    def x_span(self) -> int:
        return abs(self.x2 - self.x1) + 1

    @property
    def y_span(self) -> int:
        return abs(self.y2 - self.y1) + 1

    @property
    def x_low(self) -> int:
        return min(self.x1, self.x2)

    @property
    def x_high(self) -> int:
        return max(self.x1, self.x2)

    @property
    def y_low(self) -> int:
        return min(self.y1, self.y2)

    @property
    def y_high(self) -> int:
        return max(self.y1, self.y2)

    @property
    def x_reversed(self) -> bool:
        return self.x1 > self.x2

    @property
    def y_reversed(self) -> bool:
        return self.y1 > self.y2

    def text(self) -> str:
        return f"[{self.x1}:{self.x2},{self.y1}:{self.y2}]"


def parse_detsec(value: object, name: str) -> DetectorSection:
    """解析 DETSEC 头部值；格式非法即视为不可用输入。"""
    if isinstance(value, bytes):
        try:
            text = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise PreparationError(f"帧 {name} 的 DETSEC 不是 ASCII 文本：{exc}") from exc
    elif isinstance(value, str):
        text = value
    else:
        raise PreparationError(
            f"帧 {name} 的 DETSEC 类型 {type(value).__name__} 非法，须为字符串"
        )

    match = _DETSEC_RE.match(text)
    if match is None:
        raise PreparationError(
            f"帧 {name} 的 DETSEC {text!r} 格式非法，须为 [x1:x2,y1:y2]"
        )
    x1, x2, y1, y2 = (int(part) for part in match.groups())
    return DetectorSection(x1, x2, y1, y2)


def detsec_from_header(header, name: str) -> DetectorSection | None:
    """读取可选 DETSEC；关键字缺失或空字符串返回 None。"""
    if "DETSEC" not in header:
        return None
    value = header["DETSEC"]
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return parse_detsec(value, name)


def binning_from_span(
    span: int, pixels: int, axis: str, name: str, section: DetectorSection
) -> int:
    """由 DETSEC 跨度与图像像素数反推分箱因子，仅允许 1 或 2。"""
    if pixels <= 0:
        raise PreparationError(f"帧 {name} 的 {axis} 轴像素数非法（{pixels}）")
    if span <= 0:
        raise PreparationError(f"帧 {name} 的 DETSEC {section.text()} 跨度非法")
    if span % pixels:
        raise PreparationError(
            f"帧 {name} 的 DETSEC {axis} 轴跨度 {span} 与图像尺寸 {pixels} "
            f"不整除，无法确定 {axis} 轴分箱"
        )
    factor = span // pixels
    if factor not in SUPPORTED_BINNING:
        raise PreparationError(
            f"帧 {name} 的 {axis} 轴分箱为 {factor}x，仅支持 1x1 与 2x2 分箱"
        )
    return factor


def check_binning_header(
    header, name: str, candidates: tuple[str, ...], expected: int, axis: str
) -> None:
    """若头部声明了分箱因子，则必须与 DETSEC 推得的因子一致。"""
    for key in candidates:
        if key not in header:
            continue
        value = header[key]
        try:
            declared = int(value)
        except (TypeError, ValueError) as exc:
            raise PreparationError(f"帧 {name} 的 {key}={value!r} 不是整数") from exc
        if declared != expected:
            raise PreparationError(
                f"帧 {name} 的 {key}={declared} 与 DETSEC 推得的 {axis} 轴分箱 "
                f"{expected} 不一致"
            )


@dataclass(frozen=True)
class AxisLayout:
    """一个轴上科学像元到平场像元的仿射映射。"""

    flat_min: int  # 该轴覆盖的最小平场像素下标
    flat_count: int  # 覆盖的平场像素总数
    block: int  # 每个科学像元覆盖的平场像元数
    flip: bool  # 科学像元顺序是否与平场下标升序相反
    science_pixels: int  # 该轴科学像元数


def _pixel_interval(first: int, second: int, index: int, binning: int) -> tuple[int, int]:
    """第 index 个像元覆盖的探测器坐标闭区间。"""
    if first <= second:
        low = first + index * binning
        return low, low + binning - 1
    high = first - index * binning
    return high - binning + 1, high


def axis_layout(
    flat: DetectorSection,
    flat_pixels: int,
    science: DetectorSection,
    science_pixels: int,
    axis: str,
    name: str,
) -> AxisLayout:
    """建立科学帧某个轴到平场帧的映射，并校验窗口/分箱对齐。"""
    if axis == "x":
        flat_first, flat_second = flat.x1, flat.x2
        sci_first, sci_second = science.x1, science.x2
        flat_span = flat.x_span
    else:
        flat_first, flat_second = flat.y1, flat.y2
        sci_first, sci_second = science.y1, science.y2
        flat_span = flat.y_span

    flat_bin = binning_from_span(flat_span, flat_pixels, axis, "主平场", flat)
    sci_bin = binning_from_span(
        abs(sci_second - sci_first) + 1, science_pixels, axis, name, science
    )

    if sci_bin < flat_bin or sci_bin % flat_bin:
        raise PreparationError(
            f"科学帧 {axis} 轴分箱 {sci_bin}x 与主平场 {flat_bin}x 不兼容，"
            "科学像元无法完整覆盖平场像元"
        )
    block = sci_bin // flat_bin

    if axis == "x":
        sci_low, sci_high = science.x_low, science.x_high
        flat_low, flat_high = flat.x_low, flat.x_high
    else:
        sci_low, sci_high = science.y_low, science.y_high
        flat_low, flat_high = flat.y_low, flat.y_high
    if sci_low < flat_low or sci_high > flat_high:
        raise PreparationError(
            f"科学窗 {axis}=[{sci_low}:{sci_high}] 超出主平场覆盖范围 "
            f"[{flat_low}:{flat_high}]"
        )

    direction = 1 if flat_first <= flat_second else -1

    def directed(value: int) -> int:
        return (value - flat_first) * direction

    def pixel_flat_index(index: int) -> int:
        low, high = _pixel_interval(sci_first, sci_second, index, sci_bin)
        d_low, d_high = directed(low), directed(high)
        d_min, d_max = min(d_low, d_high), max(d_low, d_high)
        if d_min < 0 or d_max >= flat_span:
            raise PreparationError(
                f"科学窗 {axis} 像元 {index} 超出主平场覆盖范围"
            )
        if d_min % flat_bin or (d_max + 1) % flat_bin:
            raise PreparationError(
                f"科学帧 {axis} 轴窗口未与主平场像元边界对齐，"
                "无法计算完整的平场系数均值"
            )
        return d_min // flat_bin

    first_index = pixel_flat_index(0)
    last_index = pixel_flat_index(science_pixels - 1)
    if science_pixels > 1:
        if abs(last_index - first_index) != (science_pixels - 1) * block:
            raise PreparationError(
                f"科学帧 {axis} 轴窗口与主平场像元步长不一致"
            )
    flat_min = min(first_index, last_index)
    flat_max = max(first_index, last_index)
    # 首像元占 [flat_min, flat_min+block)，末像元占 [flat_max, flat_max+block)。
    if flat_max - flat_min + block != science_pixels * block:
        raise PreparationError(f"科学帧 {axis} 轴窗口与主平场覆盖不对齐")
    if flat_min < 0 or flat_max + block > flat_pixels:
        raise PreparationError(
            f"科学窗 {axis} 轴映射到主平场下标 [{flat_min}:{flat_max}]，"
            f"超出 0~{flat_pixels - 1}"
        )

    return AxisLayout(
        flat_min=flat_min,
        flat_count=science_pixels * block,
        block=block,
        flip=science_pixels > 1 and last_index < first_index,
        science_pixels=science_pixels,
    )


def band_flat_start(
    layout: AxisLayout, index0: int, index1: int, total: int
) -> int:
    """科学帧 [index0, index1) 区间对应的连续平场像素起点。"""
    if not layout.flip:
        return layout.flat_min + index0 * layout.block
    return layout.flat_min + (total - index1) * layout.block
