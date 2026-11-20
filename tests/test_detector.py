"""DETSEC 几何解析与映射测试。"""

from __future__ import annotations

import pytest

from app.detector import (
    DetectorSection,
    axis_layout,
    band_flat_start,
    binning_from_span,
    parse_detsec,
)
from app.errors import PreparationError


def test_parse_forward_section():
    section = parse_detsec("[11:110,21:120]", "f")
    assert section == DetectorSection(11, 110, 21, 120)
    assert section.x_span == 100 and section.y_span == 100
    assert not section.x_reversed and not section.y_reversed


def test_parse_reversed_section_and_prefix():
    section = parse_detsec("1:[110:11,120:21]", "f")
    assert section.x_span == 100
    assert section.x_reversed and section.y_reversed
    assert section.text() == "[110:11,120:21]"


@pytest.mark.parametrize(
    "text",
    ["", "11:110", "[1:2,3]", "[1:2,3:4,5:6]", "abc", "[1.5:2,3:4]", None],
)
def test_parse_rejects_malformed(text):
    with pytest.raises(PreparationError):
        parse_detsec(text, "f")


def test_binning_from_span():
    section = parse_detsec("[1:200,1:100]", "f")
    assert binning_from_span(200, 200, "x", "f", section) == 1
    assert binning_from_span(200, 100, "x", "f", section) == 2
    with pytest.raises(PreparationError, match="仅支持"):
        binning_from_span(200, 50, "x", "f", section)
    with pytest.raises(PreparationError, match="不整除"):
        binning_from_span(201, 100, "x", "f", section)


def test_layout_same_binning_identity():
    section = DetectorSection(1, 100, 1, 100)
    layout = axis_layout(section, 100, section, 100, "x", "sci")
    assert layout.flat_min == 0
    assert layout.block == 1
    assert not layout.flip
    assert band_flat_start(layout, 10, 20, 100) == 10


def test_layout_reversed_science_flips_band():
    flat = DetectorSection(1, 100, 1, 100)
    science = DetectorSection(100, 1, 1, 100)
    layout = axis_layout(flat, 100, science, 100, "x", "sci")
    assert layout.flip
    # 科学行带 [0, 10) 对应平场最后 10 个像元（下标 90..99）。
    assert band_flat_start(layout, 0, 10, 100) == 90
    assert band_flat_start(layout, 90, 100, 100) == 0


def test_layout_reversed_flat_axis():
    # 平场自身沿 x 反向存储：DETSEC x 从高到低。
    flat = DetectorSection(100, 1, 1, 100)
    science = DetectorSection(1, 100, 1, 100)
    layout = axis_layout(flat, 100, science, 100, "x", "sci")
    # 科学 x=1 对应平场 DETSEC x=1，即平场数组下标 99。
    assert layout.flip
    assert band_flat_start(layout, 0, 1, 100) == 99
    assert layout.flat_min == 0


def test_layout_2x2_binning_science_over_1x1_flat():
    flat = DetectorSection(1, 100, 1, 100)
    science = DetectorSection(21, 80, 41, 100)  # 60 个探测器像元 -> 30 像素
    layout = axis_layout(flat, 100, science, 30, "x", "sci")
    assert layout.block == 2
    assert layout.flat_min == 20
    assert layout.flat_count == 60
    assert band_flat_start(layout, 3, 5, 30) == 20 + 6


def test_layout_science_window_outside_flat_rejected():
    flat = DetectorSection(11, 60, 1, 100)
    science = DetectorSection(1, 100, 1, 100)
    with pytest.raises(PreparationError, match="超出主平场覆盖范围"):
        axis_layout(flat, 50, science, 100, "x", "sci")


def test_layout_misaligned_window_rejected():
    # 主平场与科学帧都是 2x2：科学像元边界必须与平场像元边界（奇数起点）对齐。
    flat = DetectorSection(1, 100, 1, 100)
    science = DetectorSection(12, 51, 1, 100)  # 起点 12、跨度 40 -> 20 像素
    with pytest.raises(PreparationError, match="对齐"):
        axis_layout(flat, 50, science, 20, "x", "sci")


def test_layout_2x2_science_over_2x2_flat():
    flat = DetectorSection(1, 100, 1, 100)
    science = DetectorSection(1, 100, 1, 100)
    layout = axis_layout(flat, 50, science, 50, "x", "sci")
    assert layout.block == 1 and layout.flat_count == 50


def test_layout_rejects_unsupported_science_binning():
    flat = DetectorSection(1, 120, 1, 100)
    science = DetectorSection(1, 120, 1, 100)
    with pytest.raises(PreparationError, match="仅支持"):
        axis_layout(flat, 40, science, 40, "x", "sci")
