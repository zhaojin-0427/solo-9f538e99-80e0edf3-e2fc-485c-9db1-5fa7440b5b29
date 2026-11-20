"""共用错误类型与头部文本清理。"""

from __future__ import annotations


class PreparationError(Exception):
    """整批制备/校正失败；status_code 决定 HTTP 响应码。"""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def sanitize_ascii(text: str, limit: int = 60) -> str:
    """把任意文件名字符串压成 FITS 头部可安全承载的 ASCII。"""
    cleaned = "".join(ch if 32 <= ord(ch) < 127 else "?" for ch in text).strip()
    return cleaned[:limit] or "unknown"
