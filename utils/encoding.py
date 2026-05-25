"""
Encoding utilities for Project Oracle.
Project Oracle 编码工具模块。

Two responsibilities / 两个职责:
1. Fix Windows console encoding so player names with diacritics (ñ, é, ü,
   Arabic, CJK, etc.) print correctly instead of crashing with UnicodeEncodeError.
   修复 Windows 终端编码，使带变音符/阿拉伯文/CJK 的球员名字能正确打印。

2. Provide a deterministic ASCII transliteration function for cross-source
   entity matching. Different data providers use different Unicode conventions
   for the same player name; normalizing to ASCII before fuzzy matching
   dramatically improves recall.
   提供确定性的 ASCII 转写函数，用于跨数据源的实体匹配。
   不同数据源对同一球员名字的 Unicode 表示不同，转 ASCII 后再做模糊匹配
   能大幅提高匹配召回率。

Usage / 用法:
    from utils.encoding import fix_console_encoding, to_ascii_name
    fix_console_encoding()          # call once at script startup
    to_ascii_name("Lionel Andrés Messi")  # -> "Lionel Andres Messi"
"""

from __future__ import annotations

import os
import sys
import unicodedata

from unidecode import unidecode


def fix_console_encoding() -> None:
    """
    Reconfigure stdout/stderr to UTF-8 on Windows.
    在 Windows 上将 stdout/stderr 重新配置为 UTF-8。

    Why? 为什么需要这个？
    Windows PowerShell / cmd 默认使用 GBK (cp936) 或其他 ANSI 编码，
    遇到西班牙语变音符 (ñ)、阿拉伯文、克罗地亚文 (č, ž) 等字符时
    会抛出 UnicodeEncodeError 导致整个脚本崩溃。
    这个函数在脚本启动时调用一次即可全局修复。
    """
    if sys.platform != "win32":
        return

    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def to_ascii_name(name: str) -> str:
    """
    Transliterate a Unicode player name to pure ASCII.
    将 Unicode 球员名字转写为纯 ASCII。

    Process / 处理流程:
    1. Unicode NFC normalization — merge combining characters.
       Unicode NFC 归一化——合并组合字符（如 e + ◌́ → é）。
    2. unidecode transliteration — é→e, ñ→n, ü→u, 遠藤→Yuan Teng, etc.
       unidecode 转写——把变音符、CJK 等映射为最接近的 ASCII。
    3. Collapse multiple spaces and strip.
       合并多余空格并去除首尾空白。

    Why NFC first? 为什么先做 NFC？
    有些数据源用"分解形式"(NFD)存储字符，例如 é 被拆成 e + combining acute。
    如果不先合并，unidecode 可能会丢掉变音符而不是转写它。

    Examples / 示例:
        "Lionel Andrés Messi Cuccittini" -> "Lionel Andres Messi Cuccittini"
        "Kylian Mbappé Lottin"           -> "Kylian Mbappe Lottin"
        "Wojciech Szczęsny"              -> "Wojciech Szczesny"
        "遠藤航"                          -> "Yuan Teng Hang"
        "محمد صلاح"                       -> "mhmd slah"

    Parameters
    ----------
    name : str
        Raw player name in any script/language.
        任何语言/文字的原始球员名字。

    Returns
    -------
    str
        ASCII-only transliteration, suitable for fuzzy matching.
        纯 ASCII 转写结果，适合用于模糊匹配。
    """
    normalized = unicodedata.normalize("NFC", name)
    ascii_name = unidecode(normalized)
    return " ".join(ascii_name.split()).strip()
