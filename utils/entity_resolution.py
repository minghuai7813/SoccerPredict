"""
Player entity resolution via fuzzy name matching (ASCII-first strategy).
球员实体对齐模块——基于 ASCII 优先策略的模糊匹配。

Why ASCII-first? 为什么 ASCII 优先？
同一个球员在不同数据源中的名字 Unicode 表示往往不一致：
  - StatsBomb: "Lionel Andrés Messi Cuccittini"  (带变音符 + 完整法定名)
  - FBref:     "Lionel Messi"                    (ASCII 化 + 常用短名)
  - FotMob:    "L. Messi"                        (简写)
先统一转写为 ASCII 再比对，消除变音符差异。

Matching algorithm / 匹配算法:
1. Convert target name to ASCII.
2. Compare ASCII target against all ASCII names in the roster.
3. Use thefuzz.fuzz.token_set_ratio (handles subset matching).
   Why token_set_ratio instead of token_sort_ratio?
   为什么用 token_set_ratio 而不是 token_sort_ratio？
   StatsBomb 用完整法定名 (4-5 words)，FBref 用常用名 (2-3 words)。
   token_sort_ratio 会因为多余单词大幅降分:
     token_sort_ratio("Lionel Andres Messi Cuccittini", "Lionel Messi") ≈ 57
   token_set_ratio 先提取公共 token 交集再比较，子集匹配分数很高:
     token_set_ratio("Lionel Andres Messi Cuccittini", "Lionel Messi") ≈ 100
4. If best score >= threshold → same player; else → new player.

Dependencies: thefuzz, unidecode (via utils.encoding)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from thefuzz import fuzz

from utils.encoding import to_ascii_name


@dataclass
class MatchResult:
    """
    Container for a single fuzzy match result.
    单次模糊匹配结果的容器。

    Attributes
    ----------
    matched_name : str | None
        The original (Unicode) name from the roster that was matched.
        匹配到的名单中的原始名字（Unicode 版）。None if this is a new player.
    matched_name_ascii : str | None
        The ASCII version of the matched name (for debugging / logging).
        匹配到的名字的 ASCII 版本（用于调试/日志）。
    score : int
        Fuzzy match score (0-100). 0 means no roster to compare against.
        模糊匹配分数。0 表示名单为空无法比对。
    internal_player_id : str
        UUID of the matched or newly created player.
        匹配到的或新建的球员 UUID。
    is_new : bool
        True if no satisfactory match was found and a new UUID was generated.
        True 表示没有找到满意的匹配，已生成新 UUID。
    """
    matched_name: str | None
    matched_name_ascii: str | None
    score: int
    internal_player_id: str
    is_new: bool


class PlayerMatcher:
    """
    Fuzzy name matcher that resolves external player names against an
    existing roster using ASCII-normalized comparison.
    基于 ASCII 归一化比较的模糊名字匹配器。

    Internal data structure / 内部数据结构:
    _roster stores three parallel mappings keyed by the same set of players:
      - _name_to_id:       original_name -> internal_player_id
      - _ascii_to_original: ascii_name -> original_name
      - _ascii_names:       list of all ASCII names (for iteration)
    This lets us match on ASCII but return the original Unicode name + UUID.
    _roster 存储三组并行映射，通过 ASCII 名比对但返回原始 Unicode 名 + UUID。
    """

    def __init__(self, existing_players: dict[str, str] | None = None):
        """
        Initialize with an optional mapping of known players.
        用已知球员映射表初始化。

        Parameters
        ----------
        existing_players : dict[str, str] | None
            Mapping of full_name (Unicode) -> internal_player_id.
            {原始名字: UUID} 映射。
        """
        self._name_to_id: dict[str, str] = {}
        self._ascii_to_original: dict[str, str] = {}

        if existing_players:
            self.bulk_register(existing_players)

    @property
    def roster_size(self) -> int:
        """Return current number of known players. / 返回当前已知球员数量。"""
        return len(self._name_to_id)

    def _register_one(self, original_name: str, player_id: str) -> None:
        """
        Register a single player into all internal indexes.
        将一个球员注册到所有内部索引中。
        """
        ascii_name = to_ascii_name(original_name)
        self._name_to_id[original_name] = player_id
        self._ascii_to_original[ascii_name] = original_name

    def match_name(
        self,
        target_name: str,
        threshold: int = 80,
    ) -> MatchResult:
        """
        Attempt to match *target_name* against the internal roster.
        尝试将目标名字与内部名单做模糊匹配（ASCII 优先）。

        Algorithm / 算法流程:
        1. Convert target_name to ASCII.
           将目标名字转为 ASCII。
        2. If the roster is empty, generate a new UUID immediately.
           如果名单为空，直接生成新 UUID。
        3. Compute token_sort_ratio(target_ascii, each_roster_ascii).
           对名单中每个 ASCII 名计算 token_sort_ratio。
        4. If best score >= threshold, return the matched player's UUID.
           如果最高分 >= 阈值，返回匹配到的球员 UUID。
        5. Otherwise, generate a new UUID and register into the roster.
           否则生成新 UUID 并注册到名单中。

        Parameters
        ----------
        target_name : str
            Raw player name from an external data source (any Unicode).
            外部数据源的原始球员名字（任何 Unicode）。
        threshold : int
            Minimum fuzzy score (0-100) to accept a match. Default 80.
            接受匹配的最低分数，默认 80。

        Returns
        -------
        MatchResult
        """
        target_ascii = to_ascii_name(target_name)

        if not self._ascii_to_original:
            new_id = str(uuid.uuid4())
            self._register_one(target_name, new_id)
            return MatchResult(
                matched_name=None,
                matched_name_ascii=None,
                score=0,
                internal_player_id=new_id,
                is_new=True,
            )

        # Brute-force scan over ASCII names.
        # 暴力遍历 ASCII 名单（世界杯量级 ~800 人，完全可承受）。
        best_ascii: str | None = None
        best_score: int = 0

        for roster_ascii in self._ascii_to_original:
            score = fuzz.token_set_ratio(target_ascii, roster_ascii)
            if score > best_score:
                best_score = score
                best_ascii = roster_ascii

        if best_score >= threshold and best_ascii is not None:
            original_name = self._ascii_to_original[best_ascii]
            return MatchResult(
                matched_name=original_name,
                matched_name_ascii=best_ascii,
                score=best_score,
                internal_player_id=self._name_to_id[original_name],
                is_new=False,
            )

        # No satisfactory match — create new player entry.
        # 没有达到阈值——新建球员条目。
        new_id = str(uuid.uuid4())
        self._register_one(target_name, new_id)
        return MatchResult(
            matched_name=None,
            matched_name_ascii=None,
            score=best_score,
            internal_player_id=new_id,
            is_new=True,
        )

    def bulk_register(self, players: dict[str, str]) -> None:
        """
        Directly register a batch of players without matching.
        批量注册球员（跳过匹配），通常用于从数据库预加载。

        Parameters
        ----------
        players : dict[str, str]
            Mapping of full_name (Unicode) -> internal_player_id.
        """
        for name, pid in players.items():
            self._register_one(name, pid)
