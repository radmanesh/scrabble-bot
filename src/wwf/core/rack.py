"""Player rack."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

BLANK = "?"
MAX_RACK = 7


@dataclass(frozen=True)
class Rack:
    tiles: tuple[str, ...]  # letters, "?" for blank, length 0-7

    def __post_init__(self) -> None:
        if len(self.tiles) > MAX_RACK:
            raise ValueError(f"rack has {len(self.tiles)} tiles, max is {MAX_RACK}")
        for t in self.tiles:
            if t != BLANK and not (len(t) == 1 and "A" <= t <= "Z"):
                raise ValueError(f"invalid rack tile {t!r}")

    @classmethod
    def from_string(cls, s: str) -> Rack:
        return cls(tuple(ch if ch == BLANK else ch.upper() for ch in s if not ch.isspace()))

    def counts(self) -> Counter[str]:
        return Counter(self.tiles)

    def __str__(self) -> str:
        return "".join(self.tiles)
