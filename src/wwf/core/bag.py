"""Tiles that are not on the board or on your rack: the opponent's rack plus the bag."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from wwf.core.board import Board
from wwf.core.config import BoardConfig
from wwf.core.rack import BLANK, Rack


@dataclass(frozen=True)
class BagReport:
    unseen: dict[str, int]  # letter ("?" for a blank) -> tiles not on the board or your rack
    over: dict[str, int]  # letters shown more often than the bag holds: probably a hidden blank
    on_board: int
    unseen_total: int  # the opponent's rack plus the bag
    in_bag: int  # unseen_total minus a full opponent rack, never below zero


def bag_report(cfg: BoardConfig, board: Board, rack: Rack, opponent_rack: int = 7) -> BagReport:
    used: Counter[str] = Counter()
    on_board = 0
    for row in board.cells:
        for tile in row:
            if tile is not None:
                used[BLANK if tile.is_blank else tile.letter] += 1
                on_board += 1
    used.update(rack.tiles)

    unseen: dict[str, int] = {}
    over: dict[str, int] = {}
    for letter, total in sorted(cfg.bag.items()):
        left = total - used[letter]
        if left < 0:
            over[letter] = -left
            left = 0
        if left:
            unseen[letter] = left
    total_tiles = sum(cfg.bag.values())
    unseen_total = max(0, total_tiles - on_board - len(rack.tiles))
    return BagReport(
        unseen=unseen,
        over=over,
        on_board=on_board,
        unseen_total=unseen_total,
        in_bag=max(0, unseen_total - opponent_rack),
    )
