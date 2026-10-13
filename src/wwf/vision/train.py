"""Build the reader model from labelled fixtures, and measure it honestly.

    uv run python -m wwf.vision.train              # build assets/models/reader.npz from
                                                   # assets/archive/ and data/games/
    uv run python -m wwf.vision.train --evaluate   # hold out each game position in turn
    uv run python -m wwf.vision.train --check      # list labels the pictures contradict

Two sources: `assets/archive/` (pictures labelled before games kept their own pictures: an image
plus a JSON label each) and `data/games/` (every picture of a saved game whose turn was
confirmed carries the confirmed board as its label). In the archive, pictures are used when their
label is "verified", or when `same_position.json` says they show the position of a verified
picture. Photos of the same position (identical board) are held out
together, so evaluation never tests on a board it trained on.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wwf.core.config import BoardConfig, assets_dir, load_board_config, project_root
from wwf.vision.board import BoardFit, find_boards, visible_mask
from wwf.vision.occupancy import Measures, OccupancyModel, measure
from wwf.vision.pipeline import DEFAULT_MODEL
from wwf.vision.reader import (
    LETTERS,
    ReaderModel,
    board_image,
    centred_crop,
    glyph_features,
    glyph_mask,
    pad_board,
    rack_crops,
    rack_letter_area,
)
from wwf.vision.synth import synthetic_tiles

DEFAULT_FIXTURES = assets_dir() / "archive"
ALIASES_FILE = "same_position.json"
# A square whose label the picture contradicts this strongly (the model gives the labelled state
# under this probability) is reported as a label problem and left out of training and scoring.
CONTRADICTION = 0.1
CLEANING_ROUNDS = 3


@dataclass
class Example:
    name: str
    image: np.ndarray
    board: list[str]
    rack: str | None  # None when the picture's own rack was never checked
    quad: list[list[float]] | None

    @property
    def position(self) -> tuple[str, ...]:
        return (*self.board,)


@dataclass(frozen=True)
class LabelIssue:
    name: str
    row: int
    col: int
    label: str  # the letter in the label, or "." for an empty square
    says: str  # what the picture shows: "empty" or "a tile"


def fit_from_quad(quad: list[list[float]]) -> BoardFit:
    src = np.float32([[0, 0], [15, 0], [15, 15], [0, 15]])
    H = cv2.getPerspectiveTransform(src, np.float32(quad)).astype(np.float64)
    pitch = float(np.linalg.norm(np.subtract(quad[1], quad[0]))) / 15
    return BoardFit(H, pitch, 0, 0, 1.0)


def load_examples(directory: Path) -> list[Example]:
    """Labelled pictures: verified ones, plus those that show another picture's position."""
    aliases: dict[str, str] = {}
    alias_file = directory / ALIASES_FILE
    if alias_file.exists():
        aliases = json.loads(alias_file.read_text()).get("same_position_as", {})
    labels: dict[str, dict[str, object]] = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == ALIASES_FILE:
            continue
        labels[path.stem] = json.loads(path.read_text())

    out = []
    for name, label in labels.items():
        verified = label.get("status", "verified") == "verified"
        truth = labels.get(aliases.get(name, ""))
        if truth is not None and truth.get("status", "verified") == "verified":
            board = truth["board"]
        elif verified:
            board = label["board"]
        else:
            continue
        image = cv2.imread(str(directory / str(label["image"])))
        if image is None:
            continue
        rack = str(label.get("rack", "")) if verified else None
        quad = label.get("board_quad")
        out.append(
            Example(name, image, list(board), rack, quad if isinstance(quad, list) else None)  # type: ignore[arg-type]
        )
    return out


def load_game_examples(games_dir: Path) -> list[Example]:
    """Pictures of saved games whose turn was confirmed: the confirmed board is the label."""
    out: list[Example] = []
    for path in sorted(games_dir.glob("*.json")):
        try:
            game = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        for pic in game.get("pictures", []):
            board = pic.get("label_board")
            if not board:
                continue
            image = cv2.imread(str(games_dir / str(game.get("id", path.stem)) / str(pic["file"])))
            if image is None:
                continue
            quad = pic.get("quad")
            out.append(
                Example(
                    f"{path.stem[:8]}-{pic['id']}",
                    image,
                    list(board),
                    pic.get("label_rack"),
                    quad if isinstance(quad, list) else None,
                )
            )
    return out


def load_all_examples(archive: Path | None, games_dir: Path | None) -> list[Example]:
    """The archive's labelled pictures plus those kept with saved games."""
    examples: list[Example] = []
    if archive is not None and archive.is_dir():
        examples += load_examples(archive)
    if games_dir is not None and games_dir.is_dir():
        examples += load_game_examples(games_dir)
    return examples


def _jitter(crop: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = crop.shape[:2]
    angle = rng.uniform(-5, 5)
    scale = rng.uniform(0.9, 1.1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[:, 2] += rng.uniform(-0.06, 0.06, 2) * (w, h)
    out = cv2.warpAffine(crop, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    gain, bias = rng.uniform(0.8, 1.2), rng.uniform(-20, 20)
    out = np.clip(out.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 1.0))
    return out


def _picture_fit(ex: Example, cfg: BoardConfig) -> BoardFit | None:
    fits = find_boards(ex.image, cfg)
    if fits:
        return fits[0]
    return fit_from_quad(ex.quad) if ex.quad else None


def _measured(ex: Example, cfg: BoardConfig) -> tuple[BoardFit, np.ndarray, Measures] | None:
    fit = _picture_fit(ex, cfg)
    if fit is None:
        return None
    board = board_image(ex.image, fit)
    return fit, board, measure(board, cfg, visible_mask(ex.image.shape, fit.homography))


def _truth(ex: Example) -> np.ndarray:
    return np.array([[ch != "." for ch in row] for row in ex.board])


def build_model(
    examples: list[Example],
    cfg: BoardConfig,
    augment: int = 6,
    seed: int = 0,
    synthetic: bool = True,
) -> ReaderModel:
    model, _ = build_model_and_issues(examples, cfg, augment, seed, synthetic)
    return model


def build_model_and_issues(
    examples: list[Example],
    cfg: BoardConfig,
    augment: int = 6,
    seed: int = 0,
    synthetic: bool = True,
) -> tuple[ReaderModel, list[LabelIssue]]:
    rng = np.random.default_rng(seed)
    gx: list[np.ndarray] = []
    gy: list[int] = []
    if synthetic:
        for crop, letter in synthetic_tiles():
            for i in range(3):
                g = glyph_mask(crop if i == 0 else _jitter(crop, rng))
                if g is not None:
                    gx.append(glyph_features(g))
                    gy.append(LETTERS.index(letter))

    measured = [(ex, _measured(ex, cfg)) for ex in examples]
    rows = [(ex, mm) for ex, mm in measured if mm is not None]

    # squares: fit, drop the labels the pictures contradict, fit again. A label with tiles the
    # picture doesn't show pulls the first fit towards calling them empty, so the dropping is
    # repeated until it settles (a single round left about half of them in).
    occupancy = OccupancyModel.default()
    issues: list[LabelIssue] = []
    if rows:
        pairs = [(m, _truth(ex)) for ex, (_, _, m) in rows]
        occupancy = OccupancyModel.train(pairs)
        for _ in range(CLEANING_ROUNDS):
            issues = []
            cleaned = []
            for (ex, (_, _, m)), (_, truth) in zip(rows, pairs, strict=True):
                p = occupancy.probability(m)
                bad = m.inside & (
                    (truth & (p < CONTRADICTION)) | (~truth & (p > 1 - CONTRADICTION))
                )
                for r, c in zip(*np.nonzero(bad), strict=True):
                    says = "empty" if truth[r, c] else "a tile"
                    issues.append(LabelIssue(ex.name, int(r), int(c), ex.board[r][c], says))
                cleaned.append((dataclasses.replace(m, inside=m.inside & ~bad), truth))
            occupancy = OccupancyModel.train(cleaned)

    bad_squares = {(i.name, i.row, i.col) for i in issues}
    for ex, (fit, board, m) in rows:
        padded = pad_board(board)
        likely = occupancy.probability(m) >= 0.5
        for r in range(15):
            for c in range(15):
                ch = ex.board[r][c]
                if ch == "." or not likely[r, c] or (ex.name, r, c) in bad_squares:
                    continue  # only squares that are labelled a tile AND look like one
                base = centred_crop(padded, r, c)
                for i in range(augment + 1):
                    crop = base if i == 0 else _jitter(base, rng)
                    g = glyph_mask(crop)
                    if g is not None:
                        gx.append(glyph_features(g))
                        gy.append(LETTERS.index(ch.upper()))
        if ex.rack is None:
            continue
        for slot, crop in zip(ex.rack, rack_crops(ex.image, fit), strict=False):
            if slot == "?":
                continue
            area = rack_letter_area(crop)
            for i in range(augment + 1):
                g = glyph_mask(area if i == 0 else _jitter(area, rng), rack=True)
                if g is not None:
                    gx.append(glyph_features(g))
                    gy.append(LETTERS.index(slot.upper()))
    model = ReaderModel(
        glyph_x=np.array(gx, dtype=np.float32),
        glyph_y=np.array(gy, dtype=np.int64),
        occupancy=occupancy,
    )
    return model, issues


def train_from_sources(
    archive: Path | None, games_dir: Path | None, cfg: BoardConfig, out: Path
) -> tuple[ReaderModel, list[LabelIssue]] | None:
    """Rebuild and save the model from the archive and the games' pictures, with the label
    problems found. None if nothing is labelled yet."""
    examples = load_all_examples(archive, games_dir)
    if not examples:
        return None
    model, issues = build_model_and_issues(examples, cfg)
    model.save(out)
    return model, issues


def picture_issues(
    model: ReaderModel, image: np.ndarray, board: list[list[str | None]], cfg: BoardConfig
) -> list[tuple[int, int, str]]:
    """Squares of `board` that the picture contradicts: (row, col, "empty" | "a tile")."""
    fits = find_boards(image, cfg)
    if not fits:
        return []
    fit = fits[0]
    m = measure(board_image(image, fit), cfg, visible_mask(image.shape, fit.homography))
    p = model.occupancy.probability(m)
    out = []
    for r in range(15):
        for c in range(15):
            if not m.inside[r, c]:
                continue
            labelled = bool(board[r][c])
            if labelled and p[r, c] < CONTRADICTION:
                out.append((r, c, "empty"))
            elif not labelled and p[r, c] > 1 - CONTRADICTION:
                out.append((r, c, "a tile"))
    return out


def label_issues(
    directory: Path, cfg: BoardConfig, games_dir: Path | None = None
) -> list[LabelIssue]:
    """Squares where a label and its picture strongly disagree (usually a typing slip)."""
    examples = load_all_examples(directory, games_dir)
    if not examples:
        return []
    return build_model_and_issues(examples, cfg, augment=0, synthetic=False)[1]


# --- evaluation ----------------------------------------------------------------------------------


def _score(
    ex: Example, result_board: list[list[str | None]], ignore: set[tuple[int, int]]
) -> tuple[int, int, int, list[str]]:
    """(errors, tiles invented, tiles missed, descriptions) against the label."""
    wrong: list[str] = []
    invented = missed = 0
    for r in range(15):
        for c in range(15):
            if (r, c) in ignore:
                continue
            truth = ex.board[r][c].upper()
            got = (result_board[r][c] or ".").upper()
            if got == truth:
                continue
            wrong.append(f"({r},{c}) {truth}->{got}")
            invented += truth == "."
            missed += got == "."
    return len(wrong), invented, missed, wrong


def evaluate(examples: list[Example], cfg: BoardConfig, synthetic: bool = True) -> int:
    from wwf.vision.pipeline import read_image

    groups: dict[tuple[str, ...], list[Example]] = defaultdict(list)
    for ex in examples:
        groups[ex.position].append(ex)
    if len(groups) < 2:
        print("need at least two distinct labelled positions to evaluate")
        return 1
    _, issues = build_model_and_issues(examples, cfg, augment=0, synthetic=False)
    ignore: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for i in issues:
        ignore[i.name].add((i.row, i.col))
    rng = np.random.default_rng(1)

    totals: dict[str, int] = defaultdict(int)
    print(f"{len(issues)} squares skipped because the picture contradicts their label")
    print(f"\n{'picture':30} | first picture: wrong (invented/missed) | next turn: wrong")
    for held_out, tests in groups.items():
        train = [e for e in examples if e.position != held_out]
        model = build_model(train, cfg, synthetic=synthetic)
        for ex in tests:
            t = time.perf_counter()
            result = read_image(ex.image, model, cfg)
            ms = (time.perf_counter() - t) * 1000
            if result is None:
                print(f"{ex.name:30} | board not found")
                totals["missed_boards"] += 1
                continue
            errors, inv, mis, wrong = _score(ex, result.board, ignore[ex.name])
            totals["first_cells"] += 225 - len(ignore[ex.name])
            totals["first_errors"] += errors
            totals["first_invented"] += inv
            totals["first_missed"] += mis

            # next turn: the same picture read against its own label with a few tiles removed
            tiles = [(r, c) for r in range(15) for c in range(15) if ex.board[r][c] != "."]
            nxt = ""
            if len(tiles) >= 8:
                k = int(rng.integers(2, 7))
                gone = [tiles[i] for i in rng.choice(len(tiles), size=k, replace=False)]
                previous: list[list[str | None]] = [
                    [
                        None if ex.board[r][c] == "." or (r, c) in gone else ex.board[r][c].upper()
                        for c in range(15)
                    ]
                    for r in range(15)
                ]
                nr = read_image(ex.image, model, cfg, previous=previous)
                if nr is not None:
                    e2, i2, m2, _ = _score(ex, nr.board, ignore[ex.name])
                    totals["next_turns"] += 1
                    totals["next_errors"] += e2
                    totals["next_invented"] += i2
                    totals["next_missed"] += m2
                    totals["next_rejected"] += nr.rejected
                    nxt = f"{e2:3d} ({k} new)  match {nr.match if nr.match is not None else 0:.2f}"
            print(f"{ex.name:30} | {errors:3d} ({inv}/{mis}) {ms:5.0f} ms          | {nxt}")
            if wrong:
                print("      wrong:", ", ".join(wrong[:8]) + (" ..." if len(wrong) > 8 else ""))
    cells = max(1, totals["first_cells"])
    print(
        f"\nfirst picture: {1 - totals['first_errors'] / cells:.2%} of squares right "
        f"({totals['first_errors']} errors: {totals['first_invented']} tiles invented, "
        f"{totals['first_missed']} missed); boards not found: {totals['missed_boards']}"
    )
    print(
        f"next turn:     {totals['next_errors']} errors over {totals['next_turns']} pictures "
        f"({totals['next_invented']} invented, {totals['next_missed']} missed, "
        f"{totals['next_rejected']} pictures refused)"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES, help="the archive")
    parser.add_argument(
        "--games", type=Path, default=project_root() / "data" / "games", help="saved games"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evaluate", action="store_true", help="leave-one-position-out test")
    parser.add_argument("--check", action="store_true", help="list labels the pictures contradict")
    parser.add_argument("--no-synthetic", action="store_true", help="skip font-rendered letters")
    args = parser.parse_args()
    cfg = load_board_config()
    examples = load_all_examples(args.fixtures, args.games)
    print(
        f"{len(examples)} labelled pictures, "
        f"{len({e.position for e in examples})} distinct positions"
    )
    if args.check:
        issues = label_issues(args.fixtures, cfg, args.games)
        for i in issues:
            what = "empty" if i.label == "." else repr(i.label)
            print(
                f"  {i.name} row {i.row + 1}, column {i.col + 1}: "
                f"label says {what}, the picture shows {i.says}"
            )
        print(f"{len(issues)} contradicted squares")
        return 0
    if args.evaluate:
        return evaluate(examples, cfg, synthetic=not args.no_synthetic)
    if not examples:
        print("no labelled pictures to learn from")
        return 1
    model, issues = build_model_and_issues(examples, cfg, synthetic=not args.no_synthetic)
    model.save(args.out)
    print(
        f"saved {args.out}: {len(model.glyph_y)} letter samples, "
        f"letters seen: {''.join(sorted({LETTERS[i] for i in model.glyph_y}))}"
    )
    if issues:
        print(
            f"{len(issues)} labelled squares contradict their pictures and were left out "
            f"(run with --check to list them)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
