"""Build the reader model from labelled fixtures, and measure it honestly.

    uv run python -m wwf.vision.train              # build assets/models/reader.npz
    uv run python -m wwf.vision.train --evaluate   # hold out each game position in turn

Only fixtures whose label status is "verified" are used. Photos of the same position (identical
board and rack) are held out together, so evaluation never tests on a board it trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from wwf.core.config import BoardConfig, assets_dir, load_board_config
from wwf.vision.board import BoardFit, find_board
from wwf.vision.pipeline import DEFAULT_MODEL
from wwf.vision.reader import (
    LETTERS,
    ReaderModel,
    board_image,
    cell_crops,
    glyph_features,
    glyph_mask,
    rack_crops,
    rack_letter_area,
)
from wwf.vision.synth import synthetic_tiles

DEFAULT_FIXTURES = assets_dir() / "fixtures"


@dataclass
class Example:
    name: str
    image: np.ndarray
    board: list[str]
    rack: str
    quad: list[list[float]] | None

    @property
    def position(self) -> tuple[str, ...]:
        return (*self.board, self.rack)


def fit_from_quad(quad: list[list[float]]) -> BoardFit:
    src = np.float32([[0, 0], [15, 0], [15, 15], [0, 15]])
    H = cv2.getPerspectiveTransform(src, np.float32(quad)).astype(np.float64)
    pitch = float(np.linalg.norm(np.subtract(quad[1], quad[0]))) / 15
    return BoardFit(H, pitch, 0, 0, 1.0)


def load_examples(directory: Path) -> list[Example]:
    out = []
    for path in sorted(directory.glob("*.json")):
        label = json.loads(path.read_text())
        if label.get("status", "verified") != "verified":
            continue
        image = cv2.imread(str(directory / label["image"]))
        if image is None:
            continue
        out.append(Example(path.stem, image, label["board"], label.get("rack", ""),
                           label.get("board_quad")))  # fmt: skip
    return out


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


def build_model(examples: list[Example], cfg: BoardConfig, augment: int = 6,
                seed: int = 0, synthetic: bool = True) -> ReaderModel:  # fmt: skip
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
    for ex in examples:
        fits = [fit_from_quad(ex.quad)] if ex.quad else []
        found = find_board(ex.image, cfg)
        if found is not None:
            fits.append(found)
        for fit in fits:
            crops = cell_crops(board_image(ex.image, fit))
            for r in range(15):
                for c in range(15):
                    ch = ex.board[r][c]
                    if ch == ".":
                        continue
                    for i in range(augment + 1):
                        crop = crops[r, c] if i == 0 else _jitter(crops[r, c], rng)
                        g = glyph_mask(crop)
                        if g is not None:
                            gx.append(glyph_features(g))
                            gy.append(LETTERS.index(ch.upper()))
            for slot, crop in zip(ex.rack, rack_crops(ex.image, fit), strict=False):
                if slot == "?":
                    continue
                area = rack_letter_area(crop)
                for i in range(augment + 1):
                    g = glyph_mask(area if i == 0 else _jitter(area, rng), rack=True)
                    if g is not None:
                        gx.append(glyph_features(g))
                        gy.append(LETTERS.index(slot.upper()))
    return ReaderModel(glyph_x=np.array(gx, dtype=np.float32), glyph_y=np.array(gy, dtype=np.int64))


def train_from_fixtures(directory: Path, cfg: BoardConfig, out: Path) -> ReaderModel | None:
    """Rebuild and save the model from a fixtures folder. None if nothing is labelled yet."""
    examples = load_examples(directory)
    if not examples:
        return None
    model = build_model(examples, cfg)
    model.save(out)
    return model


def evaluate(examples: list[Example], cfg: BoardConfig, synthetic: bool = True) -> int:
    from wwf.vision.pipeline import read_image

    groups: dict[tuple[str, ...], list[Example]] = defaultdict(list)
    for ex in examples:
        groups[ex.position].append(ex)
    if len(groups) < 2:
        print("need at least two distinct labelled positions to evaluate")
        return 1
    totals = defaultdict(int)
    for held_out, tests in groups.items():
        train = [e for e in examples if e.position != held_out]
        model = build_model(train, cfg, synthetic=synthetic)
        for ex in tests:
            t = time.perf_counter()
            result = read_image(ex.image, model, cfg)
            ms = (time.perf_counter() - t) * 1000
            if result is None:
                print(f"{ex.name}: board not found")
                totals["missed_boards"] += 1
                continue
            wrong, flagged = [], 0
            for r in range(15):
                for c in range(15):
                    truth = ex.board[r][c].upper()
                    got = (result.board[r][c] or ".").upper()
                    totals["cells"] += 1
                    if got != truth:
                        wrong.append(f"({r},{c}) {truth}->{got}")
                        flagged += result.needs_review[r][c]
            totals["cell_errors"] += len(wrong)
            totals["flagged_errors"] += flagged
            rack_truth, rack_got = ex.rack.upper(), "".join(result.rack).upper()
            rack_ok = rack_truth == rack_got
            totals["racks"] += 1
            totals["rack_errors"] += not rack_ok
            review = sum(map(sum, result.needs_review))
            print(f"{ex.name}: {225 - len(wrong)}/225 squares right, {flagged}/{len(wrong)} errors "
                  f"flagged, {review} flagged in all; rack {rack_got!r} "
                  f"{'ok' if rack_ok else f'(truth {rack_truth!r})'}; {ms:.0f} ms")  # fmt: skip
            if wrong:
                print("   wrong:", ", ".join(wrong[:12]) + (" ..." if len(wrong) > 12 else ""))
    acc = 1 - totals["cell_errors"] / max(1, totals["cells"])
    print(f"\nheld-out square accuracy {acc:.2%} ({totals['cell_errors']} errors in "
          f"{totals['cells']}), {totals['flagged_errors']} of the errors flagged for review; "
          f"racks right {totals['racks'] - totals['rack_errors']}/{totals['racks']}")  # fmt: skip
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--evaluate", action="store_true", help="leave-one-position-out test")
    parser.add_argument("--no-synthetic", action="store_true", help="skip font-rendered letters")
    args = parser.parse_args()
    cfg = load_board_config()
    examples = load_examples(args.fixtures)
    print(f"{len(examples)} verified fixtures, "
          f"{len({e.position for e in examples})} distinct positions")  # fmt: skip
    if args.evaluate:
        return evaluate(examples, cfg, synthetic=not args.no_synthetic)
    if not examples:
        print("no verified fixtures to learn from")
        return 1
    model = build_model(examples, cfg, synthetic=not args.no_synthetic)
    model.save(args.out)
    print(f"saved {args.out}: {len(model.glyph_y)} letter samples, "
          f"letters seen: {''.join(sorted({LETTERS[i] for i in model.glyph_y}))}")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
