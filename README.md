# WWF Solver

Takes an image of a Words With Friends game screen, reconstructs the board and rack, and
returns the highest-scoring legal moves. Built milestone by milestone from the build spec.

## Development

```sh
uv sync                 # create .venv and install everything (add --extra train for PyTorch)
uv run pytest           # fast tests; add -m "" to include tests marked slow
uv run ruff check . && uv run ruff format --check .
uv run mypy             # strict on wwf.core and wwf.solvers
uv run lint-imports     # enforces core <- {vision, solvers} <- api layering
```

Config lives in `config/` (board layout, letter values, solver registry); the default word list is
`assets/dictionaries/enable1.txt`. The trie is cached in `.cache/` keyed by the word list's hash.
Set `WWF_ROOT` to point at a different directory holding `config/` and `assets/`.

**Dictionary caveat.** The WWF dictionary is proprietary. ENABLE1 is the public default and will
occasionally accept a word WWF rejects, or the reverse (ENABLE1 has no `QI` or `ZA`, for
example). Any newline-separated word list can be swapped in.

## Milestone notes

**M1, scaffold and core.** `wwf.core` holds the immutable domain model: `Board`, `Tile`,
`Bonus`, `Rack`, `Move`/`Placement` (with `build_move` to derive direction, main word and start
from a set of placements), `score_move` and `validate_move` (named violations), the trie
dictionary with its pickle cache, and the validated `BoardConfig` loader for
`config/wwf_classic.json`. Tests cover the worked scoring examples (HELLO first move, a blank on a
triple word, a 7-tile bingo, parallel plays, stacked word bonuses), every named violation, a
Hypothesis transpose-invariance property for scoring, and config rejection of malformed layouts.
`validate_move` takes an optional `rack` argument beyond the spec signature, needed for
`tile_not_in_rack`.
