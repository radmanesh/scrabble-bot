# WWF Solver: usage guide

This guide covers everything you can run and use today: the web app, the HTTP API, the Python
library (board and move model, scoring and rule checks, solvers, dictionaries), configuration,
tests, and how to extend it.

## What works today

The project is built in milestones (see the build spec).

| Area | Status |
| --- | --- |
| Core model: board, rack, moves, scoring, rule validation | Done (M1) |
| Dictionary loading, caching and choosing between word lists | Done |
| Solvers: `anchor_trie` ("Smart brute force") and `dummy_random` (demo) | Done (M2) |
| HTTP API (`wwf.api`) | Done (M5) |
| Web pages: upload, review, results, options | Done (M6), not yet tried on a real phone |
| Image recognition (`wwf.vision`) | Not built (M3-4; waiting on screenshot fixtures) |
| Docker image | Not built (M7) |

Milestones 5 and 6 were built ahead of 3 and 4. Until recognition exists, uploading a screenshot
keeps the image for reference, but you type the board in yourself on the Review page.

## Installing

Requirements: Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
cd scrabble-bot
uv sync
```

This creates `.venv/` with the runtime and development dependencies pinned in `uv.lock`. PyTorch
is only needed for training the tile classifier in a later milestone; add it with
`uv sync --extra train`.

Run anything inside the environment with `uv run`, for example `uv run python` or
`uv run pytest`.

> If you have another virtualenv active, uv prints a warning that `VIRTUAL_ENV` does not match
> `.venv` and ignores it. That's harmless. Run `deactivate` (or `unset VIRTUAL_ENV`) to silence it.

## Running the web app

```sh
uv run wwf-serve
```

Then open <http://127.0.0.1:8000>. The first start takes a few seconds while the default
dictionary is built and cached.

| Option | Effect |
| --- | --- |
| `--host 0.0.0.0` | Listen on your network so a phone on the same Wi-Fi can open `http://<your-computer's-IP>:8000` (find the IP with `ipconfig getifaddr en0` on a Mac) |
| `--port 8080` | Use another port |
| `--reload` | Restart automatically when code changes (development) |

Stop it with Ctrl+C. The server keeps everything in memory, so sessions don't survive a restart.

### Using it

1. **Upload** (`/`): there are three ways in, in this order.
   - **Take a photo of the board** is the main one:
     - On a phone it opens the camera app directly.
     - On a computer with a webcam it shows a live viewfinder; press **Capture**. Browsers only
       allow the webcam on `localhost` or `https`; elsewhere the button opens a file picker.
   - **Or upload a screenshot**: choose an existing image, or drop one onto the box.
   - **Type a board in by hand**: start from an empty board.

   The picture is sent as soon as it's taken or chosen, with no extra button. The browser first
   scales it to at most 2000 px on the long side, which keeps phone photos small and converts
   iPhone HEIC photos to JPEG. The last three boards you opened are listed as links.
2. **Review** (`/review/<id>`): the uploaded image is shown next to an editable 15x15 grid. Click
   or tap a square and type.
   - Typing a letter fills the square and moves right.
   - The arrow keys move between squares.
   - Backspace clears the square and moves left; on an empty square, it moves left and clears that one.
   - A **lowercase** letter marks a blank tile. On a phone, type the letter then press **Blank**
     to toggle the last square between tile and blank.
   - Enter up to 7 rack tiles below the board, using `?` for a blank.
   - **Save and solve** saves the board and opens the results; **Save** only saves.
   - **Retake photo** (or **Upload instead**) replaces the board with a new picture.
3. **Results** (`/results/<id>`): the ranked moves with score, start square (row, column,
   counting from 1), direction and leave. Click a move to see the board after playing it, with the
   new tiles outlined and the score of each word formed. The top shows which solver ran, how long
   it took, and its parameters and stats.
4. **Options** (`/options`): choose the solver, its parameters and the word list. Saving applies
   them to the board you came from (if any) and remembers them in a cookie for new boards.

Sessions expire after 30 minutes without use.

Everything works without JavaScript except:
- the review grid's keyboard shortcuts and the Blank button
- the webcam viewfinder, sending as soon as a picture is chosen, and downscaling
- the recent-boards list

Without JavaScript, pick the picture and press the button that appears under it.

## Collecting test pictures (fixtures)

Board recognition (milestones 3-4) has to be built and tested against real pictures with known
answers. Each of these "fixtures" is an image plus a JSON label holding the true board and rack.
The app has a collection mode for making them:

```sh
uv run wwf-serve --collect-fixtures
```

1. **Take pictures.** There are two ways.
   - **Mac webcam (the game on your phone):** run the server on the Mac and open
     <http://127.0.0.1:8000> there. Click **Take a photo of the board** and hold the phone up to
     the webcam.
     - Press **space** (or **Capture**) for each picture, or **In 3 s** for a countdown.
     - The camera stays open between shots, and the preview is mirrored like a selfie camera.
       Saved pictures are not mirrored.
     - **Done** or Escape closes it.
   - **Another device's camera:** start with `--host 0.0.0.0` and open
     `http://<your computer's IP>:8000` on it. Its camera app opens for each picture.

   Either way:
   - Every picture is saved straight into `assets/fixtures/` on the computer running the server.
   - Camera pictures are named `photo-<date>-<time>.jpg`. Uploaded PNGs count as screenshots
     (`screenshot-...png`); other uploads count as photos.
2. **Label them.** On your computer, open **Fixtures** in the top bar (`/fixtures`).
   - It lists every picture and whether it has a label yet.
   - Press **Label** to open one in the review grid next to the picture.
   - Type exactly what the picture shows: every tile on the board (lowercase for blanks) and the
     rack (`?` for a blank).
   - Press **Save label**. That writes `<name>.json` next to the image. **Edit label** reopens it.
3. **Prune.** Delete any picture you don't want straight from the folder.

What to capture for the photo path. Variety matters more than count, and 20-40 labelled photos
is a good target:
- different angles: straight on, tilted, rotated in the frame
- different distances, with the phone filling a third of the frame up to all of it
- glare from a lamp or window across the screen
- dim rooms, and screen brightness low and high
- different backgrounds behind the phone
- the moiré stripes you get when the camera is close to a screen
- early, mid and late games, including boards with blanks on them
- 8 or so plain screenshots, for the screenshot path

The label format, if you ever write one by hand:

```json
{
  "image": "photo-20261012-203300.jpg",
  "input_type": "photo",
  "board": ["...............", "... 15 strings of 15 characters ..."],
  "rack": "AEIRS?T",
  "labelled_at": "2026-10-12T20:40:00"
}
```

Use `.` for an empty square, uppercase for a tile and lowercase for a blank. Add
`--collect-fixtures some/other/folder` to save somewhere else.

## The HTTP API

Interactive documentation for every endpoint is at <http://127.0.0.1:8000/docs> while the server
runs; the raw schema is at `/openapi.json`. All endpoints live under `/api/v1`.

| Method and path | Body | Returns |
| --- | --- | --- |
| `POST /recognize` | multipart `image` | Recognition result plus a new `session_id` |
| `POST /session` | none | A new empty session, for typing a board in |
| `GET /session/{id}` | none | Session state: board, rack, options, whether it has an image or result |
| `GET /session/{id}/image` | none | The uploaded image |
| `PUT /session/{id}/board` | `{board, rack}` | Session state; replaces both |
| `PUT /session/{id}/options` | `{solver_id, params, dictionary_id?}` | Session state; 422 if invalid |
| `POST /session/{id}/solve` | `{solver_id?, params?, dictionary_id?}` (all optional) | Ranked moves |
| `GET /solvers` | none | Every solver with its parameter schema |
| `GET /dictionaries` | none | Every word list |
| `POST /validate` | `{board, rack, move, dictionary_id?}` | Named violations, plus the score if legal |
| `GET /health` | none | Status, default dictionary and word count, whether a vision model is loaded |

Formats:

- **Board:** a 15x15 array of `null` or a one-letter string. Uppercase is a normal tile, lowercase
  a blank.
- **Rack:** a string, with `?` for a blank.
- **Coordinates:** `[row, col]`, zero-based. The web pages show them counting from 1.
- **Moves:** each solve returns a list of moves. Every move has `word`, `direction`, `start`,
  `placements` (`row`, `col`, `letter`, `is_blank`), `score` (`total`, `bingo_bonus`, and per-word
  `points`), `leave`, and `board_after`, the full board after playing it.

On `POST /session/{id}/solve`:
- **Params:** any `params` given are merged over the session's saved options for that call only.
- **Switching solver:** a `solver_id` different from the session's starts from that solver's
  defaults.

Every response carries a `request_id`, also sent as the `X-Request-ID` header and written to the
server log. Errors look like `{"error": "...", "detail": ..., "request_id": "..."}`:

| Status | When |
| --- | --- |
| 400 `undecodable_image` | The upload isn't an image |
| 404 `session_not_found` | Unknown or expired session |
| 413 `image_too_large` | Upload over 10 MB |
| 422 `validation_error`, `invalid_params`, `unknown_solver`, `unknown_dictionary` | Bad input |
| 504 `solve_timeout` | The solver ran over 10 seconds (`detail.partial` is true) |

A complete round trip with curl:

```sh
SID=$(curl -s -X POST localhost:8000/api/v1/session | python3 -c "import sys,json;print(json.load(sys.stdin)['session_id'])")
python3 -c "
import json; b = [[None]*15 for _ in range(15)]
for i, ch in enumerate('HELLO'): b[7][5+i] = ch
print(json.dumps({'board': b, 'rack': 'SATIRE?'}))" > board.json
curl -s -X PUT -H 'content-type: application/json' -d @board.json localhost:8000/api/v1/session/$SID/board > /dev/null
curl -s -X POST -H 'content-type: application/json' -d '{"params": {"max_results": 3}}' \
  localhost:8000/api/v1/session/$SID/solve | python3 -m json.tool | head -40
```

## Using it from Python

The web app is built on a plain Python library, which you can use directly.

### A first solve

```python
from wwf.core.board import Board
from wwf.core.config import load_board_config
from wwf.core.dictionary import default_dictionary_registry
from wwf.core.rack import Rack
from wwf.solvers.registry import load_registry

cfg = load_board_config()  # config/wwf_classic.json
words = default_dictionary_registry().get()  # the default word list (ENABLE1)
solver = load_registry().get("anchor_trie")

rows = ["..............."] * 15
rows[7] = ".....HELLO....."  # HELLO across row 7, columns 5-9
board = Board.from_rows(cfg, rows)
rack = Rack.from_string("SATIRE?")  # "?" is a blank

result = solver.solve(board, rack, words, cfg, max_results=5)
for m in result.moves:
    print(m.score.total, m.move.main_word, m.move.direction.value, m.move.start, m.leave)
```

Output:

```text
61 AERIEST down (2, 10)
61 AERIEST down (2, 10)
61 AIGRETS down (1, 10)
61 AIRIEST down (2, 10)
61 ARTIEST down (2, 10)
```

The two AERIEST rows are different moves: the blank stands for the first E in one and the second
E in the other. Each move here uses all seven tiles, so `leave` is empty. Print
`[(p.row, p.col, p.tile.to_char()) for p in m.move.placements]` to see where the blank went; a
lowercase letter is a blank.

The first call in a fresh checkout takes a few seconds while the dictionary is built and cached.
Later runs load the cache in under a second.

## Board notation and coordinates

- Coordinates are `(row, col)`, zero-based, from the top left. The board is 15x15 and the centre
  is `(7, 7)`.
- `Board.from_rows(cfg, rows)` takes 15 strings of 15 characters:
  - `.` or a space: empty cell
  - `A`-`Z`: a normal tile
  - `a`-`z`: a blank tile standing for that letter (it scores 0)
- `from_rows` also accepts 15 lists of 15 items, each a one-character string or `None`. That's the
  format the API will use.
- `board.to_rows()` and `board.to_grid()` convert back.
- Racks are strings: `Rack.from_string("abc?")` gives `("A", "B", "C", "?")`. Case doesn't matter
  for racks, and they hold at most 7 tiles.

Bonus squares come from the layout in `config/wwf_classic.json`:

| Character | Meaning |
| --- | --- |
| `T` | triple word |
| `D` | double word |
| `t` | triple letter |
| `d` | double letter |
| `*` | centre (no bonus in WWF) |
| `.` | no bonus |

Useful board methods:

| Method | Returns |
| --- | --- |
| `Board.empty(cfg)` | An empty board with the configured bonuses |
| `board.get(r, c)` | The `Tile` at a cell, or `None` (also `None` off the board) |
| `board.bonus(r, c)` | The `Bonus` at a cell |
| `board.anchors()` | Empty cells next to a tile, or `{(7, 7)}` on an empty board |
| `board.with_move(move)` | A new board with the move played (boards are immutable) |
| `board.transpose()` | The board flipped across its diagonal |
| `board.is_empty()`, `board.tile_count()` | Board state |

## Moves, scoring and validation

A `Move` holds only the newly placed tiles, plus the main word's direction, text and start cell.
Build one from placements with `build_move`, which works out the direction and main word for you:

```python
from wwf.core.board import Tile
from wwf.core.move import Placement, build_move
from wwf.core.scoring import score_move, validate_move

move = build_move(board, [Placement(8, 8, Tile("A")), Placement(8, 9, Tile("X"))])
# Move(..., direction=ACROSS, main_word='AX', start=(8, 8))

score_move(board, move, cfg)
# ScoreBreakdown(words=(AX 9, LA 3, OX 9), bingo_bonus=0, total=21)

validate_move(board, move, words, cfg, Rack.from_string("AX"))
# []   (an empty list means the move is legal)
```

For a blank, use `Tile("E", is_blank=True)`, or `Tile.from_char("e")`.

`score_move` applies the WWF rules:

- Letter bonuses apply only to tiles placed this turn.
- Word bonuses multiply the whole word and stack.
- Every perpendicular word formed is scored too.
- Placing all 7 tiles adds 35 points.

`validate_move(board, move, dictionary, cfg, rack=None)` returns a list of named violations. The
`rack` argument is optional; without it, the rack check is skipped.

| Violation | Meaning |
| --- | --- |
| `not_in_line` | New tiles are not all in one row or one column |
| `gap_in_word` | An empty cell between the new tiles |
| `first_move_not_on_center` | The first move doesn't cover `(7, 7)` |
| `no_contact_with_existing_tiles` | A later move touches no existing tile |
| `cell_occupied` | A tile placed on an occupied cell |
| `word_not_in_dictionary:<WORD>` | A formed word isn't in the dictionary; single letters never count as words |
| `tile_not_in_rack:<L>` | The rack lacks that tile (`?` for blanks) |
| `no_tiles_placed`, `out_of_bounds` | Malformed input |

For example, a two-tile `QI` in the corner with rack `Q` on the board above returns:

```text
['no_contact_with_existing_tiles', 'tile_not_in_rack:I', 'word_not_in_dictionary:QI']
```

`score_raw(board, placements, cfg)` returns only the total, given raw
`(row, col, letter, is_blank)` tuples. It's the fast path solvers use to rank thousands of
candidates, and it always equals `score_move(...).total`.

## Solvers

### The registry

```python
from wwf.solvers.registry import load_registry

registry = load_registry()  # reads config/solvers.json
registry.default_id  # "anchor_trie"
solver = registry.get("anchor_trie")  # KeyError for unknown or disabled ids
for info in registry.infos():  # every listed solver, including planned ones
    print(info.id, info.display_name, info.enabled, [p.name for p in info.params])
```

```text
anchor_trie Smart brute force True ['max_results', 'min_score', 'allow_blank_branching', 'tie_break']
dummy_random Random legal play (demo) True ['attempts', 'max_results', 'seed']
gaddag GADDAG (planned) False []
leave_aware Leave-aware (planned) False []
monte_carlo Monte Carlo (planned) False []
```

Every solver is called the same way, with its parameters as keyword arguments:

```python
result = solver.solve(board, rack, dictionary, cfg, **params)
```

### Results

`solve` returns a `SolverResult` with these fields:

- **`moves`:** a tuple of `ScoredMove`, best first. Each has:
  - `.move`: the `Move`
  - `.score`: a `ScoreBreakdown` with `.total`, `.bingo_bonus`, and `.words` (each with `.word`, `.points`, `.start`, `.direction`)
  - `.leave`: the rack tiles left after the move, as a string
- **`elapsed_ms`:** how long the solve took.
- **`stats`:** solver-specific counters. `anchor_trie` reports `anchors`, `cross_checks`, `candidates`, `nodes_visited` and `moves`; `dummy_random` reports `attempts` and `legal_found`.

### `anchor_trie`: "Smart brute force"

Finds every legal move (Appel-Jacobson anchors and cross-checks over a trie) and ranks them by
score. Its completeness is tested against a brute-force reference generator.

| Parameter | Type | Default | Range | Effect |
| --- | --- | --- | --- | --- |
| `max_results` | int | 25 | 1-500 | Moves returned |
| `min_score` | int | 0 | >= 0 | Drop moves scoring less |
| `allow_blank_branching` | bool | true | | Off ignores blanks entirely (faster) |
| `tie_break` | choice | `score_then_leave` | `score_then_leave`, `score_only` | How equal scores are ordered |

With `score_then_leave`, equal scores are ordered by a simple leave heuristic. It prefers a
balanced mix of vowels and consonants, penalises a Q without a U and duplicate letters, and values
keeping a blank or an S. With `score_only`, equal scores are ordered alphabetically by word.

Typical speed: a full mid-game board with a 7-tile rack including one blank solves in a few
hundred milliseconds. Two blanks take roughly 1.5 s.

### `dummy_random`: demo only

Tries random single-tile plays at anchor squares and keeps the legal ones. It exists to exercise
the plugin interface; don't use it to play.

| Parameter | Type | Default | Range |
| --- | --- | --- | --- |
| `attempts` | int | 200 | 1-10000 |
| `max_results` | int | 25 | 1-500 |
| `seed` | int | -1 | -1 means random each time; other values repeat results |

### Bad parameters

Unknown names, wrong types and out-of-range values raise `wwf.solvers.base.ParamError`. Its
`.errors` attribute lists every problem at once:

```python
solver.solve(board, rack, words, cfg, max_results=0, tie_break="nope")
# ParamError: max_results: must be >= 1; tie_break: expected one of ['score_then_leave', 'score_only']
```

Strings are coerced where sensible (`"7"` becomes 7, `"false"` becomes False), so form values can
be passed straight through.

## Dictionaries

Two word lists ship in `assets/dictionaries/`:

| Id | File | Notes |
| --- | --- | --- |
| `enable1` (default) | `enable1.txt` | Public-domain ENABLE1, ~173k words. Has no QI or ZA. |
| `words` | `words.txt` | ~194k entries. Includes QI. |

The real WWF dictionary is proprietary, so neither list matches the live game exactly. Expect the
occasional word the game rejects, or a game word the solver never suggests.

### Choosing one

```python
from wwf.core.dictionary import default_dictionary_registry

dictionaries = default_dictionary_registry()
dictionaries.get()  # the default
dictionaries.get("words")  # a specific list; KeyError if unknown
[i.id for i in dictionaries.infos()]  # ['enable1', 'words']
```

You can change the default in two ways:

- Edit `"default"` in `config/dictionaries.json`.
- Set the environment variable, e.g. `WWF_DICTIONARY=words uv run python ...`. This wins over the
  config file.

Each list is loaded on first use and then kept in memory, so switching back and forth is free
after the first load.

### Adding a word list

Drop a newline-separated word list into `assets/dictionaries/`. Any `.txt` file there becomes
selectable under its file name (`my_list.txt` becomes id `my_list`) with no config change. To give
it a label and description, add an entry to `config/dictionaries.json`:

```json
{"id": "my_list", "file": "my_list.txt", "label": "My list", "description": "..."}
```

Case doesn't matter. Lines that are single letters or contain anything other than A-Z (such as
`ain't`) are skipped.

### The cache

The first load of a list builds a trie and saves it to `.cache/<name>.<hash>.v2.pickle`. The hash
is of the file's contents, so editing a word list automatically invalidates its cache. Deleting
`.cache/` is always safe.

`load_dictionary` also calls `gc.freeze()`. This keeps Python's garbage collector from rescanning
the ~390k trie nodes, which would otherwise roughly double solve times.

## Configuration reference

All game data lives in `config/`, not in Python.

### `config/wwf_classic.json`

| Field | Meaning |
| --- | --- |
| `size` | Board size (15) |
| `center` | Centre cell, `[7, 7]` |
| `layout` | The full 15 row strings, using the bonus characters above |
| `letter_values` | Points per letter; `?` (blank) is 0 |
| `bag` | Tile distribution, 104 tiles. Read but unused in v1. |
| `bingo_bonus` | 35 (WWF; Scrabble uses 50) |
| `bingo_tiles` | 7 |

The loader rejects a layout with the wrong number of rows or columns, unknown characters, or one
that isn't mirror-symmetric top-to-bottom and left-to-right. A typo fails loudly at startup.

To load a different game config, use `load_board_config(Path("config/other.json"))`.

> The bonus layout follows the build spec and still needs checking against a real WWF screenshot.
> WWF variants (Solo, Fast Play) use other layouts.

### `config/solvers.json`

```json
{"solvers": [
  {"id": "anchor_trie", "enabled": true, "default": true},
  {"id": "dummy_random", "enabled": true},
  {"id": "gaddag", "enabled": false, "display_name": "GADDAG (planned)", "description": "..."}
]}
```

- `enabled: true` needs a matching implementation in `src/wwf/solvers/`, or loading fails with
  `RegistryError`.
- `enabled: false` entries with no code are listed as planned; the Options page will show them
  greyed out.
- At most one entry may be `default`.
- `display_name` and `description` here override the solver class's own.

### `config/dictionaries.json`

This file is described under [Dictionaries](#dictionaries).

### Environment variables

| Variable | Effect |
| --- | --- |
| `WWF_ROOT` | Directory that holds `config/` and `assets/`. Defaults to the repo root. |
| `WWF_DICTIONARY` | Default dictionary id |

## Adding a solver

1. Create a module in `src/wwf/solvers/`, for example `greedy_vowels.py`:

   ```python
   from typing import Any

   from wwf.solvers.base import SolverParam, SolverResult, resolve_params


   class GreedyVowelsSolver:
       id = "greedy_vowels"
       display_name = "Greedy vowels"
       description = "Example solver."
       params = (SolverParam("max_results", "Max results", "int", 10, min=1, max=100),)

       def solve(self, board, rack, dictionary, cfg, **params: Any) -> SolverResult:
           p = resolve_params(self.params, params)
           ...
   ```

2. Add `{"id": "greedy_vowels", "enabled": true}` to `config/solvers.json`.

The registry finds the class automatically. A few rules:

- Score moves with `wwf.core.scoring`; don't reimplement scoring.
- Return moves sorted best first.
- A solver may import only from `wwf.core`; import-linter enforces this.
- `wwf.solvers.anchor_trie.generate_raw_moves` gives you every legal move to build on.

## Tests and code quality

```sh
uv run pytest               # fast suite, about 30 s
uv run pytest -m ""         # also the slow tests (the 500 ms performance check)
uv run pytest tests/solvers -k reference   # just the completeness proof
uv run ruff check . && uv run ruff format --check .
uv run mypy                 # strict on wwf.core and wwf.solvers
uv run lint-imports         # layering: core <- {vision, solvers} <- api
```

What the tests cover:

- **`tests/core`:**
  - worked scoring examples (HELLO, a blank on a triple word, a bingo, parallel plays, stacked word bonuses)
  - every named violation
  - config rejection of malformed layouts
  - the dictionary cache and the dictionary registry
  - a Hypothesis property that scores don't change when the board is transposed
- **`tests/api`:**
  - every API route and error code, plus session expiry, options validation and the solve timeout
  - the pages, driven through their HTML forms as a browser without JavaScript would
- **`tests/solvers`:**
  - the equivalence test: `anchor_trie` must find exactly the same legal moves and scores as the slow brute-force generator in `tests/solvers/reference.py`, on 20 random boards
  - every returned move passes `validate_move`
  - registry behaviour, parameter validation, `dummy_random` limits, and the performance bar

CI (`.github/workflows/ci.yml`) runs all of this on every push, plus the slow tests nightly.

## Project layout

```text
config/                   board layout, solver registry, dictionary list
assets/dictionaries/      word lists (.txt)
src/wwf/core/             board.py, rack.py, move.py, scoring.py, dictionary.py, config.py
src/wwf/solvers/          base.py, registry.py, anchor_trie.py, dummy_random.py
src/wwf/api/              app.py (factory, wwf-serve), services.py, session.py, schemas.py, routes_*.py
src/wwf/web/              templates/ (Jinja2 pages), static/ (board.css, app.js, vendored htmx)
src/wwf/vision/           (milestones 3-4)
tests/                    core/, solvers/, vision/, api/
docs/usage.md             this file
.cache/                   dictionary caches (generated, git-ignored)
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `address already in use` from `wwf-serve` | Another server is on that port. Use `--port 8080`, or stop the other one. |
| Phone can't open the app | Start with `--host 0.0.0.0`, use the computer's LAN IP, keep both on the same Wi-Fi, and allow Python through the macOS firewall if asked. |
| Fixtures page says "fixture_mode_off" | Start the server with `--collect-fixtures`. |
| "Session expired" | Sessions last 30 minutes after last use, and are lost when the server restarts. Start a new board. |
| First solve takes several seconds | The dictionary is being built and cached. Later runs are fast. |
| `RegistryError: solvers.json enables unknown solver id` | An enabled entry in `config/solvers.json` has no class in `src/wwf/solvers/`. Fix the id or set `"enabled": false`. |
| `DictionaryError: default dictionary 'x' is not available` | `WWF_DICTIONARY` or the `"default"` in `config/dictionaries.json` names a list that doesn't exist. |
| `ConfigError: layout is not symmetric` | A typo in `config/wwf_classic.json`. |
| A real WWF word is never suggested | It isn't in the chosen word list. Try `words`, or add your own list. |
| `ValueError: cell (r,c) is occupied` from `with_move` | The move places a tile on an occupied cell. Run `validate_move` first. |
| Solves are slow in your own scripts | Load dictionaries with `load_dictionary` or the registry so `gc.freeze()` runs. Avoid creating thousands of other long-lived objects between solves. |
