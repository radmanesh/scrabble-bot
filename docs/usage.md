# WWF Solver: usage guide

This guide covers everything you can run and use today: the web app, the Python
library (board and move model, scoring and rule checks, solvers, dictionaries), configuration,
tests, and how to extend it.

## What works today

The project is built in milestones (see the build spec).

| Area | Status |
| --- | --- |
| Core model: board, rack, moves, scoring, rule validation | Done (M1) |
| Dictionary loading, caching and choosing between word lists | Done |
| Solvers: `anchor_trie` ("Smart brute force") and `dummy_random` (demo) | Done (M2) |
| Web pages: home, game (board, moves, typed moves, pictures, history), options | Done (M6), not yet tried on a real phone |
| Multi-turn games: play, opponent's move, undo, scores, unseen tiles, saved to disk | Done |
| Board reader (`wwf.vision`): finds the board in a photo and reads squares and rack | First version, trained on your labelled photos |
| Docker image | Not built (M7) |

Milestones 5 and 6 were built ahead of 3 and 4. The board reader is a first version: it fills the
board in from a photo, outlines squares it is unsure of in amber, and you correct the rest.

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

Stop it with Ctrl+C. Games are saved to disk after every action, so nothing is lost.

### Playing a game

A game is one page, `/game/<id>`. Pictures, typed words and the solver's suggestions all feed the
same thing: an **open turn**, which stays editable until you **confirm** it.

```text
 set up:       picture or typing -> fix any square -> Confirm starting position (locks it)
 each turn:    put tiles on the board (suggestion / typed word / picture)
               -> fix anything that's wrong -> Confirm turn (locks it)
```

**Games (`/`)** shows each game as a card: a thumbnail of its saved board (unconfirmed tiles are
dashed), whether it is your turn or the opponent's, the score as *You* and *Opponent*, when it was
last played, and **Continue game**. **More** on a card holds Rename and Delete (Delete names the
game and asks first). With more than four games a filter box narrows them by name. **New game**
beside the title offers:

- **Take a photo of the board**, the main way. On a phone it opens the camera app; on a computer
  with a webcam it shows a live viewfinder (browsers only allow the webcam on `localhost` or
  `https`; elsewhere it opens a file picker).
- **Upload image**: choose an existing screenshot or photo, or drop one onto the box.
- **Enter board manually**: start from an empty board.

A chosen or captured picture is shown first with **Read the board** and **Choose another**:
nothing is sent, and nothing changes, until you read it. The browser scales it to at most 2000 px
on the long side, which keeps phone photos small and converts iPhone HEIC photos to JPEG.

**The starting position.** A new game isn't tracked yet. The board is read from the picture (or
typed) and every square is editable. Check it against your phone, then press **Confirm · it's my
turn** or **Confirm · it's their turn**. That locks the position as the first entry in *Turns*.
It isn't checked for legality: it's whatever your phone shows.

**Locked and open tiles.** From then on, tiles from confirmed turns are locked: grey, and not
editable (the server ignores any change to them too). Tiles of the open turn have a green edge and
can be changed freely. Locked tiles can only be fixed with **Undo last turn**, which reopens the
last confirmed turn so you can edit it and confirm again. Your **rack is always editable**.

**The game page**, from the top:

1. **Header**: *Back to games*, the game's name (with Rename), the score, the tiles left in the
   bag, and whose move is next. Points count turns confirmed here.
2. **What this turn needs**:
   - *Starting position* (before the game is tracked): the two **Confirm** buttons.
   - *What did you draw?* on your turn when your rack isn't full. **Add and solve** goes on.
   - The **open turn card**, once there are new tiles: the word, where it starts, its score, and
     problems if it isn't one legal move. Buttons: **Confirm your move** (or **opponent's**),
     **It was the other player's move**, **Cancel this turn** (board and rack go back to how the
     turn began) and **Just update the board**, which keeps the tiles without scoring them (a
     picture that caught up several turns; it shows in *Turns* as *Board updated*).
3. **The board**, with row and column numbers along its edges. It is one tab stop: use the arrow
   keys to move (Home and End jump along a row) and Tab to leave it. A letter fills the square and
   moves right; Backspace clears; lowercase marks a blank (or press **Blank**). Amber squares are
   ones the reader isn't sure of. **Larger board** (phones) makes the squares big enough to touch
   precisely and scrolls inside the card; Escape leaves it. **Save** keeps your edits.
4. **What to do next**, beside the board on a wide screen and above it on a phone:
   **Find my best move**, **Record my move**, **Record opponent's move** and **Update from photo**.
   The one that matches the turn is already showing.
   - *Find my best move* lists the best five moves (**Show more** for the rest) with the tiles
     each leaves you. Choosing one **previews it on the board** (dashed tiles, with its score);
     nothing changes. **Record this move** places it and confirms it as your move in one step.
   - *Record my move / Record opponent's move* is the move composer. Type the whole word
     (including letters already on the board), tap a square or type the row and column of its
     first letter, and choose **Across** or **Down**. The word is drawn on the board with its
     score and any problem before anything is recorded: dashed tiles are new, underlined tiles
     are ones it reuses. Mark a blank by tapping its letter under the word. **Record** places and
     confirms it as exactly one turn for the player shown. If only the word list disagrees you're
     offered *record it even so*, because the game itself accepted it; such turns are marked *not
     in list*. A rejected move keeps what you typed and changes nothing.
   - *Pass* is a separate button, worded for the player being recorded.
5. **Pictures**: **Take a picture** / **Upload instead**, after the preview step. Confirmed tiles
   are kept; new tiles and your rack are read into the open turn (replacing any open tiles from
   an earlier picture). Squares the reader was unsure of are listed with an enlarged crop from the
   picture. Pictures open in an in-app viewer with *Original* and *Straightened board*, a zoom,
   and previous/next.
6. **Turns**: every confirmed turn with who played it, its words and points, the totals after it
   and the picture it came from. The starting position and board updates are marked *Setup* and
   aren't numbered. **Undo the last turn** says what it will take back; it is different from
   **Cancel this turn**, which discards tiles that were never confirmed. Also the unseen tiles and
   **Delete this game**.

A turn is either a suggestion you played, a typed word, or a picture. Only one is open at a time:
confirm or cancel it before putting another on the board. A picture's new tiles must form one
legal move to confirm; if they don't, correct the outlined squares, or use **Just update the
board**.

**Unseen tiles.** Everything not on the board or your rack: the opponent's rack plus the bag. If
the board shows more of a letter than the game has, the page says so, because one of them is a
blank that looks like a normal tile. The squares with those letters get a dotted edge: mark the
blank with the **Blank** button and the count comes right. Nothing is changed for you.

**Saving.** Every game is `data/games/<id>.json` plus a folder `data/games/<id>/` holding all its
pictures. Games survive restarting the server and can last weeks. Set `WWF_GAMES_DIR` to keep
them elsewhere. A game saved by an earlier version keeps its one picture, and a game that had
turns but no confirmed starting position asks you to confirm it.

**Settings (`/options`)**: *Maximum suggestions*, *Minimum score*, *Include blank tiles*, *Tie
preference* and the *Word list*. The page says what a save applies to: the game you came from (if
any) and games you start from now on; other games keep theirs. **Save and back to the game**
returns with a confirmation. Changing anything clears the suggestions on screen, because they
were found with the old settings. *Advanced* holds how moves are found, including engines that
are planned but not available yet.

Everything works without JavaScript except:
- the live preview of a typed word (without it, **Check the word** draws it after a page load)
- choosing a suggestion without a page load, the single board tab stop and the Larger board mode
- the picture preview step, the webcam viewfinder and downscaling
- the games filter, and the delete confirmation

Without JavaScript, pick the picture and press the button that appears under it.

## Pictures and how the reader learns

There is no separate "collect" mode or labelling page any more: **every picture belongs to a
game**, and fixing the board on the game page *is* the label.

- Each picture is kept with its game (`data/games/<id>/`), along with what the reader said about
  it and where it found the board.
- When you **confirm** a turn that came from a picture, the confirmed board and rack become that
  picture's label (the latest picture of that turn; earlier ones are kept but not used). Pictures behind turns you typed, or suggestions you played, are not used,
  because their board may include tiles the picture doesn't show.
- The reader is **re-trained in the background after each such turn**, from those pictures plus
  the older pictures in `assets/archive/` (labelled before games kept their own pictures; they
  are used for training only and don't appear in the app). Improved readings apply to the next
  picture; no restart is needed.

## The board reader

When you take or upload a picture, the app reads the board and rack and fills in the game page.
Squares it is unsure of are outlined in amber, and so are rack tiles it is unsure of. Check those,
fix anything wrong, then solve.

How it works:
1. **Finding the board.** The coloured bonus squares (blue DL, green TL, red DW, orange TW) sit
   at known places. The reader finds blobs of those colours and votes for the size, tilt and
   position that line the layout up with them, over a wide range of sizes (so a board filling the
   picture and one far away both work), then fits a full perspective correction. A picture under
   a warm lamp is white-balanced and tried again if the first look finds nothing solid. Several
   candidate positions are compared by how cream-like the empty squares look, so a wrong size
   (which puts the grid across tile edges) is not chosen.
2. **Aligning.** Every letter sits at its square's centre, so where the letters actually are
   corrects the grid by a fraction of a square. This matters late in a game, when most bonus
   squares are covered.
3. **Empty or tile.** Every square is measured **relative to this picture's own colours**: how far
   it is from the picture's empty-cream colour (or from what its bonus kind looks like here), and
   how strongly a letter-shaped stroke stands out from its own background, and how dark its
   darkest ink is (a tile's letter is dark, while an empty bonus square's "DL" or "TW" label is
   white). A small model learned from your labelled pictures turns that into a probability. Because it is relative, a warm
   lamp, glare and vignetting don't turn empty squares into tiles.
4. **Letters.** Each square's crop is first moved so its letter is in the middle (the grid is
   rarely exact to better than a fraction of a square, and a letter cut off at the edge reads as
   another: an L without its foot is an I). The letter is then cut out, whether dark or white (the
   last move's tiles have white letters), without the small point value in the tile's corner, and
   matched against letters cut from your labelled photos plus letters drawn from
   system fonts so every letter has examples.
5. **Rack.** The seven slots below the board are read the same way. A tile with no letter is a
   blank (`?`). If the rack is partly outside the picture it isn't read, and your rack is left
   alone.

### Reading the next turn

When a game already has a board, a new picture is read against it:

- **Tiles stay.** Tiles never leave a WWF board, so every tile on file is kept and only empty
  squares are read.
- **Is it the same game?** The reader checks how well the picture agrees with the board on file.
  If it doesn't (a picture of another game, or a board cut off at the edge), nothing is changed
  and a message says why.
- **The board on file tunes the reader** to this picture's light: it knows which squares are
  tiles and which are empty, so the few new tiles are judged against that.
- **New tiles must join the game.** A new tile that isn't connected to the tiles on the board
  (a reflection, an on-screen button) is ignored. At most 21 new tiles are accepted in one turn;
  more than that is a misreading and none are added.
- **Kept tiles are never overwritten.** If the picture confidently shows a kept tile as empty, it
  is outlined in amber and keeps its letter.

What you may see after a picture:

| Message | Meaning |
| --- | --- |
| *That picture doesn't look like this game's board* | It disagrees with the tiles on file. Is it another game? Nothing changed. |
| *The board runs off the edge of that picture* | Hold the phone further away so the whole board and rack are in view. Nothing changed. |
| *Your rack is partly outside the picture* | The rack wasn't read; check the rack squares. |
| *More new tiles than one turn can add* | Probably glare. None were added: retake, or type them in. |
| *Some marks aren't joined to the rest of the game* | A reflection or button over the board. They're outlined in amber. |

### How well it does

On pictures held out of training, each game position in turn (`python -m wwf.vision.train
--evaluate`):

| Test | Result |
| --- | --- |
| First picture of a game: squares read right | 99.4% (46 errors in 34 pictures: 14 tiles invented, 9 missed, 23 wrong letters) |
| Next turn, a few tiles removed from the board on file | 6 errors over 27 pictures; none refused wrongly |
| Empty board under a heavy orange lamp | 0 tiles (it used to read 168) |
| Picture of a different game | refused every time |

Each picture takes 0.3 to 1 second.

Limits for now:
- The rack's position is measured for your phone's screen shape. A phone with a different shape
  needs it re-measured (`RACK_*` in `src/wwf/vision/reader.py`).
- Blanks already on the board can't be seen, because this WWF version shows no point values on
  board tiles.
- Look-alike letters (F/E, P/R, O/Q, C/G, I/L) are the usual mistakes, and get rarer as you label
  more pictures.
- Squares are scored against your confirmed boards, so a wrong label counts as a reader error.
  Squares a picture strongly contradicts are left out of training and scoring (`--check` lists
  them): fix those labels and retrain. Only the latest picture of a turn becomes a label, because
  an earlier one may show the board before some of the confirmed tiles were played.
- The board is found from its coloured bonus squares. If glare or a hand hides most of them, or
  the picture is tinted so heavily that colours cannot be told apart, it says it couldn't find
  the board rather than guessing.

### It learns from your games

Confirming a turn that came from a picture teaches the reader (see *Pictures and how the reader
learns*). To retrain or measure it by hand:

```sh
uv run python -m wwf.vision.train              # rebuild assets/models/reader.npz from assets/archive/ and data/games/
uv run python -m wwf.vision.train --evaluate   # accuracy, holding out each game position in turn
uv run python -m wwf.vision.train --check      # labels that their pictures contradict
```

Pass `--fixtures DIR` (the archive) or `--games DIR` to use other folders. The model file
`assets/models/reader.npz` is built from your photos and isn't in git. On a fresh checkout, run
the first command once labelled pictures are in place. Without a model, the app still works, but
you type boards in by hand.

## Health check

`GET /api/v1/health` returns `{status, dictionary_id, dictionary_words, model_loaded, games}`.
The earlier JSON API (sessions, `/recognize`, `/solve`, `/validate`) was removed: the pages are
the only interface, and the Python library below is the programmatic one.

## Using it from Python

The web app is built on a plain Python library, which you can use directly.

### A first solve

```python
from wwf.core.board import Board
from wwf.core.config import load_board_config
from wwf.core.dictionary import default_dictionary_registry
from wwf.core.rack import Rack
from wwf.solvers.registry import load_registry

cfg = load_board_config()  # config/wwf_current.json
words = default_dictionary_registry().get()  # the default word list (words.txt)
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
57 CHARIEST down (6, 5)
57 CHARITES down (6, 5)
57 HAIRIEST down (7, 5)
57 HARDIEST down (7, 5)
57 HEARTIES down (7, 5)
```

Each of these plays all seven tiles down through the H of HELLO, so `leave` is empty and the
35-point bingo bonus is included. The blank stands in for the letter that isn't on the rack
(the C, an extra I, or the D). Print
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

Bonus squares come from the layout in `config/wwf_current.json` (see the configuration reference):

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
| `enable1` | `enable1.txt` | Public-domain ENABLE1, ~173k words. Has no QI or ZA. |
| `words` (default) | `words.txt` | ~194k entries. Includes QI. |

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

### Board configs: `config/wwf_current.json` and `config/wwf_classic.json`

Two layouts ship:
- **`wwf_current.json` (the default):** the bonus layout of the current WWF app, read from real
  photos. Its centre square was hidden in every photo, so it's assumed to carry no bonus.
- **`wwf_classic.json`:** the layout from the build spec.

Both use the same letter values, bag and 35-point bingo. Pick one with
`WWF_BOARD_CONFIG=wwf_classic.json`. The core scoring tests use the classic layout, since their
worked examples were computed on it.

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
| `WWF_GAMES_DIR` | Where saved games live (default `data/games`) |
| `WWF_BOARD_CONFIG` | Board config file in `config/` to use, e.g. `wwf_classic.json` |

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
  - the pages, driven through their HTML forms as a browser without JavaScript would: the open turn and its confirmation, locked squares, undo, pictures and their labels, options, saving
- **`tests/solvers`:**
  - the equivalence test: `anchor_trie` must find exactly the same legal moves and scores as the slow brute-force generator in `tests/solvers/reference.py`, on 20 random boards
  - every returned move passes `validate_move`
  - registry behaviour, parameter validation, `dummy_random` limits, and the performance bar

CI (`.github/workflows/ci.yml`) runs all of this on every push, plus the slow tests nightly.

## Project layout

```text
config/                   board layout, solver registry, dictionary list
assets/dictionaries/      word lists (.txt)
assets/archive/           older labelled pictures, training only (git-ignored)
src/wwf/core/             board.py, rack.py, move.py, scoring.py, dictionary.py, config.py
src/wwf/solvers/          base.py, registry.py, anchor_trie.py, dummy_random.py
src/wwf/api/              app.py (factory, wwf-serve), services.py, schemas.py,
                          games.py (games, turns, pictures, storage), gameplay.py (open turn,
                          confirm, undo), routes_web.py, routes_game_pages.py
src/wwf/web/              templates/ (Jinja2 pages), static/ (board.css, app.js, vendored htmx)
src/wwf/vision/           board.py (find the board), align.py (fine alignment), occupancy.py (tile or
                          empty, relative to the picture), reader.py (letters, rack), pipeline.py,
                          train.py (build, evaluate, check labels), synth.py, perspective.py
tests/                    core/, solvers/, vision/, api/
docs/usage.md             this file
.cache/                   dictionary caches (generated, git-ignored)
data/games/               saved games (generated, git-ignored)
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `address already in use` from `wwf-serve` | Another server is on that port. Use `--port 8080`, or stop the other one. |
| Phone can't open the app | Start with `--host 0.0.0.0`, use the computer's LAN IP, keep both on the same Wi-Fi, and allow Python through the macOS firewall if asked. |
| A tile won't change | It belongs to a confirmed turn. Press **Undo last turn** until it's open again, fix it and confirm. |
| "Confirm or cancel the open turn first" | There are unconfirmed tiles on the board. Confirm them, or press **Cancel this turn**. |
| "Game not found" | It was deleted, or the server is using a different `WWF_GAMES_DIR`. |
| First solve takes several seconds | The dictionary is being built and cached. Later runs are fast. |
| `RegistryError: solvers.json enables unknown solver id` | An enabled entry in `config/solvers.json` has no class in `src/wwf/solvers/`. Fix the id or set `"enabled": false`. |
| `DictionaryError: default dictionary 'x' is not available` | `WWF_DICTIONARY` or the `"default"` in `config/dictionaries.json` names a list that doesn't exist. |
| `ConfigError: layout is not symmetric` | A typo in the board config file. |
| A real WWF word is never suggested | It isn't in the chosen word list. Try `words`, or add your own list. |
| `ValueError: cell (r,c) is occupied` from `with_move` | The move places a tile on an occupied cell. Run `validate_move` first. |
| Solves are slow in your own scripts | Load dictionaries with `load_dictionary` or the registry so `gc.freeze()` runs. Avoid creating thousands of other long-lived objects between solves. |
