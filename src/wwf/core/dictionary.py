"""Word list as a trie, cached on disk as a pickle keyed by the source file's hash."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

CACHE_VERSION = 2


class TrieNode:
    __slots__ = ("children", "terminal")

    def __init__(self) -> None:
        self.children: dict[str, TrieNode] = {}
        self.terminal: bool = False


class Dictionary(Protocol):
    def contains(self, word: str) -> bool: ...
    def root(self) -> TrieNode: ...
    def __len__(self) -> int: ...


class TrieDictionary:
    """Trie plus a set for O(1) membership checks."""

    def __init__(self, words: Iterable[str]) -> None:
        self._root = TrieNode()
        self._words: set[str] = set()
        for raw in words:
            w = raw.strip().upper()
            if len(w) < 2 or not w.isascii() or not w.isalpha():
                continue  # single letters are not words; skip junk lines
            if w in self._words:
                continue
            self._words.add(w)
            node = self._root
            for ch in w:
                nxt = node.children.get(ch)
                if nxt is None:
                    nxt = node.children[ch] = TrieNode()
                node = nxt
            node.terminal = True

    def contains(self, word: str) -> bool:
        return word.upper() in self._words

    def root(self) -> TrieNode:
        return self._root

    def __len__(self) -> int:
        return len(self._words)

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and self.contains(word)


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load_dictionary(path: Path, cache_dir: Path | None = None) -> TrieDictionary:
    """Load a newline-separated word list. Case is normalised to uppercase.

    The result is moved out of the garbage collector's view (gc.freeze): the trie is several
    hundred thousand long-lived objects, and rescanning them on every full collection otherwise
    roughly doubles move-generation time.
    """
    dictionary = _load(path, cache_dir)
    gc.freeze()
    return dictionary


def _load(path: Path, cache_dir: Path | None) -> TrieDictionary:
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{path.stem}.{_file_hash(path)}.v{CACHE_VERSION}.pickle"
        if cache_path.exists():
            try:
                with cache_path.open("rb") as f:
                    cached = pickle.load(f)
                if isinstance(cached, TrieDictionary):
                    return cached
            except (OSError, pickle.UnpicklingError, EOFError, AttributeError):
                pass  # corrupt or stale cache: rebuild below

    with path.open(encoding="utf-8") as f:
        dictionary = TrieDictionary(f)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump(dictionary, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    return dictionary


# --- choosing between word lists ------------------------------------------------------------


class DictionaryError(ValueError):
    """config/dictionaries.json is malformed or names a missing file."""


@dataclass(frozen=True)
class DictionaryInfo:
    id: str
    label: str
    description: str
    path: Path


class DictionaryRegistry:
    """The word lists the user can choose from, loaded lazily and kept once loaded.

    Entries come from config/dictionaries.json; any other *.txt file in the dictionaries folder
    is offered too, under its file stem, so dropping in a word list is enough to make it
    selectable. The WWF_DICTIONARY environment variable overrides the configured default.
    """

    def __init__(
        self,
        infos: list[DictionaryInfo],
        default_id: str,
        cache_dir: Path | None = None,
    ) -> None:
        self._infos = {i.id: i for i in infos}
        if default_id not in self._infos:
            raise DictionaryError(
                f"default dictionary {default_id!r} is not available (have {sorted(self._infos)})"
            )
        self.default_id = default_id
        self._cache_dir = cache_dir
        self._loaded: dict[str, TrieDictionary] = {}
        self._lock = threading.Lock()

    def infos(self) -> list[DictionaryInfo]:
        return list(self._infos.values())

    def info(self, dictionary_id: str) -> DictionaryInfo:
        return self._infos[dictionary_id]

    def is_loaded(self, dictionary_id: str) -> bool:
        return dictionary_id in self._loaded

    def get(self, dictionary_id: str | None = None) -> TrieDictionary:
        """The dictionary with this id (the default if None). KeyError for unknown ids."""
        did = dictionary_id or self.default_id
        info = self._infos[did]
        with self._lock:
            if did not in self._loaded:
                self._loaded[did] = load_dictionary(info.path, self._cache_dir)
            return self._loaded[did]


def load_dictionary_registry(
    config_path: Path, dictionaries_dir: Path, cache_dir: Path | None = None
) -> DictionaryRegistry:
    data: dict[str, Any] = {}
    if config_path.exists():
        with config_path.open() as f:
            data = json.load(f)

    infos: list[DictionaryInfo] = []
    listed_files: set[str] = set()
    for entry in data.get("dictionaries", []):
        if "id" not in entry or "file" not in entry:
            raise DictionaryError(f"dictionary entry needs 'id' and 'file': {entry}")
        path = dictionaries_dir / str(entry["file"])
        if not path.is_file():
            raise DictionaryError(f"dictionary {entry['id']!r}: file {path} not found")
        if any(i.id == entry["id"] for i in infos):
            raise DictionaryError(f"dictionary id {entry['id']!r} listed twice")
        listed_files.add(path.name)
        infos.append(
            DictionaryInfo(
                id=str(entry["id"]),
                label=str(entry.get("label", entry["id"])),
                description=str(entry.get("description", "")),
                path=path,
            )
        )
    taken = {i.id for i in infos}
    for path in sorted(dictionaries_dir.glob("*.txt")):
        if path.name in listed_files or path.stem in taken:
            continue
        infos.append(DictionaryInfo(path.stem, path.name, "Unlisted word list", path))
        taken.add(path.stem)

    if not infos:
        raise DictionaryError(f"no word lists found in {dictionaries_dir}")
    default_id = os.environ.get("WWF_DICTIONARY") or str(data.get("default", infos[0].id))
    return DictionaryRegistry(infos, default_id, cache_dir)


def default_dictionary_registry() -> DictionaryRegistry:
    """Registry built from config/dictionaries.json and assets/dictionaries/."""
    from wwf.core.config import cache_dir, config_dir, dictionaries_dir

    return load_dictionary_registry(
        config_dir() / "dictionaries.json", dictionaries_dir(), cache_dir()
    )
