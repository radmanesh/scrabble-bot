"""Word list as a trie, cached on disk as a pickle keyed by the source file's hash."""

from __future__ import annotations

import gc
import hashlib
import pickle
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

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
