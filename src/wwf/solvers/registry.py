"""Discovers Solver classes in this package and cross-checks them against config/solvers.json."""

from __future__ import annotations

import importlib
import inspect
import json
import pkgutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from wwf.core.config import config_dir
from wwf.solvers.base import Solver, SolverParam

_SKIP_MODULES = frozenset({"base", "registry"})


class RegistryError(ValueError):
    """solvers.json and the solver implementations disagree."""


@dataclass(frozen=True)
class SolverInfo:
    id: str
    display_name: str
    description: str
    enabled: bool
    default: bool
    params: tuple[SolverParam, ...]

    @property
    def planned(self) -> bool:
        return not self.enabled


def discover_solvers(package: str = "wwf.solvers") -> dict[str, type[Solver]]:
    """Import every module in `package` and collect classes that look like a Solver."""
    pkg = importlib.import_module(package)
    found: dict[str, type[Solver]] = {}
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        if mod_info.name in _SKIP_MODULES or mod_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package}.{mod_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module.__name__:
                continue
            if not all(hasattr(cls, a) for a in ("id", "display_name", "params", "solve")):
                continue
            solver_cls = cast("type[Solver]", cls)
            solver_id = solver_cls.id
            if solver_id in found:
                raise RegistryError(f"duplicate solver id {solver_id!r}")
            found[solver_id] = solver_cls
    return found


class SolverRegistry:
    def __init__(self, entries: list[dict[str, Any]], implementations: dict[str, type[Solver]]):
        self._instances: dict[str, Solver] = {}
        self._infos: list[SolverInfo] = []
        seen: set[str] = set()
        defaults: list[str] = []
        for entry in entries:
            if "id" not in entry:
                raise RegistryError(f"solvers.json entry without an id: {entry}")
            sid = str(entry["id"])
            if sid in seen:
                raise RegistryError(f"solvers.json lists {sid!r} twice")
            seen.add(sid)
            enabled = bool(entry.get("enabled", True))
            impl = implementations.get(sid)
            if impl is None and enabled:
                raise RegistryError(
                    f"solvers.json enables unknown solver id {sid!r}; no implementation in "
                    f"wwf/solvers/ (known: {sorted(implementations)})"
                )
            is_default = bool(entry.get("default", False))
            if is_default:
                if not enabled:
                    raise RegistryError(f"default solver {sid!r} is not enabled")
                defaults.append(sid)
            if impl is not None:
                instance = impl()
                if enabled:
                    self._instances[sid] = instance
                info = SolverInfo(
                    id=sid,
                    display_name=str(entry.get("display_name", instance.display_name)),
                    description=str(entry.get("description", instance.description)),
                    enabled=enabled,
                    default=is_default,
                    params=tuple(instance.params),
                )
            else:
                info = SolverInfo(
                    id=sid,
                    display_name=str(entry.get("display_name", sid)),
                    description=str(entry.get("description", "")),
                    enabled=False,
                    default=False,
                    params=(),
                )
            self._infos.append(info)
        if len(defaults) > 1:
            raise RegistryError(f"more than one default solver: {defaults}")
        if not self._instances:
            raise RegistryError("no enabled solvers")
        self._default = defaults[0] if defaults else next(iter(self._instances))

    @property
    def default_id(self) -> str:
        return self._default

    def infos(self) -> list[SolverInfo]:
        return list(self._infos)

    def info(self, solver_id: str) -> SolverInfo:
        for i in self._infos:
            if i.id == solver_id:
                return i
        raise KeyError(solver_id)

    def get(self, solver_id: str) -> Solver:
        """An enabled solver instance. KeyError for unknown or disabled ids."""
        return self._instances[solver_id]


def load_registry(path: Path | None = None, package: str = "wwf.solvers") -> SolverRegistry:
    path = path or config_dir() / "solvers.json"
    with path.open() as f:
        data = json.load(f)
    entries = data.get("solvers")
    if not isinstance(entries, list):
        raise RegistryError(f"{path} must contain a 'solvers' list")
    return SolverRegistry(entries, discover_solvers(package))
