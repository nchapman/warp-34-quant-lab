"""Hermetic Hugging Face Hub cache resolution honoring pinned revisions."""

from __future__ import annotations

import os
from pathlib import Path


def _cache_dir() -> Path:
    direct = os.environ.get("HF_HUB_CACHE")
    if direct:
        return Path(direct).expanduser()
    home = os.environ.get("HF_HOME")
    return Path(home if home else "~/.cache/huggingface").expanduser() / "hub"


class StaleCacheError(RuntimeError):
    """Cache ref file is corrupt or empty."""


def _bare_sha(s: str) -> bool:
    return len(s) == 40 and all(c in "0123456789abcdef" for c in s.lower())


def cached_snapshot_dir(hub_id: str, revision: str | None = None) -> str:
    base = _cache_dir() / ("models--" + hub_id.replace("/", "--"))
    if revision:
        if Path(revision).name != revision or revision in (".", ".."):
            raise RuntimeError(
                f"revision {revision!r} is not a single safe path segment — refusing to resolve it"
            )
        if _bare_sha(revision):
            got = base / "snapshots" / revision.lower()
        else:
            ref_file = base / "refs" / revision
            ref = ref_file.read_text(encoding="utf-8").strip() if ref_file.is_file() else ""
            if not _bare_sha(ref):
                raise StaleCacheError(
                    f"stale/invalid ref {ref_file} for {hub_id} (content {ref[:40]!r}) — "
                    "cache corrupt; re-run with network (HF_HUB_OFFLINE=0)"
                )
            got = base / "snapshots" / ref
        if got.resolve().is_dir():
            return str(got.resolve())
    else:
        ref_file = base / "refs" / "main"
        if ref_file.is_file():
            ref = ref_file.read_text(encoding="utf-8").strip()
            if _bare_sha(ref) and (base / "snapshots" / ref).resolve().is_dir():
                return str((base / "snapshots" / ref).resolve())
    raise RuntimeError(
        f"no ref-resolvable local snapshot for {hub_id} @ {revision or 'main'} "
        f"(cache incomplete at {base}); re-run with network (HF_HUB_OFFLINE=0)"
    )


def resolve_snapshot(hub_id: str, revision: str | None = None, offline: bool = False) -> str:
    if Path(hub_id).is_dir():
        return str(Path(hub_id).resolve())

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import IncompleteSnapshotError

    if not offline:
        offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    try:
        return snapshot_download(hub_id, revision=revision, local_files_only=offline)
    except IncompleteSnapshotError as e:
        snap = getattr(e, "snapshot_path", None)
        if snap and Path(snap).is_dir():
            return str(snap)
        return cached_snapshot_dir(hub_id, revision)
