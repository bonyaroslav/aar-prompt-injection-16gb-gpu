"""Small state machine enforcing held-out result secrecy."""
from __future__ import annotations
import hashlib, json, os
from pathlib import Path

class HeldOutSealer:
    def __init__(self, root: str | Path):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        try: os.chmod(self.root, 0o700)
        except OSError: pass
        self.state_file = self.root / "state.json"
        if not self.state_file.exists(): self._write({"state": "SEALED", "candidates": None, "validity": None})

    def _read(self): return json.loads(self.state_file.read_text(encoding="utf-8"))
    def _write(self, value):
        tmp = self.state_file.with_suffix(".tmp"); tmp.write_text(json.dumps(value, sort_keys=True), encoding="utf-8"); os.replace(tmp, self.state_file)
    @staticmethod
    def digest(value) -> str:
        raw = value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()
    def freeze(self, candidate_ids, validity_rules: str):
        s = self._read()
        if s["candidates"] is not None: raise RuntimeError("candidate commitment already frozen")
        s.update(candidates=self.digest(sorted(candidate_ids)), validity=self.digest(validity_rules)); self._write(s)
    def store_receipt(self, label: str, payload: bytes, valid: int, invalid: int):
        if self._read()["state"] != "SEALED": raise RuntimeError("results may only be stored while sealed")
        path = self.root / f"{label}.blob"; path.write_bytes(payload)
        try: os.chmod(path, 0o600)
        except OSError: pass
        return {"label": label, "digest": self.digest(payload), "valid": valid, "invalid": invalid}
    def authorize(self, selection_record: dict):
        s = self._read()
        if s["candidates"] is None or not selection_record.get("finalized"): raise PermissionError("selection is not finalized")
        s["state"] = "AUTHORIZED"; s["selection_digest"] = self.digest(selection_record); self._write(s)
    def reveal(self, selection_record: dict):
        s = self._read()
        if s["state"] != "AUTHORIZED" or s.get("selection_digest") != self.digest(selection_record): raise PermissionError("held-out reveal is unauthorized")
        return {"baseline": (self.root / "baseline.blob").read_bytes(), "trained": (self.root / "trained.blob").read_bytes()}
