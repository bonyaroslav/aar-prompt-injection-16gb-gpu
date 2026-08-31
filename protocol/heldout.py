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
        if not self.state_file.exists():
            self._write({"state": "SEALED", "candidates": None, "validity": None, "receipts": {}})

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
    def commitments(self) -> dict:
        """Public digests only (never plaintext candidate IDs or validity rules)."""
        s = self._read()
        return {"state": s["state"], "candidates": s["candidates"], "validity": s["validity"]}
    def store_receipt(self, label: str, payload: bytes, valid: int, invalid: int):
        s = self._read()
        if s["state"] not in {"SEALED", "AUTHORIZED"}:
            raise RuntimeError("results may only be stored before reveal completion")
        receipt = {"label": label, "digest": self.digest(payload), "valid": valid, "invalid": invalid}
        existing = self.receipt(label)
        if existing is not None:
            if existing != receipt:
                raise RuntimeError(f"receipt already sealed with different content: {label}")
            return existing
        path = self.root / f"{label}.blob"; path.write_bytes(payload)
        try: os.chmod(path, 0o600)
        except OSError: pass
        s.setdefault("receipts", {})[label] = receipt
        self._write(s)
        return receipt
    def receipt(self, label: str):
        """Return public receipt metadata only; never load the sealed blob."""
        return self._read().get("receipts", {}).get(label)
    def authorize(self, selection_record: dict, authorization_identity: str = "default"):
        s = self._read()
        if s["candidates"] is None or not selection_record.get("finalized"): raise PermissionError("selection is not finalized")
        selection_digest = self.digest(selection_record)
        if s["state"] == "AUTHORIZED":
            if (s.get("selection_digest") != selection_digest or
                    s.get("authorization_identity", "default") != authorization_identity):
                raise PermissionError("held-out authorization identity or selection changed")
            return
        if s["state"] != "SEALED":
            raise PermissionError("held-out authorization is no longer available")
        s.update(
            state="AUTHORIZED", selection_digest=selection_digest,
            authorization_identity=authorization_identity,
        )
        self._write(s)
    def reveal(self, selection_record: dict):
        s = self._read()
        if s["state"] != "AUTHORIZED" or s.get("selection_digest") != self.digest(selection_record): raise PermissionError("held-out reveal is unauthorized")
        return {"baseline": (self.root / "baseline.blob").read_bytes(), "trained": (self.root / "trained.blob").read_bytes()}
