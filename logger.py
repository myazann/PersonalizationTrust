import os
import json, uuid, pathlib, asyncio, datetime, re

DATA_DIR = pathlib.Path(os.getenv("APP_DATA_DIR", "./user_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

_pid_re = re.compile(r"[^A-Za-z0-9._-]+")

def _slugify(pid: str) -> str:
    pid = (pid or "").strip().lower()
    pid = _pid_re.sub("_", pid)
    return pid or f"anon_{uuid.uuid4().hex[:8]}"

def _pid_log_path(pid: str) -> pathlib.Path:
    return DATA_DIR / f"{_slugify(pid)}.jsonl"

def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat() + "Z"

async def _append_jsonl(path: pathlib.Path, obj: dict):

    line = json.dumps(obj, ensure_ascii=False)
    def _write():
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    await asyncio.to_thread(_write)

async def log_event(pid: str, kind: str, payload: dict):
    record = {
        "ts": _utc_now(),
        "pid": pid,
        "kind": kind,
        **payload,
    }
    await _append_jsonl(_pid_log_path(pid), record)

def load_chat_history(pid: str):
    """
    Return a list of role/content dictionaries representing the conversation
    history for the provided pid. Only chat events are returned.
    """
    path = _pid_log_path(pid)
    if not path.exists():
        return []

    history = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            kind = entry.get("kind")
            text = entry.get("text")
            if not text:
                continue

            if kind == "chat_user":
                history.append({"role": "user", "content": text})
            elif kind == "chat_assistant":
                history.append({"role": "assistant", "content": text})

    return history
