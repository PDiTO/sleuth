"""Model access through headless agent CLIs, with every call recorded to disk.

sleuth has no API keys. It shells out to `claude -p` and `codex exec`, like retro does,
and turns everything off that would let a model do more than chat. A model that could
run code could enumerate the DSL and brute-force the box, so this matters more here than
usual.

- **claude**: `--system-prompt` replaces Claude Code's own system prompt, `--tools ""`
  removes every built-in tool, `--strict-mcp-config` with no config loads no MCP servers,
  `--setting-sources ""` skips user and project settings (and with them hooks and
  plugins), and `--disable-slash-commands` turns off skills. It runs in an empty temporary
  directory, so there is no project CLAUDE.md to find.
- **codex**: runs with an isolated `CODEX_HOME` holding only the login and the model
  provider from your own config, with every optional feature off (shell, unified exec,
  apps, browser, computer use, image tools, multi-agent, plugins, skills, hooks, memories),
  web search disabled, AGENTS.md loading off and a read-only sandbox in an empty
  directory. `model_instructions_file` replaces Codex's base instructions with ours. The
  event stream is checked for tool calls, and any that appear are recorded.

`Recorder` wraps a provider. Each call is keyed by a hash of the provider id, system
prompt and prompt, and saved with the reply in one JSON file per episode. Run again and
recorded calls come from disk. With no inner provider it is a pure replay that fails on
any miss, which is how the report rebuilds every number offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ProviderError(RuntimeError):
    pass


class ReplayMiss(ProviderError):
    pass


@dataclass(frozen=True)
class Reply:
    text: str
    seconds: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Completed:
    returncode: int
    stdout: str
    stderr: str


Exec = Callable[[list[str], Path, float, dict[str, str] | None], Completed]


def _subprocess(
    args: list[str], cwd: Path, timeout: float, env: dict[str, str] | None
) -> Completed:
    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        stdin=subprocess.DEVNULL,
        env=env,
    )
    return Completed(proc.returncode, proc.stdout, proc.stderr)


class Provider(ABC):
    @property
    @abstractmethod
    def id(self) -> str: ...

    @abstractmethod
    def complete(self, system: str, prompt: str) -> Reply: ...


@dataclass
class Claude(Provider):
    model: str
    effort: str | None = "medium"
    timeout: float = 600
    exec: Exec = field(default=_subprocess, repr=False)
    binary: str = "claude"

    @property
    def id(self) -> str:
        return f"claude/{self.model}" + (f"@{self.effort}" if self.effort else "")

    def args(self, system: str, prompt: str) -> list[str]:
        args = [
            self.binary,
            "-p",
            prompt,
            "--system-prompt",
            system,
            "--tools",
            "",
            "--strict-mcp-config",
            "--setting-sources",
            "",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format",
            "json",
            "--model",
            self.model,
        ]
        if self.effort:
            args += ["--effort", self.effort]
        return args

    def complete(self, system: str, prompt: str) -> Reply:
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="sleuth-claude-") as tmp:
            done = _run(self.exec, self.args(system, prompt), Path(tmp), self.timeout, None)
        out = _last_json(done.stdout)
        if out.get("is_error") or out.get("subtype") != "success":
            raise ProviderError(f"claude error: {str(out.get('result'))[:300]}")
        usage = out.get("usage") or {}
        meta = {
            "cost_usd": out.get("total_cost_usd"),
            "num_turns": out.get("num_turns"),
            "stop_reason": out.get("stop_reason"),
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "thinking_tokens": (usage.get("output_tokens_details") or {}).get("thinking_tokens"),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "server_tool_use": usage.get("server_tool_use"),
            "permission_denials": out.get("permission_denials"),
            "models": sorted((out.get("modelUsage") or {}).keys()),
            "tool_calls": 0,
        }
        return Reply(str(out.get("result") or ""), time.monotonic() - start, meta)


CODEX_FEATURES_OFF = (
    "shell_tool",
    "unified_exec",
    "apps",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "image_generation",
    "multi_agent",
    "multi_agent_v2",
    "plugins",
    "remote_plugin",
    "view_image",
    "code_mode_host",
    "skill_search",
    "skill_mcp_dependency_install",
    "tool_suggest",
    "hooks",
    "goals",
    "memories",
    "in_app_browser",
    "sleep_tool",
    "shell_snapshot",
    "workspace_dependencies",
    "plugin_discovery_prefetch",
    "recommended_plugins",
)

_codex_home_lock = threading.Lock()


def _toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int | float):
        return str(v)
    if isinstance(v, str):
        return json.dumps(v)
    raise ValueError(f"unsupported TOML value {v!r}")


def codex_config(user_config: dict[str, Any]) -> str:
    """A minimal config.toml: the user's model provider, everything optional off."""
    lines = ['web_search = "disabled"', "project_doc_max_bytes = 0"]
    provider = user_config.get("model_provider")
    providers = user_config.get("model_providers") or {}
    if isinstance(provider, str):
        lines.insert(0, f"model_provider = {_toml_value(provider)}")
        if provider in providers:
            lines.append(f"[model_providers.{provider}]")
            for k, v in providers[provider].items():
                if isinstance(v, bool | int | float | str):
                    lines.append(f"{k} = {_toml_value(v)}")
    lines.append("[features]")
    lines += [f"{name} = false" for name in CODEX_FEATURES_OFF]
    return "\n".join(lines) + "\n"


def isolated_codex_home(root: Path | None = None, source: Path | None = None) -> Path:
    """Build `~/.cache/sleuth/codex-home`: your login and model provider, nothing else."""
    src = source or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
    home = root or Path.home() / ".cache" / "sleuth" / "codex-home"
    with _codex_home_lock:
        home.mkdir(parents=True, exist_ok=True)
        auth = src / "auth.json"
        if auth.exists():
            shutil.copy2(auth, home / "auth.json")
        cfg_path = src / "config.toml"
        cfg = tomllib.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        (home / "config.toml").write_text(codex_config(cfg))
        os.chmod(home / "config.toml", 0o600)
    return home


@dataclass
class Codex(Provider):
    model: str
    effort: str = "medium"
    timeout: float = 600
    exec: Exec = field(default=_subprocess, repr=False)
    binary: str = "codex"
    home: Path | None = None

    @property
    def id(self) -> str:
        return f"codex/{self.model}@{self.effort}"

    def args(self, prompt: str, scratch: Path) -> list[str]:
        return [
            self.binary,
            "exec",
            "--skip-git-repo-check",
            "--ephemeral",
            "--json",
            "-s",
            "read-only",
            "-C",
            str(scratch),
            "-c",
            f"model_instructions_file={json.dumps(str(scratch / 'system.md'))}",
            "-c",
            f"model_reasoning_effort={json.dumps(self.effort)}",
            "-m",
            self.model,
            prompt,
        ]

    def complete(self, system: str, prompt: str) -> Reply:
        if self.home is None:
            self.home = isolated_codex_home()
        env = {**os.environ, "CODEX_HOME": str(self.home)}
        start = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="sleuth-codex-") as tmp:
            scratch = Path(tmp)
            (scratch / "system.md").write_text(system)
            done = _run(self.exec, self.args(prompt, scratch), scratch, self.timeout, env)
        text, meta = parse_codex_events(done.stdout)
        return Reply(text, time.monotonic() - start, meta)


def parse_codex_events(stdout: str) -> tuple[str, dict[str, Any]]:
    messages: list[str] = []
    tools: list[str] = []
    usage: dict[str, Any] = {}
    errors: list[str] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "turn.failed":
            msg = (event.get("error") or {}).get("message")
            raise ProviderError(f"codex turn failed: {str(msg)[:300]}")
        if kind == "error":
            errors.append(str(event.get("message"))[:300])
        if kind == "turn.completed":
            usage = event.get("usage") or {}
        item = event.get("item")
        if kind == "item.completed" and isinstance(item, dict):
            itype = item.get("type")
            if itype == "agent_message":
                messages.append(str(item.get("text", "")))
            elif itype not in ("reasoning", "error"):
                tools.append(str(itype))
    if not messages:
        raise ProviderError(f"codex returned no agent message: {' | '.join(errors)[:300]}")
    meta = {
        "input_tokens": usage.get("input_tokens"),
        "cached_input_tokens": usage.get("cached_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
        "tool_calls": len(tools),
        "tool_types": tools,
        "errors": errors,
    }
    return messages[-1], meta


def _run(
    exec: Exec, args: list[str], cwd: Path, timeout: float, env: dict[str, str] | None
) -> Completed:
    try:
        done = exec(args, cwd, timeout, env)
    except subprocess.TimeoutExpired as e:
        raise ProviderError(f"{args[0]} timed out after {timeout:.0f}s") from e
    except FileNotFoundError as e:
        raise ProviderError(f"{args[0]} is not installed") from e
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        raise ProviderError(f"{args[0]} exited {done.returncode}: {' | '.join(tail)}")
    return done


def _last_json(stdout: str) -> dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ProviderError("expected a JSON object on stdout")


@dataclass
class Scripted(Provider):
    """A fake provider for tests: replies come from a function of the prompt."""

    reply: Callable[[str, str], str]
    name: str = "scripted"
    calls: int = 0

    @property
    def id(self) -> str:
        return self.name

    def complete(self, system: str, prompt: str) -> Reply:
        self.calls += 1
        return Reply(self.reply(system, prompt), 0.0, {"tool_calls": 0})


def call_key(provider_id: str, system: str, prompt: str) -> str:
    blob = json.dumps([provider_id, system, prompt], ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class Recorder(Provider):
    """Record every call of `inner` to `path`, or replay from it when `inner` is None."""

    def __init__(self, path: Path, provider_id: str, inner: Provider | None = None) -> None:
        if inner is not None and inner.id != provider_id:
            raise ValueError(f"provider id mismatch: {inner.id} != {provider_id}")
        self.path = path
        self._id = provider_id
        self.inner = inner
        self.data: dict[str, Any] = {"provider": provider_id, "system": None, "calls": []}
        if path.exists():
            self.data = json.loads(path.read_text())
        self._by_key = {c["key"]: c for c in self.data["calls"]}
        self.hits = 0
        self.misses = 0

    @property
    def id(self) -> str:
        return self._id

    def complete(self, system: str, prompt: str) -> Reply:
        key = call_key(self._id, system, prompt)
        hit = self._by_key.get(key)
        if hit is not None:
            self.hits += 1
            r = hit["reply"]
            return Reply(r["text"], r.get("seconds", 0.0), r.get("meta", {}))
        if self.inner is None:
            raise ReplayMiss(f"no recorded reply for call {key[:12]} in {self.path}")
        reply = self.inner.complete(system, prompt)
        self.misses += 1
        record = {
            "key": key,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "prompt": prompt,
            "reply": {"text": reply.text, "seconds": round(reply.seconds, 2), "meta": reply.meta},
        }
        if self.data.get("system") is None:
            self.data["system"] = system
        elif self.data["system"] != system:
            record["system"] = system
        self.data["calls"].append(record)
        self._by_key[key] = record
        self.save()
        return reply

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False) + "\n")
        tmp.replace(self.path)


def from_spec(spec: str) -> Provider:
    """`claude:claude-sonnet-5-5`, `claude:claude-opus-5-5@high`, `codex:gpt-6.1-sol`."""
    backend, _, rest = spec.partition(":")
    model, _, effort = rest.partition("@")
    if not model:
        raise ValueError(f"bad model spec {spec!r}; expected backend:model[@effort]")
    if backend == "claude":
        return Claude(model, effort or "medium")
    if backend == "codex":
        return Codex(model, effort or "medium")
    raise ValueError(f"unknown backend {backend!r}; choose claude or codex")


def slug(provider_id: str) -> str:
    return provider_id.replace("/", "_").replace("@", "_")
