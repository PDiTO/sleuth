from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from sleuth import providers
from sleuth.providers import (
    Claude,
    Codex,
    Completed,
    ProviderError,
    Recorder,
    ReplayMiss,
    Scripted,
    codex_config,
    from_spec,
    isolated_codex_home,
    parse_codex_events,
    slug,
)

CLAUDE_OK = json.dumps(
    {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": '{"hypothesis": "h", "commit": true}',
        "total_cost_usd": 0.01,
        "num_turns": 1,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 5,
            "output_tokens_details": {"thinking_tokens": 3},
        },
        "modelUsage": {"claude-haiku-5-5": {}},
        "permission_denials": [],
    }
)

CODEX_OK = "\n".join(
    [
        json.dumps({"type": "thread.started", "thread_id": "t"}),
        json.dumps({"type": "item.completed", "item": {"type": "error", "message": "x"}}),
        json.dumps({"type": "item.completed", "item": {"type": "reasoning", "text": "hm"}}),
        json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": '{"predictions": [1]}'},
            }
        ),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 9}}),
    ]
)


def test_claude_turns_everything_off() -> None:
    seen: list[list[str]] = []

    def fake(args: list[str], cwd: Path, timeout: float, env: dict[str, str] | None) -> Completed:
        seen.append(args)
        assert cwd.exists()
        return Completed(0, "warning line\n" + CLAUDE_OK, "")

    c = Claude("claude-haiku-5-5", exec=fake)
    reply = c.complete("SYSTEM", "PROMPT")
    args = seen[0]
    assert args[args.index("--tools") + 1] == ""
    assert args[args.index("--system-prompt") + 1] == "SYSTEM"
    assert args[args.index("--setting-sources") + 1] == ""
    for flag in ("--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
        assert flag in args
    assert args[args.index("--effort") + 1] == "medium"
    assert reply.text.startswith('{"hypothesis"')
    assert reply.meta["thinking_tokens"] == 3
    assert reply.meta["tool_calls"] == 0
    assert c.id == "claude/claude-haiku-5-5@medium"


def test_claude_errors() -> None:
    def err(out: str, code: int = 0) -> providers.Exec:
        return lambda a, c, t, e: Completed(code, out, "bad things")

    with pytest.raises(ProviderError, match="claude error"):
        Claude("m", exec=err(json.dumps({"is_error": True, "result": "rate"}))).complete("s", "p")
    with pytest.raises(ProviderError, match="exited 1"):
        Claude("m", exec=err("", 1)).complete("s", "p")
    with pytest.raises(ProviderError, match="expected a JSON"):
        Claude("m", exec=err("not json")).complete("s", "p")

    def timeout(a: list[str], c: Path, t: float, e: dict[str, str] | None) -> Completed:
        raise subprocess.TimeoutExpired(a, t)

    with pytest.raises(ProviderError, match="timed out"):
        Claude("m", exec=timeout).complete("s", "p")

    def missing(a: list[str], c: Path, t: float, e: dict[str, str] | None) -> Completed:
        raise FileNotFoundError(a[0])

    with pytest.raises(ProviderError, match="not installed"):
        Claude("m", exec=missing).complete("s", "p")


def test_codex_isolated_home_and_args(tmp_path: Path) -> None:
    src = tmp_path / "codex"
    src.mkdir()
    (src / "auth.json").write_text("{}")
    (src / "config.toml").write_text(
        'model = "x"\nmodel_provider = "proxy"\nsandbox_mode = "danger-full-access"\n'
        '[mcp_servers.browser]\ncommand = "b"\n'
        '[model_providers.proxy]\nname = "P"\nbase_url = "https://p"\nrequires_openai_auth = true\n'
    )
    home = isolated_codex_home(tmp_path / "home", src)
    cfg = (home / "config.toml").read_text()
    assert 'model_provider = "proxy"' in cfg
    assert 'base_url = "https://p"' in cfg
    assert "mcp_servers" not in cfg
    assert "danger-full-access" not in cfg
    assert "shell_tool = false" in cfg
    assert 'web_search = "disabled"' in cfg
    assert (home / "auth.json").exists()

    seen: list[tuple[list[str], dict[str, str] | None, str]] = []

    def fake(args: list[str], cwd: Path, timeout: float, env: dict[str, str] | None) -> Completed:
        seen.append((args, env, (cwd / "system.md").read_text()))
        return Completed(0, CODEX_OK, "")

    c = Codex("gpt-6.1-sol", exec=fake, home=home)
    reply = c.complete("SYSTEM", "PROMPT")
    args, env, system = seen[0]
    assert system == "SYSTEM"
    assert env is not None
    assert env["CODEX_HOME"] == str(home)
    assert args[-1] == "PROMPT"
    assert "read-only" in args
    assert any(a.startswith("model_instructions_file=") for a in args)
    assert reply.text == '{"predictions": [1]}'
    assert reply.meta["tool_calls"] == 0
    assert c.id == "codex/gpt-6.1-sol@medium"


def test_codex_config_without_provider() -> None:
    cfg = codex_config({})
    assert "model_provider" not in cfg
    assert "[features]" in cfg


def test_codex_event_parsing_counts_tools_and_fails_loudly() -> None:
    tool = json.dumps({"type": "item.completed", "item": {"type": "command_execution"}})
    _, meta = parse_codex_events(tool + "\n" + CODEX_OK)
    assert meta["tool_calls"] == 1
    assert meta["tool_types"] == ["command_execution"]
    failed = json.dumps({"type": "turn.failed", "error": {"message": "unsupported model"}})
    with pytest.raises(ProviderError, match="unsupported model"):
        parse_codex_events(failed)
    retry = json.dumps({"type": "error", "message": "Reconnecting 1/5"})
    assert parse_codex_events(retry + "\nnot json\n[1]\n" + CODEX_OK)[1]["errors"]
    with pytest.raises(ProviderError, match="no agent message"):
        parse_codex_events(retry)


def test_recorder_records_then_replays(tmp_path: Path) -> None:
    inner = Scripted(lambda s, p: p.upper(), name="fake")
    path = tmp_path / "ep.json"
    rec = Recorder(path, "fake", inner)
    assert rec.complete("sys", "hello").text == "HELLO"
    assert rec.complete("sys", "hello").text == "HELLO"
    rec.complete("other system", "x")
    assert inner.calls == 2
    assert (rec.hits, rec.misses) == (1, 2)
    data = json.loads(path.read_text())
    assert data["system"] == "sys"
    assert data["calls"][1]["system"] == "other system"

    replay = Recorder(path, "fake")
    assert replay.complete("sys", "hello").text == "HELLO"
    with pytest.raises(ReplayMiss):
        replay.complete("sys", "never asked")
    with pytest.raises(ValueError, match="mismatch"):
        Recorder(path, "other", inner)


def test_specs_and_slugs() -> None:
    p = from_spec("claude:claude-opus-5-5@high")
    assert isinstance(p, Claude)
    assert p.effort == "high"
    q = from_spec("codex:gpt-6.1-sol")
    assert isinstance(q, Codex)
    assert q.effort == "medium"
    with pytest.raises(ValueError, match="unknown backend"):
        from_spec("gemini:x")
    with pytest.raises(ValueError, match="bad model spec"):
        from_spec("claude")
    assert slug("claude/claude-opus-5-5@medium") == "claude_claude-opus-5-5_medium"
