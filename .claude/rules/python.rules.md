# Python rules for opensre

Concrete project conventions. Match the existing code; don't invent new styles.
Every rule cites a config file or a real example so you can verify the pattern in
context. When in doubt, grep for "how does the rest of the codebase do this?".

## 1. Scope & tooling

- Python **3.13** target. `ruff.toml:4` (`target-version = "py313"`), `mypy.ini:2` (`python_version = 3.13`).
- Line length **100**. Long lines are the formatter's problem (`E501` is ignored).
- Three gates before any push: `make test-cov`, `make lint`, `make typecheck`. Defined in `Makefile`; enforced in `AGENTS.md` "Before Push".
- Active ruff lint rules: `E, W, F, I, B, C4, UP, ARG, SIM`. Ignored: `E501, B008, B905, SIM108`. See `ruff.toml:20` and `:32`.
- Per-file ruff exemptions live under `[lint.per-file-ignores]` in `ruff.toml`. Tests already get `F401, ARG001, E402`.
- mypy `enable_incomplete_feature = NewGenericSyntax` is on but not all PEP 695 forms parse cleanly — see §3.

## 2. Type hints

- Always modern: `dict[str, Any]`, `list[T]`, `tuple[X, Y]`, `X | None`. Never `Dict`, `List`, `Tuple`, `Optional` from `typing`.
- `from __future__ import annotations` is used **selectively** when you need forward refs or want to delay evaluation. It is not required at the top of every file. Examples that have it: `app/services/llm_client.py:7`, `app/pipeline/driver.py:23`. Examples that don't: `app/main.py`.
- `TypedDict` comes from `typing_extensions`, not `typing`, to keep PEP 728 fields available (`app/state/agent_state.py:14`).
- Use `Literal[...]` for closed string sets (`app/types/tools.py`).
- Annotate every public function. Internal helpers may skip annotations only if the function is one expression long; mypy is configured with `disallow_untyped_defs = False` so it won't yell, but reviewers will.

## 3. Generics

- Mypy 1.20 doesn't accept PEP 695 `def fn[T](...)` syntax in every position, but ruff `UP047` (under target `py313`) wants it. The repo resolves this conflict with `TypeVar` + a `# noqa: UP047` on the `def` line.
- Examples: `app/tools/tool_decorator.py:38, 59, 79`; `app/services/structured_llm.py:35`; `app/tools/utils/compaction.py:23`.
- Don't add a fresh PEP 695 generic without first running `make typecheck`. If mypy complains, fall back to the noqa pattern.

## 4. Dataclasses vs Pydantic

- **Frozen dataclass** for plain DTOs / response wrappers / claims that don't need runtime validation.
  - `app/auth/jwt_auth.py:29` (`JWTClaims`)
  - `app/auth/middleware.py:34` (`UserContext`)
  - `app/services/llm_client.py:55` (`RootCauseResult`), `:64` (`LLMResponse`)
  - `app/pipeline/chat_session.py:62` (`ChatTurn`)
- **Pydantic** for runtime-validated config and any external input (HTTP body, JSON file, env-driven settings).
  - `app/strict_config.py:14` is the project base — `ConfigDict(extra="forbid")`.
  - Subclass it for new strict models (`app/types/retrieval.py`).
  - `LLMSettings` in `app/config.py` is the env-driven config example.
- Default to **frozen** for dataclasses (`@dataclass(frozen=True)`) unless you genuinely need mutation. Frozen prevents the "I'll just stick another field on later" tax debt.

## 5. Errors

- Build a small custom hierarchy when callers need to distinguish failure modes. The canonical example: `app/auth/jwt_auth.py:56` defines `JWTVerificationError` then `JWTExpiredError`/`JWTInvalidIssuerError`/`JWTMissingClaimError` subclasses; `app/auth/middleware.py` catches each branch with a different HTTP response.
- Wrap-and-reraise uses `from`. Always: `raise NewError("msg") from exc`. Examples: `app/auth/middleware.py:96`, `app/services/llm_client.py:131`.
- Bare `except Exception` is allowed only at top-level fallback paths and must carry `# noqa: BLE001`. Examples: `app/pipeline/driver.py:89`, `app/pipeline/runners.py` (the contextlib.suppress pattern around the runner task).
- Never `except:` (bare). Never swallow exceptions silently — at minimum `logger.exception(...)`.

## 6. Logging

- One module-level logger per file: `logger = logging.getLogger(__name__)`. Examples: `app/services/llm_client.py:33`, `app/services/datadog/client.py:20`, `app/pipeline/driver.py:45`.
- Inside an `except` block: `logger.exception("what failed")` — it auto-attaches the traceback. Outside: `logger.warning(...)` / `logger.debug(...)` / `logger.info(...)`.
- Do NOT `print()` from production code. CLI surfaces use `click.echo` (`app/cli/commands/deploy.py`) or `rich.console.Console` for styled output.
- The project also exposes `app.output.debug_print` for spinner-friendly node debug output — use that inside nodes when log lines would clutter the console.

## 7. Async

- Make a function `async def` when it does I/O (HTTP, DB, file, subprocess) or awaits one. Don't add `async` for symmetry alone.
- To call sync code from an async path without blocking the event loop: `await asyncio.to_thread(fn, *args)`. Example: `app/webapp.py:138, 200, 216`.
- For real parallelism (multiple sync calls at once), use the existing `concurrent.futures.ThreadPoolExecutor` pattern in `app/nodes/investigate/execution/execute_actions.py:162`. Claude Agent SDK does **not** parallelise tool calls; that's why this executor exists.
- Async tests need `@pytest.mark.asyncio` (the project uses `pytest-asyncio`). Example: `tests/app/auth/test_jwt_auth.py:10`.
- HTTP from async paths uses `httpx.AsyncClient` (`app/auth/jwt_auth.py:125`).

## 8. Imports

- Three groups, alphabetical within each: stdlib → third-party → first-party. `ruff` rule `I` enforces this; let `ruff format` reorder them.
- Lazy imports inside a function get `# noqa: PLC0415` and exist for one of three reasons: breaking a circular import, deferring a heavy import (LLM client), or avoiding an optional dependency at module load. Always include a one-line comment if the reason isn't immediately obvious. Examples: `app/webapp.py:129, 158, 174, 191` ("noqa: PLC0415" — lazy to avoid pulling heavy modules at import time).
- Never use `from x import *`.
- E402 (module-level import not at top) is acceptable when `load_env(...)` must run before imports that read env at module load — example: `app/main.py`. The ruff per-file ignore for tests already covers test-side cases.

## 9. Naming

- `snake_case` for functions, methods, variables, modules.
- `PascalCase` for classes. No `Type` suffix.
- `SCREAMING_SNAKE_CASE` for module-level constants (`MAX_INVESTIGATION_LOOPS` in `app/investigation_constants.py`).
- Single underscore (`_helper`) for module-private. Double underscore reserved for dunder methods.
- Tests: prefer `tests/<area>/test_<feature>.py`. Co-located `*_test.py` modules exist in legacy folders (`app/nodes/**/node_test.py`, `app/dockerfile_test.py`); don't add new ones — put new tests under `tests/`.

## 10. Pydantic

- Use `Model.model_validate(payload)` and `model.model_dump(mode="python")`. Never `parse_obj` / `dict()` (Pydantic v1 names).
- `Field(default_factory=list)` (or `dict`, `set`) for mutable defaults.
- Validators: `@field_validator("name", mode="before")` for normalisation, `@model_validator(mode="after")` for cross-field invariants. Examples: `app/strict_config.py:16, 23`.
- Strict project base: `from app.strict_config import StrictConfigModel`. Subclass it to inherit `extra="forbid"`. New ad-hoc models without `StrictConfigModel` should still set `model_config = ConfigDict(extra="forbid")` unless they have a specific reason to accept unknown fields.
- The `AgentState` TypedDict and `AgentStateModel` Pydantic model in `app/state/agent_state.py` must declare the same set of keys. `tests/app/test_agent_state_sync.py` enforces this — if you change one, change the other.

## 11. Tests

- Pytest functions are the default. `Test*` classes only when grouping clearly related cases (`tests/test_bug_fixes_e2e.py`).
- Prefer `monkeypatch` over `unittest.mock.patch` for environment, attribute, and import patches. Example: `tests/app/test_driver.py:88`. Use `unittest.mock.patch` only when you specifically need its `with patch(...)` ergonomics for nested context (see `tests/e2e/test_webapp_auth.py`).
- Async tests: `@pytest.mark.asyncio async def test_...`.
- Fixtures live in the nearest `conftest.py` (root: `tests/conftest.py`; per-area conftests are also fine).
- Live-LLM and live-infra tests live under `tests/synthetic/` and `tests/e2e/` and are excluded from `make test-cov`. New fast unit tests go under `tests/app/`, `tests/nodes/`, `tests/integrations/`, `tests/tools/`, etc., matching the source layout.
- Stub the network. If your test reaches the open internet, you've done something wrong.

## 12. Docstrings

- One-line summary for functions/methods. Multi-line only when documenting non-obvious side effects, return shape, or migration context.
- Module docstrings welcome on entry-points and key surfaces (`app/webapp.py:1`, `app/pipeline/driver.py:1`). Skip them on small leaf modules.
- No Sphinx `:param:` / `:returns:` / `:raises:` blocks. Types belong in annotations; behavior belongs in prose.
- Reading `app/pipeline/chat_session.py:1-22` is a good template for a "design notes" module docstring.

## 13. Comments

- Default to none. Add a comment when removing it would hide a non-obvious constraint, invariant, or workaround.
- Examples that earn their keep: `app/pipeline/runners.py:_DEFAULT_RUNNER` ("Phase 6 default flip" — explains why the value is what it is); `app/auth/middleware.py:tenant_filter` ("mirrors the @auth.on.*.search filter").
- Never write `# Update the foo` above `foo = ...`. The code already says that.
- Section dividers (`# ───── …`) are accepted in long modules but not required (`app/services/llm_client.py:35`).

## 14. `__all__`

- Pin the public surface in package roots and small public-API modules.
- Examples: `app/__init__.py:5`, `app/services/structured_llm.py:60`, `app/pipeline/__init__.py:7`, `app/pipeline/chat_session.py:199`.
- Skip `__all__` on internal modules — it's noise there.

## 15. CLI

- Click only. `@click.group(invoke_without_command=True)` for top-level groups (`app/cli/commands/deploy.py:214`), `@click.command(name="...")` for leaf commands.
- User-facing failures raise `OpenSREError` (`app/cli/errors.py:16`), which renders cleanly. Don't `sys.exit(1)` from a command — let Click handle the exit code.
- For interactive prompts use `questionary`; for output use `click.echo` or `rich.console.Console`.
- Never parse `sys.argv` by hand.

## 16. Tool registry

- To add a tool: write a function or `BaseTool` subclass and decorate with `@tool(...)` from `app.tools.tool_decorator`, OR build a `RegisteredTool.from_function(...)` / `.from_base_tool(...)` directly. The auto-discovery loop in `app/tools/registry.py` picks it up at import time.
- The full checklist (metadata, surfaces, schema, tests, docs) is in `AGENTS.md §2 "Adding a Tool"` — follow it rather than this rules file for the procedure.
- For a class-based tool example see `app/tools/GrafanaAlertRulesTool/__init__.py`.

## 17. State mutation

- The procedural pipeline mutates `AgentState` dicts in place via `_merge_state` (`app/pipeline/driver.py:50`, `app/pipeline/runners.py:27`). That is the one place this pattern is endorsed — concentrated, named, and tested.
- Elsewhere prefer immutability: return a new dict or build a frozen dataclass. Don't reach into someone else's object and mutate a field; pass values explicitly.
- Don't add reducer infrastructure (annotated reducers, Add/Remove markers) — the LangGraph reducer system was removed in Phase 7 and we don't want it back.

## 18. Pre-push checklist

Mirror `AGENTS.md` "Before Push":

1. clean working tree
2. `make test-cov`
3. `make lint`
4. `make typecheck`

If any of these fail, fix the cause — don't `--no-verify` your way past them. If lint/typecheck flags a legitimate exception, add a per-file or per-line ignore with a one-line comment explaining why.
