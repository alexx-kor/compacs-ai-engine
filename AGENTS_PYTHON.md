# AGENTS_PYTHON.md — Python Coding Rules

Universal rules for Python code in this repository. Any coding agent MUST follow them. Keywords `MUST`, `MUST NOT`, `NEVER`, `ALWAYS`, `PREFER`, `AVOID` have RFC-2119 meaning.

## 1. Meta

- Reply to the user in Russian; keep code identifiers, docstrings, log messages, commits, and PR descriptions in English unless existing code clearly dictates otherwise.
- Read the target file and at least one neighbour in the same package/module before editing; match existing style even when it deviates slightly from rules below.
- Do only what is asked: no reformatting of untouched files, no unrelated renames, no new dependencies without explicit user approval.
- On ambiguous requirements or multiple valid approaches with different trade-offs, ask one concise question instead of guessing.
- Cite code to the user as `path/to/file.py:lineNumber`.

## 2. Project conventions discovery

- Read `pyproject.toml` / `setup.cfg` / `requirements.txt` for Python version, existing libraries, and quality plugins (ruff, black, mypy, pylint) before writing code.
- Skim `__init__.py`, top-level `README.md` if present.
- Pre-existing formatter config (`.editorconfig`, `pyproject.toml [tool.black]`, `ruff.toml`) overrides defaults in this file.

## 3. Python version & language features

- Target the Python version from the project config. On 3.10+ PREFER:
  - `dataclass` or `TypedDict` for data carriers; `@dataclass(frozen=True)` for immutable ones.
  - Structural pattern matching (`match`/`case`) for closed hierarchies.
  - `X | Y` union types instead of `Optional[X]` or `Union[X, Y]`.
  - f-strings for string formatting; triple-quoted strings for multi-line SQL, JSON, prompts.
  - `list` / `dict` / `set` / `tuple` built-in generics instead of `List` / `Dict` from `typing`.
- MUST NOT introduce features newer than the configured Python version.
- PREFER type annotations on all public functions and class fields.

## 4. Naming

Naming is the primary vehicle of intent; bad names cannot be fixed by comments.

**Case conventions**

| Kind                              | Convention        | Example                          |
| --------------------------------- | ----------------- | -------------------------------- |
| class / exception                 | `PascalCase`      | `OrderRepository`, `PaymentKind` |
| function / method / variable      | `snake_case`      | `find_by_email`, `order_id`      |
| constant / module-level final     | `UPPER_SNAKE_CASE`| `MAX_RETRY_COUNT`, `BASE_URL`    |
| package / module                  | `snake_case`      | `log_analyzer`, `click_house`    |
| type variable                     | single upper `T`  | `T`, `K`, `V`                    |

**Variables, functions, parameters**

- Meaningful full words; abbreviate only industry-standard terms (`url`, `id`, `http`, `sql`).
- Collections plural, scalars singular (`users`, `order_ids`, `user`).
- Boolean names start with `is_`/`has_`/`can_`/`should_`/`was_`/`needs_`.
- Functions start with a verb: `calculate`, `find`, `load`, `send`, `validate`, `apply`, `resolve`. Boolean functions use predicate prefixes. Converters use `to_xxx`/`as_xxx`; factories use `create`/`from_xxx`/`parse`.
- MUST NOT use `get_xxx` for heavy operations (I/O, remote calls) — use `fetch`/`load`/`resolve`.
- ≤ 4 positional parameters; beyond that, introduce a `dataclass` or `TypedDict`.
- AVOID boolean flags in signatures — split the function or accept an `Enum`.
- NEVER use machine names (`arg0`, `param`, `tmp`, `obj`) or numbered duplicates (`user1`, `user2`).
- PREFER abstract types in annotations (`Sequence[T]`, `Mapping[K, V]`, not `list`, `dict`).

```python
# Good
def find_active_users_by_tenant(tenant_id: TenantId) -> list[User]: ...
def has_permission(user: User, permission: Permission) -> bool: ...
def load_order(order_id: OrderId) -> Order: ...  # hits the database

# Bad
def get_data(t: str) -> list: ...               # vague verb, vague param, no types
def permission(u: User, p: Permission) -> bool: ...  # missing verb
def get_order(order_id: OrderId) -> Order: ...  # hides I/O behind a getter
```

**Classes, enums, constants**

- Classes are nouns with meaningful role suffixes: `Service`, `Repository`, `Controller`, `Factory`, `Builder`, `Validator`, `Mapper`, `Exception`, `Config`. AVOID `Helper`/`Util`/`Manager`/`Processor`/`Handler` when a more specific responsibility exists.
- Enums: name is a singular noun (`Status`, not `Statuses` / `StatusEnum`). Use `enum.Enum` or `enum.StrEnum`.
- Extract magic numbers/strings into module-level `UPPER_SNAKE_CASE` constants with explanatory names.

**Forbidden naming patterns**

- Hungarian notation: `str_name`, `i_count`, `b_enabled`.
- Meaningless nouns: `data`, `info`, `item`, `object`, `value`, `temp`, `foo`, `bar`.
- Numbered duplicates: `user1`, `list2`.
- Transliterated or native-language identifiers — identifiers MUST be English.
- Non-standard abbreviations: `cust_cnt`, `ord_ref`, `usr_svc`.

## 5. Best-practice checklist

- Single responsibility per class and function.
- Composition over inheritance; prefer `Protocol` for structural typing over ABC hierarchies.
- Fail fast: raise early on invalid input; use guard clauses and return early on invalid state.
- Pure core logic; push I/O, randomness, time, and mutation to the edges.
- DRY on the third occurrence, not the second.
- Make illegal states unrepresentable via `Enum` / `dataclass` / typed wrappers rather than runtime `if`-checks.
- Immutability by default; use `@dataclass(frozen=True)` or `NamedTuple` for value objects.
- `datetime` with timezone (`datetime.now(timezone.utc)`); NEVER naive datetimes for timestamps.
- Resources via `with` (context managers); NEVER rely on `__del__`.

```python
from dataclasses import dataclass
from decimal import Decimal

@dataclass(frozen=True)
class Money:
    amount: Decimal
    currency: str

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError(f"amount must be non-negative, got {self.amount}")

    def add(self, other: "Money") -> "Money":
        if self.currency != other.currency:
            raise ValueError(f"currency mismatch: {self.currency} vs {other.currency}")
        return Money(self.amount + other.amount, self.currency)
```

## 6. Layout & style

- One public class per file when the class is the primary export; file name matches the class in `snake_case`.
- Package names all-lowercase with underscores. PREFER feature-based packages (`orders/`, `payments/`) over layer-based (`controllers/`, `services/`).
- 4-space indent, no tabs; line width ≤ 120 (follow project's black/ruff config).
- Import order: stdlib → third-party → project. No wildcard imports (`from module import *`).
- One blank line between methods; two blank lines between top-level definitions.
- `__all__` on public modules to declare the API surface explicitly.

## 7. Immutability

- PREFER `@dataclass(frozen=True)` or `NamedTuple` for DTOs and value objects.
- Return `tuple` instead of `list` when the caller should not mutate the result.
- Defensive-copy mutable inputs (`list(items)`, `dict(mapping)`) when storing them.
- No public setters on domain objects; model changes as new instances.

## 8. Null-safety

- NEVER return `None` for collections or strings — return an empty instance (`[]`, `""`, `{}`).
- `Optional[X]` / `X | None` is a return type only when absence is meaningful. MUST NOT overuse it for parameters.
- Raise `ValueError` / `TypeError` at function entry for parameters that must not be `None`.
- PREFER `x if x is not None else default` over chained `or` for clarity.

```python
# Bad
def find_by_status(status: Status | None) -> list[Order] | None:
    if status is None:
        return None
    ...

# Good
def find_by_status(status: Status) -> list[Order]:
    if status is None:
        raise ValueError("status must not be None")
    return repository.find_by_status(status)  # never None; may be empty
```

## 9. Exceptions

- Raise the most specific exception type available; define one exception class per failure mode.
- NEVER swallow (`except Exception: pass`) — at minimum log with context and re-raise.
- ALWAYS preserve the cause when wrapping: `raise OrderLoadException(f"order={id}") from e`.
- NEVER use exceptions for control flow — return `None | X` or a sentinel for normal "not found" cases.
- Do not catch `BaseException` or `KeyboardInterrupt` outside top-level entry points.

## 10. Collections & iteration

- PREFER built-in comprehensions (`[x for x in ...]`) for simple transformations.
- Use `itertools` for chaining, grouping, and lazy pipelines.
- PREFER `dict.get(key, default)` / `collections.defaultdict` over check-then-insert idioms.
- NEVER mutate a collection while iterating over it.
- Pre-size lists (`[None] * n`) when size is known and performance matters.

## 11. Concurrency

- PREFER `asyncio` for I/O-bound work; `concurrent.futures.ThreadPoolExecutor` for blocking I/O in sync context; `ProcessPoolExecutor` for CPU-bound.
- NEVER `time.sleep` in business logic — use `asyncio.sleep` or scheduled executors.
- Shared mutable state MUST be protected with `asyncio.Lock`, `threading.Lock`, or queue-based design.
- Document thread-safety guarantees in class-level docstring.
- NEVER use deprecated `threading.Thread.stop` or similar unsafe primitives.

## 12. Logging

- Obtain loggers via stdlib: `log = logging.getLogger(__name__)`.
- ALWAYS use `%`-style or `logging.LogRecord` lazy formatting; NEVER f-strings directly in log calls (defeats lazy evaluation).
- NEVER log secrets, tokens, card numbers, or other PII; mask before logging.
- Include correlation IDs (request/trace) when available.
- Levels: `ERROR` actionable failure, `WARNING` recoverable anomaly/retry, `INFO` lifecycle and key business outcomes, `DEBUG` flow diagnostics.

```python
# Good
log.info("order processed order_id=%s amount=%s currency=%s", order_id, amount, currency)

# Bad
log.info(f"order processed {order_id} {amount}")   # f-string defeats lazy eval
log.error("failed")                                 # no context
log.info("charged card %s", full_card_number)       # PII leak
```

## 13. Testing

- pytest + pytest-mock (or the project's established stack).
- Structure Arrange — Act — Assert with blank-line separators; one logical assertion per test.
- Descriptive names: `test_should_<expected>_when_<condition>`.
- NEVER `time.sleep` — use `freezegun`, fake clocks, or `pytest-asyncio` fixtures.
- `tmp_path` fixture for filesystem I/O.
- Cover happy and failure paths for every public function.
- NEVER put logic (`if`/`for`/`while`) in test bodies — use helpers or `@pytest.mark.parametrize`.
- Tests MUST be deterministic and order-independent.

## 14. Docs & comments

- Docstring required on every public module, class, and function; optional on internal/private code when names are self-explanatory.
- Docstring describes **what** and **why**, not **how**. PREFER Google or NumPy style consistently across the project. Include `Args:`, `Returns:`, `Raises:` sections when non-obvious.
- Comments explain intent, trade-offs, constraints, or non-obvious context — never what the code already says.
- `TODO`/`FIXME` MUST include owner and reason: `# TODO(alice): remove after migration to v2 API (PROJ-1234)`.
- NEVER leave commented-out code in committed changes — version control remembers.
- NEVER add comments that narrate the change you just made (`# added null check`).

## 15. Security baseline

- NEVER hardcode secrets (API keys, passwords, tokens) — read from config/env/secret manager (`os.environ`, `pydantic-settings`).
- ALWAYS use parameterized queries; NEVER build SQL by concatenating user input.
- Validate and normalize input at trust boundaries (HTTP, queue, file).
- AVOID `subprocess` with user input; if unavoidable, pass arguments as a list, never a shell string.
- `secrets` module (not `random`) for tokens, IDs, nonces, salts.
- Hash passwords with adaptive algorithms (`bcrypt`, `argon2-cffi`); never plain SHA-*.

## 16. Performance

- Measure before optimizing; profile with `cProfile` or `py-spy`, don't guess.
- AVOID premature caching — caches are correctness hazards.
- Use generators / `itertools` for large pipelines to avoid materializing full lists.
- PREFER `str.join` over `+` concatenation in loops.
- Pre-size collections when size is known; use `array` / `numpy` for numeric-heavy work.

## 17. Build & dependencies

- NEVER bump versions in `pyproject.toml` / `requirements.txt` without explicit user approval.
- PREFER existing dependencies; propose new ones with justification before adding.
- Keep the code importable at every intermediate step.
- Before declaring a task done, run the project's verification command (typically `pytest -q` or `ruff check . && mypy .`); say so explicitly if unavailable locally.

## 18. Anti-patterns (DO NOT)

- Mutable module-level state or global singletons with mutable data.
- Returning `None` instead of an empty collection.
- Untyped `list` / `dict` in public APIs.
- `print()` in production code — log via `logging`.
- `==` comparison with `None` — use `is None` / `is not None`.
- String concatenation inside loops (use `"".join(...)`).
- `except Exception: pass` — always handle or re-raise with context.
- Wildcard imports `from module import *`.
- Mutable default arguments `def f(items=[])` — use `None` and set inside.
- Modifying a list/dict while iterating over it.
