"""Tool access control: allow/deny lists, rate limiting, human confirmation."""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable

from .tools import ToolError, ToolRegistry


class AccessPolicy:
    """Wraps a ToolRegistry with access control.

    - allow / deny: name-based filtering (deny wins over allow)
    - rate_limit_per_minute: per-tool sliding-window rate limit
    - confirm: callable(tool_name, arguments) -> bool; returning False blocks
      the call (e.g. hook to ask a human before dangerous tools)
    """

    def __init__(self, registry: ToolRegistry,
                 allow: set[str] | None = None,
                 deny: set[str] | None = None,
                 rate_limit_per_minute: int | None = None,
                 confirm: Callable[[str, dict], bool] | None = None):
        self.registry = registry
        self.allow = allow
        self.deny = deny or set()
        self.rate_limit = rate_limit_per_minute
        self.confirm = confirm
        self._calls: dict[str, deque[float]] = {}

    def check(self, name: str, arguments: dict) -> None:
        """Raise ToolError if the call is not permitted."""
        if name in self.deny:
            raise ToolError(f"Tool '{name}' is denied by policy")
        if self.allow is not None and name not in self.allow:
            raise ToolError(f"Tool '{name}' is not in the allow-list")
        if self.rate_limit is not None:
            now = time.time()
            window = self._calls.setdefault(name, deque())
            while window and now - window[0] > 60:
                window.popleft()
            if len(window) >= self.rate_limit:
                raise ToolError(
                    f"Tool '{name}' rate limit exceeded "
                    f"({self.rate_limit}/min)")
            window.append(now)
        if self.confirm is not None and not self.confirm(name, arguments):
            raise ToolError(f"Tool '{name}' call rejected by confirmation hook")


def guarded_registry(registry: ToolRegistry, policy: AccessPolicy
                     ) -> ToolRegistry:
    """Return a new registry whose tools enforce the policy before running."""
    guarded = ToolRegistry()
    for tool in registry.all():
        original = tool.func

        def make(fn, name):
            def wrapper(**kwargs):
                policy.check(name, kwargs)
                return fn(**kwargs)
            wrapper.__name__ = name
            return wrapper

        # preserve schema by re-registering with stored metadata
        from .tools import Tool
        t = Tool(make(original, tool.name),
                 name=tool.name, description=tool.description)
        t.schema = tool.schema  # keep original generated schema
        guarded._tools[t.name] = t
    return guarded
