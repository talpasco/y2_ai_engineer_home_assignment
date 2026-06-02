from __future__ import annotations

import copy
import time
from collections import OrderedDict
from dataclasses import replace
from typing import Generic, TypeVar

from app.models import ParseResult, TokenUsage

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, max_items: int = 10_000, ttl_seconds: int = 3600) -> None:
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, T]] = OrderedDict()

    def get(self, key: str) -> T | None:
        item = self._items.get(key)
        if item is None:
            return None
        created_at, value = item
        if time.time() - created_at > self.ttl_seconds:
            self._items.pop(key, None)
            return None
        self._items.move_to_end(key)
        return copy.deepcopy(value)

    def set(self, key: str, value: T) -> None:
        self._items[key] = (time.time(), copy.deepcopy(value))
        self._items.move_to_end(key)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)

    def clear(self) -> int:
        count = len(self._items)
        self._items.clear()
        return count

    def __len__(self) -> int:
        return len(self._items)


def cache_hit_result(result: ParseResult) -> ParseResult:
    clone = replace(result)
    clone.notes = list(result.notes)
    clone.params = copy.deepcopy(result.params)
    clone.source = "cache"
    clone.cache_hit = True
    clone.model_call_success = False
    clone.model_call_failure = False
    clone.usage = TokenUsage()
    return clone
