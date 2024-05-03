from typing import Any

import collections.abc
import itertools


__all__ = ["make_n_tuple"]


# Functional (sorted in alphabetical order)

def make_n_tuple(x: Any, n: int) -> tuple[Any, ...]:
    if isinstance(x, collections.abc.Iterable):
        return tuple(x)
    return tuple(itertools.repeat(x, n))
