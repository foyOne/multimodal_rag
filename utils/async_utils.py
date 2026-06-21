import asyncio

from collections.abc import Callable
from contextvars import copy_context
from functools import partial
from typing import (
    ParamSpec,
    TypeVar,
    cast,
)

P = ParamSpec("P")
T = TypeVar("T")


async def run_as_async_function(
    fn: Callable[P, T],
    *args: P.args,
    **kwargs: P.kwargs,
) -> T:
    """Run a function in an executor.

    Args:
        fn: The function.
        *args: The positional arguments to the function.
        **kwargs: The keyword arguments to the function.

    Returns:
        The output of the function.
    """

    def wrapper() -> T:
        try:
            return fn(*args, **kwargs)
        except StopIteration as exc:
            # StopIteration can't be set on an asyncio.Future
            # it raises a TypeError and leaves the Future pending forever
            # so we need to convert it to a RuntimeError
            raise RuntimeError from exc

    # Use default executor with context copied from current context
    return await asyncio.get_running_loop().run_in_executor(
        None,
        cast("Callable[..., T]", partial(copy_context().run, wrapper)),
    )
