import logging
from typing import Callable

logger = logging.getLogger(__name__)


class ToolFactory:
    registry = {}

    @classmethod
    def register(cls, name: str) -> Callable:
        def inner_wrapper(wrapped_class: Callable) -> Callable:
            if name in cls.registry:
                logger.warning("Tool %s already exists. Will replace it", name)  # noqa
            cls.registry[name] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def get(cls, name: str) -> Callable:
        assert name in cls.registry, f"Tool {name} does not exist in."  # noqa
        return cls.registry[name]
