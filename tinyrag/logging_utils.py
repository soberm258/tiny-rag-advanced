from __future__ import annotations

from typing import Any


def _format(msg: str, args: Any) -> str:
    try:
        return str(msg).format(*args)
    except Exception:
        # 兜底：避免格式化失败导致日志本身抛异常
        return " ".join([str(msg), *[str(a) for a in args]])


try:
    from loguru import logger as logger  # type: ignore
except Exception:  # pragma: no cover
    import logging

    _base = logging.getLogger("tinyrag")

    class _ShimLogger:
        def info(self, msg: str, *args: Any) -> None:
            _base.info(_format(msg, args))

        def warning(self, msg: str, *args: Any) -> None:
            _base.warning(_format(msg, args))

        def error(self, msg: str, *args: Any) -> None:
            _base.error(_format(msg, args))

    logger = _ShimLogger()

