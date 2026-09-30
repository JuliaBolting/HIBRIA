"""Cancelamento cooperativo isolado por execução, inclusive entre consultas."""
from contextvars import ContextVar
from functools import wraps

_event = ContextVar("analysis_cancel_event", default=None)


class AnalysisCancelled(BaseException):
    # Como asyncio.CancelledError, não é uma falha tolerável de provedor.
    # A API captura este sinal; except Exception não pode esconder o cancelamento.
    pass


def checkpoint():
    event = _event.get()
    if event is not None and event.is_set():
        raise AnalysisCancelled("Análise cancelada pelo usuário.")


def cancellable(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        token = _event.set(kwargs.get("cancel_event"))
        try:
            checkpoint()
            return function(*args, **kwargs)
        finally:
            _event.reset(token)
    return wrapped
