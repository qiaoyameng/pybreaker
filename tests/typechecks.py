from pybreaker import STATE_CLOSED, CircuitBreaker, CircuitMemoryStorage, FailureWindowStats

# issue #90: incorrect typing for exclude argument
# this should not give errors in mypy
CircuitBreaker(
    fail_max=1,
    reset_timeout=1000,
    exclude=[lambda e: isinstance(e, RuntimeError)],
    state_storage=CircuitMemoryStorage(STATE_CLOSED),
)

window_breaker = CircuitBreaker(
    reset_timeout=30,
    failure_window=60,
    min_samples=20,
    failure_rate_threshold=0.5,
    clock=lambda: 1.0,
)
window_stats: FailureWindowStats | None = window_breaker.failure_window_stats
if window_stats is not None:
    _sample_count: int = window_stats.sample_count
    _failure_rate: float = window_stats.failure_rate
