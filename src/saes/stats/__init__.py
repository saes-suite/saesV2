"""Statistical engine package containing tests, assumption checks, and decision trees."""

from saes.stats.power import (
    analyze_statistical_power,
    calculate_statistical_power,
    compute_post_hoc_power_audit,
    estimate_required_sample_size,
    generate_power_curves,
)
from saes.stats.profiles import compute_performance_profiles

__all__ = [
    "analyze_statistical_power",
    "calculate_statistical_power",
    "compute_post_hoc_power_audit",
    "estimate_required_sample_size",
    "generate_power_curves",
    "compute_performance_profiles",
]

