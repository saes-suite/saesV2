"""Statistical engine package containing tests, assumption checks, and decision trees."""

from saesV2.stats import power, profiles
from saesV2.stats.power import (
    analyze_statistical_power,
    calculate_statistical_power,
    compute_post_hoc_power_audit,
    estimate_required_sample_size,
    generate_power_curves,
)
from saesV2.stats.profiles import compute_performance_profiles

__all__ = [
    "power",
    "profiles",
    "analyze_statistical_power",
    "calculate_statistical_power",
    "compute_post_hoc_power_audit",
    "estimate_required_sample_size",
    "generate_power_curves",
    "compute_performance_profiles",
]

