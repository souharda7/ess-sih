from __future__ import annotations

import pytest

from ess_module_a.models import DangerDirection
from ess_module_a.statistics import (
    direction_risk,
    distribution_stats,
    empirical_percentile,
    iqr_distance,
    robust_z,
)
from ess_module_a.units import convert_value


def test_robust_z_matches_problem_example():
    stats = distribution_stats([8, 8, 10, 10, 10, 10, 12, 12, 45])
    custom = type(stats)(count=9, median=10.0, mad=2.0, q1=8.0, q3=12.0, values=stats.values)
    assert robust_z(45.0, custom) == pytest.approx(11.8036, rel=1e-3)


def test_mad_zero_uses_epsilon_without_crashing():
    stats = distribution_stats([10.0] * 20)
    assert robust_z(10.0, stats) == 0.0
    assert robust_z(11.0, stats) > 1_000_000


def test_percentile_iqr_and_direction():
    stats = distribution_stats(range(1, 11))
    assert empirical_percentile(10, stats) == 100.0
    assert iqr_distance(5, stats) == 0.0
    assert direction_risk(-4, DangerDirection.LOWER) == 4
    assert direction_risk(-4, DangerDirection.TWO_SIDED) == 4


def test_unit_conversion():
    assert convert_value(0.045, "mA", "uA") == pytest.approx(45.0)
    assert convert_value(2500, "mV", "V") == pytest.approx(2.5)
    with pytest.raises(ValueError, match="Cannot convert"):
        convert_value(1, "V", "uA")
