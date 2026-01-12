"""Core data processing components for visualization framework."""

from flex_model.visualization.core.result_processor import OptimizationResult
from flex_model.visualization.core.lp_result import LPOptimizationResult
from flex_model.visualization.core.metrics_calculator import EconomicMetrics
from flex_model.visualization.core.color_schemes import (
    get_color_scheme,
    get_rgba_with_alpha,
    ColorScheme,
    LIGHT_MODE,
    DARK_MODE,
)

__all__ = [
    'OptimizationResult',
    'LPOptimizationResult',
    'EconomicMetrics',
    'get_color_scheme',
    'get_rgba_with_alpha',
    'ColorScheme',
    'LIGHT_MODE',
    'DARK_MODE',
]
