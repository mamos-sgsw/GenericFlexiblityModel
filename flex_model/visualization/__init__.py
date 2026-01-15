"""
Visualization framework for FlexAsset optimization results.

This module provides tools for analyzing and visualizing optimization results,
targeting business stakeholders with interactive plots and economic decision metrics.

Key Components:
    - OptimizationResult: Abstract base class for all optimizer results
    - LPOptimizationResult: Linear programming result implementation
    - EconomicMetrics: Calculator for ROI, payback period, NPV, etc.
    - Executive plots: High-level KPI dashboards
    - Operational plots: Time-series analysis (power dispatch, SOC, prices)
    - Economic plots: Cost breakdown, waterfall charts
    - Utilization plots: Heatmaps, capacity factors, cycle analysis

Usage:
    from flex_model.visualization import LPOptimizationResult
    from flex_model.visualization.plots import ExecutivePlots, OperationalPlots

    # Wrap LP optimization result
    result = LPOptimizationResult(
        lp_result=optimizer.solve(),
        assets={'battery': battery, 'market': market},
        imbalance=imbalance_profile
    )

    # Generate visualizations
    fig = OperationalPlots.create_dispatch_profile(result)
    fig.show()
"""

from flex_model.visualization.core.result_processor import OptimizationResult
from flex_model.visualization.core.lp_result import LPOptimizationResult
from flex_model.visualization.core.metrics_calculator import EconomicMetrics

__all__ = [
    'OptimizationResult',
    'LPOptimizationResult',
    'EconomicMetrics',
]
