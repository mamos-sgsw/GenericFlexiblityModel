"""
Global simulation settings for the Generic Flexibility Model.

These settings define parameters that should be consistent across all flexibility
assets and optimization algorithms.
"""

# Time step duration [hours]
# All FlexUnit implementations assume this time step for:
#   - Power limit calculations (P = E / dt)
#   - Energy conversions (E = P * dt)
#   - State updates
DT_HOURS = 0.25  # 15 minutes (0.25 hours)