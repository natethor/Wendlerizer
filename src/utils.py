"""
Utility functions for weight calculations and related operations.
"""

from typing import Optional


def round_weight(
    weight: float, barbell_weight: float = 45.0, precision: Optional[float] = None
) -> float:
    """
    Round a weight to the nearest loadable plate combination.

    Args:
        weight: The weight to round
        barbell_weight: Weight of the barbell (45lb/20kg standard)
        precision: Smallest plate increment available (defaults based on barbell)

    Returns:
        Rounded weight value that can be loaded with available plates
    """
    if weight < barbell_weight:
        return barbell_weight

    plate_weight = weight - barbell_weight

    # Set precision based on barbell type if not specified
    if precision is None:
        # Assume pounds if bar is 33/35/44/45, else kilos
        if float(barbell_weight) in (33.0, 35.0, 44.0, 45.0):
            precision = 5.0  # Standard plate increment in pounds
        else:
            precision = 1.0  # Standard plate increment in kilos

    base = int(plate_weight / precision) * precision
    rounded_up = base + precision
    delta_down = plate_weight - base
    delta_up = rounded_up - plate_weight

    return barbell_weight + (base if delta_down < delta_up else rounded_up)


def estimate_1rm(weight: float, reps: int) -> int:
    """
    Calculate estimated 1RM using Wendler's formula.

    Args:
        weight: Weight used for the set
        reps: Number of reps performed

    Returns:
        Estimated one rep max rounded to nearest 5
    """
    # Validate inputs
    if weight <= 0:
        raise ValueError("Weight must be greater than 0")
    if reps <= 0:
        raise ValueError("Reps must be greater than 0")

    # Wendler's formula: weight * reps * 0.0333 + weight
    est_1rm = weight * reps * 0.0333 + weight

    # Round to nearest 5
    return round(est_1rm / 5) * 5
