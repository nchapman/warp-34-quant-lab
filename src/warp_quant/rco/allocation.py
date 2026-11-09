"""Exact integer-byte multiple-choice knapsack allocation solver for RCO."""

from __future__ import annotations

import math
import random
from array import array
from dataclasses import dataclass
from fractions import Fraction

MAX_DP_CELLS = 10_000_000


@dataclass(frozen=True)
class Allocation:
    choices: tuple[int, ...]
    total_bytes: int
    score: float


@dataclass(frozen=True)
class Lattice:
    costs: tuple[tuple[int, ...], ...]
    increments: tuple[tuple[int, ...], ...]
    base_bytes: int
    unit_bytes: int
    capacity: int


def _lattice(costs, budget_bytes: int, max_cells: int = MAX_DP_CELLS) -> Lattice:
    costs = tuple(tuple(row) for row in costs)
    if not costs or any(not row for row in costs):
        raise ValueError("costs must contain nonempty option rows")
    num_opts = len(costs[0])
    if num_opts < 2 or any(len(row) != num_opts for row in costs):
        raise ValueError("every group must have the same number of options (>= 2)")
    if any(type(c) is not int or c < 0 for row in costs for c in row):
        raise ValueError("costs must be nonnegative integer bytes")
    if type(budget_bytes) is not int or budget_bytes < 0:
        raise ValueError("budget must be nonnegative integer bytes")
    if type(max_cells) is not int or max_cells <= 0:
        raise ValueError("max_cells must be a positive integer")
    minima = [min(row) for row in costs]
    base = sum(minima)
    if budget_bytes < base:
        raise ValueError(f"budget {budget_bytes} is below minimum {base}")
    deltas = [[c - low for c in row] for low, row in zip(minima, costs, strict=True)]
    unit = math.gcd(*(d for row in deltas for d in row)) or 1
    increments = tuple(tuple(d // unit for d in row) for row in deltas)
    capacity = min((budget_bytes - base) // unit, sum(max(row) for row in increments))
    if len(costs) * (capacity + 1) > max_cells:
        raise ValueError("exact cost lattice exceeds max_cells; refuse lossy budget rounding")
    return Lattice(costs, increments, base, unit, capacity)


def largest_reachable_budget(costs, ceiling_bytes: int, *, max_cells: int = MAX_DP_CELLS) -> int:
    """Largest attainable byte total not exceeding ceiling_bytes."""
    lattice = _lattice(costs, ceiling_bytes, max_cells)
    reachable = 1
    mask = (1 << (lattice.capacity + 1)) - 1
    for row in lattice.increments:
        next_reachable = 0
        for delta in set(row):
            if delta <= lattice.capacity:
                next_reachable |= reachable << delta
        reachable = next_reachable & mask
    return lattice.base_bytes + (reachable.bit_length() - 1) * lattice.unit_bytes


def upgrade_budget(costs, fraction: float | str) -> dict[str, int | str]:
    """Resolve a fraction in [0, 1] of the all-cheapest -> all-most-expensive byte gap."""
    frac = Fraction(str(fraction))
    if not 0 <= frac <= 1:
        raise ValueError("upgrade fraction must be within [0, 1]")
    costs = tuple(tuple(row) for row in costs)
    if not costs or any(not row for row in costs):
        raise ValueError("costs must contain nonempty option rows")
    if any(type(c) is not int or c < 0 for row in costs for c in row):
        raise ValueError("costs must be nonnegative integer bytes")
    low = sum(min(row) for row in costs)
    high = sum(max(row) for row in costs)
    ceiling = low + (high - low) * frac.numerator // frac.denominator
    realized = low if frac == 0 else (high if frac == 1 else largest_reachable_budget(costs, ceiling))
    return {
        "fraction": str(fraction),
        "all_low_bytes": low,
        "all_high_bytes": high,
        "requested_ceiling_bytes": ceiling,
        "realized_bytes": realized,
    }


def solve(
    costs,
    scores,
    budget_bytes: int,
    *,
    exact: bool = True,
    max_cells: int = MAX_DP_CELLS,
) -> Allocation:
    """Maximize sum of option scores at exactly budget_bytes (or <= budget_bytes if exact=False)."""
    if type(exact) is not bool:
        raise ValueError("exact must be boolean")
    lattice = _lattice(costs, budget_bytes, max_cells)
    scores = tuple(tuple(row) for row in scores)
    if len(scores) != len(lattice.costs) or any(
        len(score) != len(cost) for score, cost in zip(scores, lattice.costs, strict=True)
    ):
        raise ValueError("scores must have the same shape as costs")
    if any(type(s) not in (int, float) or math.isnan(s) or s == math.inf for row in scores for s in row):
        raise ValueError("scores must be finite numbers or -inf")
    if exact and lattice.base_bytes + lattice.capacity * lattice.unit_bytes != budget_bytes:
        raise ValueError("exact budget is unreachable on the byte-cost lattice")
    width = lattice.capacity + 1
    previous = array("d", [-math.inf]) * width
    previous[0] = 0.0
    trace = []
    for row, row_scores in zip(lattice.increments, scores, strict=True):
        current = array("d", [-math.inf]) * width
        choices = array("i", [-1]) * width
        for option, (delta, score) in enumerate(zip(row, row_scores, strict=True)):
            if score == -math.inf:
                continue
            for units in range(delta, width):
                prior = previous[units - delta]
                if prior == -math.inf:
                    continue
                candidate = prior + score
                if not math.isfinite(candidate):
                    raise ValueError("allocation score overflow")
                if candidate > current[units]:
                    current[units] = candidate
                    choices[units] = option
        trace.append(choices)
        previous = current
    end = lattice.capacity if exact else max(range(width), key=previous.__getitem__)
    if previous[end] == -math.inf:
        raise ValueError("no finite-score assignment satisfies the budget")
    optimum = previous[end]
    selected = []
    units = end
    for row, choices in zip(reversed(lattice.increments), reversed(trace), strict=True):
        choice = choices[units]
        if choice < 0:
            raise RuntimeError("invalid allocation traceback")
        selected.append(choice)
        units -= row[choice]
    selected.reverse()
    total = sum(row[i] for row, i in zip(lattice.costs, selected, strict=True))
    if units != 0 or total > budget_bytes or (exact and total != budget_bytes):
        raise RuntimeError("allocation failed independent byte accounting")
    return Allocation(tuple(selected), total, optimum)


def random_control(costs, budget_bytes: int, seed: int) -> Allocation:
    """Seeded random-score exact-budget control."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    costs = tuple(tuple(row) for row in costs)
    rng = random.Random(seed)
    scores = [[-math.log(-math.log(max(rng.random(), 2**-53))) for _ in row] for row in costs]
    return solve(costs, scores, budget_bytes)
