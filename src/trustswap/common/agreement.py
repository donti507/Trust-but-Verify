"""Statistics for paired human labels."""
from collections import Counter


def cohen_kappa(left, right):
    if len(left) != len(right):
        raise ValueError("status pairs must have equal lengths")
    count = len(left)
    if not count:
        return None
    a, b = Counter(left), Counter(right)
    observed = sum(x == y for x, y in zip(left, right)) / count
    expected = sum(a[status] * b[status] for status in a.keys() | b.keys()) / count ** 2
    return None if expected == 1 else (observed - expected) / (1 - expected)
