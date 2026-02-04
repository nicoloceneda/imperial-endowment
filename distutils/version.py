"""Minimal LooseVersion implementation for pandas_datareader compatibility."""

from packaging.version import Version


class LooseVersion:
    def __init__(self, vstring):
        self._version = Version(str(vstring))

    def __repr__(self) -> str:
        return f"LooseVersion('{self._version}')"

    def _compare(self, other, op):
        other_version = other._version if isinstance(other, LooseVersion) else Version(str(other))
        return op(self._version, other_version)

    def __lt__(self, other):
        return self._compare(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._compare(other, lambda a, b: a <= b)

    def __eq__(self, other):
        return self._compare(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._compare(other, lambda a, b: a != b)

    def __gt__(self, other):
        return self._compare(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._compare(other, lambda a, b: a >= b)
