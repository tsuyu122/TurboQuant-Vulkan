"""End-to-end entry point: run the full bench, then render the report."""

from __future__ import annotations

import sys

from bench_suite import runner, report


def main() -> int:
    rc = runner.main()
    if rc != 0:
        return rc
    return report.main()


if __name__ == "__main__":
    sys.exit(main())
