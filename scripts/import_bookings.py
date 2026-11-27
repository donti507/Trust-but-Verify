"""Import fictional booking CSV rows into the local desk and report rejected lines."""
import argparse
import os
from pathlib import Path

from trustswap.desk import imports, store


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--db", default=os.environ.get("DESK_DB", "desk.db"))
    args = parser.parse_args(argv)
    store.initialize(args.db)
    db = store.connect(args.db)
    try:
        with args.csv.open(encoding="utf-8-sig", newline="") as source:
            report = imports.import_bookings(db, source)
    except (ValueError, OSError, UnicodeError) as error:
        print(f"Import rejected: {error}")
        return 1
    finally:
        db.close()
    print(f"Matched {report.matched}. Updated {report.updated}. Rejected {len(report.rejected)}.")
    for row in report.rejected:
        print(f"Line {row.line}: {row.reason}")
    return int(bool(report.rejected))


if __name__ == "__main__":
    raise SystemExit(main())
