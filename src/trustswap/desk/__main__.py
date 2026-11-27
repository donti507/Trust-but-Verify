"""Run the local Synthetic demo desk or create a role user."""
import argparse
import getpass
import os

from trustswap.desk import imports, store
from trustswap.desk.app import create_app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.environ.get("DESK_DB", "desk.db"))
    commands = parser.add_subparsers(dest="command")
    user = commands.add_parser("create-user")
    user.add_argument("username")
    user.add_argument("role", choices=store.ROLES)
    seed = commands.add_parser("seed", help="Load fictional confirmations from replay and feed")
    seed.add_argument("--replay", default="results/demo/index.html")
    seed.add_argument("--feed-dir", default=os.environ.get("DESK_FEED_DIR", "feed"))
    bookings = commands.add_parser("import-bookings")
    bookings.add_argument("csv")
    args = parser.parse_args(argv)
    if args.command == "create-user":
        password = getpass.getpass("Password: ")
        store.initialize(args.db)
        db = store.connect(args.db)
        try:
            store.create_user(db, args.username, args.role, password)
        finally:
            db.close()
    elif args.command in {"seed", "import-bookings"}:
        store.initialize(args.db)
        db = store.connect(args.db)
        try:
            if args.command == "seed":
                count, planted = imports.seed(db, args.replay, args.feed_dir)
                print(f"Added {count} fictional trades with {planted} planted differences")
            else:
                with open(args.csv, encoding="utf-8-sig", newline="") as source:
                    report = imports.import_bookings(db, source)
                print(f"Updated {report.updated} bookings. Rejected {len(report.rejected)} rows.")
                for row in report.rejected:
                    print(f"Line {row.line}: {row.reason}")
        finally:
            db.close()
    else:
        create_app(args.db).run(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
