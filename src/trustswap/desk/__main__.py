"""Run the local Synthetic demo desk or create a role user."""
import argparse
import getpass
import os

from trustswap.desk import store
from trustswap.desk.app import create_app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.environ.get("DESK_DB", "desk.db"))
    commands = parser.add_subparsers(dest="command")
    user = commands.add_parser("create-user")
    user.add_argument("username")
    user.add_argument("role", choices=store.ROLES)
    args = parser.parse_args(argv)
    if args.command == "create-user":
        password = getpass.getpass("Password: ")
        store.initialize(args.db)
        db = store.connect(args.db)
        try:
            store.create_user(db, args.username, args.role, password)
        finally:
            db.close()
    else:
        create_app(args.db).run(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
