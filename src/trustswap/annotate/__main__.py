"""Start the local annotation tool or manage its private database."""
import argparse
from contextlib import closing
import getpass
import os
import sqlite3

from trustswap.annotate import ingest, store
from trustswap.annotate.app import create_app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=os.environ.get("ANNOTATION_DB", "annotation.db"))
    subcommands = parser.add_subparsers(dest="command")
    account = subcommands.add_parser("create-user")
    account.add_argument("user")
    account.add_argument("role", choices=store.ROLES)
    documents = subcommands.add_parser("import")
    documents.add_argument("directory")
    documents.add_argument("--manifest", required=True)
    documents.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.command is None:
        create_app(args.db, secret=os.environ.get("ANNOTATION_SECRET")).run(host="127.0.0.1", port=8001)
        return 0

    store.initialize(args.db)
    try:
        with closing(store.connect(args.db)) as db:
            if args.command == "create-user":
                password = getpass.getpass("Password: ")
                store.create_user(db, args.user, args.role, password)
                print(f"Created {args.role} account.")
            else:
                count = ingest.import_documents(db, args.directory, args.manifest, random_seed=args.seed)
                print(f"Imported {count} documents.")
    except sqlite3.IntegrityError:
        parser.error("That user ID or role already exists")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
