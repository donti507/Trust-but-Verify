"""SQLite desk state and an append-only audit trail."""
from contextlib import contextmanager
import json
import sqlite3

import bcrypt

ROLES = ("analyst", "supervisor", "auditor")
SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, password BLOB NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('analyst','supervisor','auditor'))
);
CREATE TABLE IF NOT EXISTS trades (
    id TEXT PRIMARY KEY, collection TEXT NOT NULL CHECK (collection='desk'),
    source_collection TEXT NOT NULL CHECK (source_collection IN ('synthetic','adversarial','live_feed')),
    source TEXT NOT NULL DEFAULT '', assigned_to INTEGER REFERENCES users(id),
    state TEXT NOT NULL DEFAULT 'Open' CHECK (state IN ('Open','Affirmed'))
);
CREATE TABLE IF NOT EXISTS fields (
    trade_id TEXT NOT NULL REFERENCES trades(id), path TEXT NOT NULL,
    confirmation TEXT NOT NULL, booking TEXT NOT NULL,
    p REAL NOT NULL, c REAL NOT NULL, r REAL NOT NULL,
    state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open','accepted','pending','approved','rejected')),
    proposed TEXT, note TEXT, proposer INTEGER REFERENCES users(id),
    reason TEXT, PRIMARY KEY (trade_id,path)
);
CREATE TABLE IF NOT EXISTS settings (name TEXT PRIMARY KEY, value REAL NOT NULL);
INSERT OR IGNORE INTO settings VALUES ('lambda',1.0);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY, at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    actor INTEGER REFERENCES users(id), action TEXT NOT NULL, trade_id TEXT, field TEXT,
    detail TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT,'audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT,'audit is append-only'); END;
"""


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute("PRAGMA recursive_triggers = ON")
    return db


def initialize(path):
    db = connect(path)
    try:
        db.executescript(SCHEMA)
    finally:
        db.close()


@contextmanager
def transaction(db):
    with db:
        db.execute("BEGIN IMMEDIATE")
        yield


def event(db, actor, action, trade=None, field=None, **detail):
    db.execute("INSERT INTO audit (actor, action, trade_id, field, detail) VALUES (?,?,?,?,?)",
               (actor, action, trade, field, json.dumps(detail, ensure_ascii=False, allow_nan=False)))


def create_user(db, name, role, password):
    encoded = password.encode()
    if role not in ROLES or not name.strip() or not 8 <= len(encoded) <= 72:
        raise ValueError("Use a valid role, a username and a password of 8 to 72 UTF-8 bytes")
    with transaction(db):
        user = db.execute("INSERT INTO users (name,role,password) VALUES (?,?,?)",
                          (name.strip(), role, bcrypt.hashpw(encoded, bcrypt.gensalt())))
        event(db, user.lastrowid, "user_created", role=role)


def new_trade(db, trade_id, source_collection, text, fields):
    """Insert a trade and its field evidence in a surrounding transaction."""
    if not fields:
        raise ValueError("A trade needs at least one field")
    db.execute("INSERT INTO trades (id,collection,source_collection,source) VALUES (?,'desk',?,?)",
               (trade_id, source_collection, text))
    db.executemany(
        "INSERT INTO fields (trade_id,path,confirmation,booking,p,c,r) VALUES (?,?,?,?,?,?,?)",
        [(trade_id, path, json.dumps(details["confirmation"], allow_nan=False),
          json.dumps(details["booking"], allow_nan=False), details["p"], details["c"], details["r"])
         for path, details in fields.items()],
    )
    event(db, None, "trade_added", trade_id, source_collection=source_collection)


def affirm_if_complete(db, trade_id, actor):
    if not db.execute("SELECT 1 FROM fields WHERE trade_id=? AND state NOT IN ('accepted','approved')",
                      (trade_id,)).fetchone():
        changed = db.execute("UPDATE trades SET state='Affirmed' WHERE id=? AND state='Open'", (trade_id,)).rowcount
        if changed:
            event(db, actor, "trade_affirmed", trade_id)
