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
CREATE TABLE IF NOT EXISTS field_evidence (
    trade_id TEXT NOT NULL, path TEXT NOT NULL, raw TEXT NOT NULL, span TEXT,
    PRIMARY KEY (trade_id,path), FOREIGN KEY (trade_id,path) REFERENCES fields(trade_id,path)
);
CREATE TABLE IF NOT EXISTS desk_uploads (
    trade_id TEXT PRIMARY KEY REFERENCES trades(id), sha256 TEXT NOT NULL,
    kind TEXT NOT NULL CHECK (kind IN ('pdf','png','jpeg','swift','fpml','text'))
);
CREATE TABLE IF NOT EXISTS desk_cases (
    trade_id TEXT PRIMARY KEY REFERENCES trades(id),
    status TEXT NOT NULL CHECK (status IN ('New','Assigned','Waiting for counterparty','Pending approval','Resolved')),
    owner INTEGER REFERENCES users(id), priority REAL NOT NULL,
    severity TEXT NOT NULL CHECK (severity IN ('critical','major','minor')),
    created_at TEXT NOT NULL, due_at TEXT NOT NULL, resolved_at TEXT
);
CREATE TABLE IF NOT EXISTS desk_trade_versions (
    trade_id TEXT NOT NULL REFERENCES trades(id), version INTEGER NOT NULL,
    event_type TEXT NOT NULL CHECK(event_type IN ('original','amendment','partial','full','novation')),
    effective TEXT NOT NULL, terms TEXT NOT NULL, actor INTEGER REFERENCES users(id),
    PRIMARY KEY(trade_id,version)
);
CREATE TRIGGER IF NOT EXISTS versions_no_update BEFORE UPDATE ON desk_trade_versions
BEGIN SELECT RAISE(ABORT,'trade versions are append-only'); END;
CREATE TRIGGER IF NOT EXISTS versions_no_delete BEFORE DELETE ON desk_trade_versions
BEGIN SELECT RAISE(ABORT,'trade versions are append-only'); END;
CREATE TABLE IF NOT EXISTS desk_lifecycle (
    id INTEGER PRIMARY KEY, trade_id TEXT NOT NULL REFERENCES trades(id), event_type TEXT NOT NULL,
    effective TEXT NOT NULL, base_version INTEGER NOT NULL, differences TEXT NOT NULL,
    risk_flags TEXT NOT NULL, upload_sha256 TEXT, proposer INTEGER NOT NULL REFERENCES users(id),
    note TEXT NOT NULL, reviewer INTEGER REFERENCES users(id), reason TEXT,
    status TEXT NOT NULL DEFAULT 'Pending approval' CHECK(status IN ('Pending approval','Approved','Rejected')),
    FOREIGN KEY(trade_id,base_version) REFERENCES desk_trade_versions(trade_id,version)
);
CREATE UNIQUE INDEX IF NOT EXISTS one_pending_lifecycle ON desk_lifecycle(trade_id) WHERE status='Pending approval';
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
    save_evidence(db, trade_id, fields)
    event(db, None, "trade_added", trade_id, source_collection=source_collection)


def save_evidence(db, trade_id, fields):
    rows = [(trade_id, path, json.dumps(info.get("raw"), ensure_ascii=False), json.dumps(info.get("span")))
            for path, info in fields.items() if info.get("raw") is not None or info.get("span") is not None]
    db.executemany("INSERT OR IGNORE INTO field_evidence (trade_id,path,raw,span) VALUES (?,?,?,?)", rows)


def affirm_if_complete(db, trade_id, actor):
    if not db.execute("SELECT 1 FROM fields WHERE trade_id=? AND state NOT IN ('accepted','approved')",
                      (trade_id,)).fetchone():
        changed = db.execute("UPDATE trades SET state='Affirmed' WHERE id=? AND state='Open'", (trade_id,)).rowcount
        if changed:
            event(db, actor, "trade_affirmed", trade_id)
