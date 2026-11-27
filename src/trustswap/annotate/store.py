"""Private SQLite state for document custody and human annotation."""
from contextlib import closing, contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3

import bcrypt

ROLES = ("A1", "A2", "ADJ")
USER_ID = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,31}")
DOC_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
DEFAULT_SEED = 20261026
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY, role TEXT NOT NULL UNIQUE CHECK(role IN ('A1','A2','ADJ')),
    password BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, raw_hash TEXT NOT NULL, text_hash TEXT NOT NULL, url TEXT,
    text TEXT NOT NULL, offsets TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS screening (
    doc TEXT PRIMARY KEY REFERENCES documents(id), user TEXT NOT NULL REFERENCES users(id),
    record TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS cohort (
    doc TEXT PRIMARY KEY REFERENCES documents(id), overlap INTEGER NOT NULL, control INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS work (
    doc TEXT NOT NULL REFERENCES documents(id), user TEXT NOT NULL REFERENCES users(id),
    stage TEXT NOT NULL CHECK(stage IN ('draft','review','final')),
    revision INTEGER NOT NULL DEFAULT 0, data TEXT NOT NULL,
    PRIMARY KEY(doc,user)
);
CREATE TABLE IF NOT EXISTS adjudication (
    doc TEXT NOT NULL REFERENCES cohort(doc), field TEXT NOT NULL, user TEXT NOT NULL REFERENCES users(id),
    answer TEXT NOT NULL, reason TEXT NOT NULL, PRIMARY KEY(doc,field)
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY, doc TEXT NOT NULL REFERENCES documents(id), field TEXT,
    user TEXT NOT NULL REFERENCES users(id), action TEXT NOT NULL, before TEXT, after TEXT,
    reason TEXT, at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
BEGIN SELECT RAISE(ABORT,'annotation audit is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
BEGIN SELECT RAISE(ABORT,'annotation audit is append-only'); END;
"""


def dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def connect(path):
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA recursive_triggers=ON")
    return db


def initialize(path):
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    if not file.exists():
        file.touch(mode=0o600, exist_ok=True)
    with closing(connect(file)) as db:
        db.executescript(SCHEMA)


@contextmanager
def transaction(db):
    with db:
        db.execute("BEGIN IMMEDIATE")
        yield


def user_for(db, role):
    row = db.execute("SELECT id FROM users WHERE role=?", (role,)).fetchone()
    return row["id"] if row else None


def create_user(db, name, role, password):
    secret = password.encode()
    if not USER_ID.fullmatch(name) or role not in ROLES or not 8 <= len(secret) <= 72:
        raise ValueError("Use a valid user ID, role, and an 8–72 byte password")
    with transaction(db):
        db.execute("INSERT INTO users VALUES (?,?,?)", (name, role, bcrypt.hashpw(secret, bcrypt.gensalt())))


def audit(db, doc, user, action, *, field=None, before=None, after=None, reason=None, at=None):
    db.execute("""INSERT INTO audit(doc,field,user,action,before,after,reason,at)
                  VALUES (?,?,?,?,?,?,?,COALESCE(?,strftime('%Y-%m-%dT%H:%M:%fZ','now')))""",
               (doc, field, user, action, dump(before), dump(after), reason, at))


def sealed(db):
    return db.execute("SELECT 1 FROM metadata WHERE key='sealed'").fetchone() is not None


def seed(db, value=None):
    row = db.execute("SELECT value FROM metadata WHERE key='seed'").fetchone()
    if value is not None and row is not None and int(row[0]) != value:
        raise ValueError("Use the seed already recorded in this database")
    selected = int(row[0]) if row else DEFAULT_SEED if value is None else value
    if row is None and value is not None:
        db.execute("INSERT INTO metadata VALUES ('seed',?)", (str(selected),))
    return selected


def assign(included_ids, random_seed):
    ids = sorted(included_ids)
    def sample(purpose, count):
        def rank(doc):
            return sha256(f"{random_seed}:{purpose}:{doc}".encode()).digest()
        return set(sorted(ids, key=rank)[:count])

    overlap = sample("overlap", (35 * len(ids) + 50) // 100)
    controls = sample("control", 10)
    return [(doc, int(doc in overlap), int(doc in controls)) for doc in ids]


def lock_cohort(db):
    if sealed(db) or db.execute("SELECT count(*) FROM documents").fetchone()[0] == 0:
        return
    if db.execute("SELECT 1 FROM documents WHERE id NOT IN (SELECT doc FROM screening)").fetchone():
        return
    records = db.execute("SELECT doc, record FROM screening")
    included = [row["doc"] for row in records if json.loads(row["record"])["include"]]
    random_seed = seed(db, seed(db))
    db.executemany("INSERT INTO cohort VALUES (?,?,?)", assign(included, random_seed))
    db.execute("INSERT INTO metadata VALUES ('sealed','1')")


def get_work(db, doc, user):
    row = db.execute("SELECT * FROM work WHERE doc=? AND user=?", (doc, user)).fetchone()
    return None if row is None else {**dict(row), "data": json.loads(row["data"])}
