"""Local desk web application and server-side authorization."""
from datetime import timedelta
from functools import wraps
import hmac
import json
import math
import os
import secrets

import bcrypt
from flask import Flask, abort, g, redirect, render_template, request, session, url_for

from trustswap.decide.policy import escalate
from trustswap.desk import matcher, store


def roles(*allowed):
    def decorate(view):
        @wraps(view)
        def authorized(*args, **kwargs):
            if g.user is None:
                return redirect(url_for("login"))
            if g.user["role"] not in allowed:
                abort(403)
            return view(*args, **kwargs)
        return authorized
    return decorate


def database():
    if "db" not in g:
        g.db = store.connect(g.app_db)
    return g.db


def csrf_token():
    if "csrf" not in session:
        session["csrf"] = secrets.token_urlsafe(32)
    return session["csrf"]


def create_app(path="desk.db", *, secret=None):
    key = secret or os.environ.get("DESK_SECRET_KEY", "")
    if len(key) < 32:
        raise ValueError("Set DESK_SECRET_KEY to at least 32 random characters")
    store.initialize(path)
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=key, SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict",
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=30), MAX_CONTENT_LENGTH=64_000,
        TRUSTED_HOSTS=["localhost", "127.0.0.1"],
    )
    app.jinja_env.globals["csrf_token"] = csrf_token
    dummy_hash = bcrypt.hashpw(b"no-account", bcrypt.gensalt())

    @app.before_request
    def authenticate():
        if request.method == "POST":
            expected = session.get("csrf", "")
            provided = request.form.get("csrf", "")
            if not expected or not hmac.compare_digest(expected.encode(), provided.encode()):
                abort(400, "Invalid CSRF token")
        g.app_db = path
        uid = session.get("uid")
        g.user = database().execute("SELECT id,name,role FROM users WHERE id=?", (uid,)).fetchone() if uid else None

    @app.teardown_appcontext
    def close_db(error):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.after_request
    def headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error = ""
        if request.method == "POST":
            db = database()
            user = db.execute("SELECT * FROM users WHERE name=?", (request.form.get("username", "").strip(),)).fetchone()
            password = request.form.get("password", "").encode()
            valid = len(password) <= 72 and bcrypt.checkpw(password, user["password"] if user else dummy_hash)
            if valid and user:
                session.clear()
                session["uid"] = user["id"]
                session.permanent = True
                with store.transaction(db):
                    store.event(db, user["id"], "login")
                return redirect(url_for("overview"))
            error = "Invalid username or password"
        return render_template("login.html", error=error), (401 if error else 200)

    @app.post("/logout")
    @roles(*store.ROLES)
    def logout():
        with store.transaction(database()):
            store.event(database(), g.user["id"], "logout")
        session.clear()
        return redirect(url_for("login"))

    @app.get("/")
    @roles(*store.ROLES)
    def overview():
        db = database()
        if g.user["role"] == "analyst":
            rows = db.execute("SELECT * FROM trades WHERE assigned_to IS NULL OR assigned_to=? ORDER BY id",
                              (g.user["id"],))
        else:
            rows = db.execute("SELECT * FROM trades ORDER BY id")
        threshold = db.execute("SELECT value FROM settings WHERE name='lambda'").fetchone()[0]
        return render_template("overview.html", trades=list(rows), threshold=threshold)

    def trade_for_user(trade_id):
        trade = database().execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        if trade is None:
            abort(404)
        if g.user["role"] == "analyst" and trade["assigned_to"] not in (None, g.user["id"]):
            abort(403)
        return trade

    @app.get("/trade/<trade_id>")
    @roles(*store.ROLES)
    def review(trade_id):
        trade = trade_for_user(trade_id)
        db = database()
        threshold = db.execute("SELECT value FROM settings WHERE name='lambda'").fetchone()[0]
        fields = []
        for row in db.execute("SELECT * FROM fields WHERE trade_id=? ORDER BY path", (trade_id,)):
            field = dict(row)
            field["confirmation"] = json.loads(row["confirmation"])
            field["booking"] = json.loads(row["booking"])
            field["match"] = matcher.match(row["path"], field["confirmation"], field["booking"])
            field["risk"] = bool(escalate(row["p"], row["c"], row["r"], threshold))
            fields.append(field)
        return render_template("review.html", trade=trade, fields=fields, threshold=threshold)

    @app.post("/trade/<trade_id>/field/<field>/review")
    @roles("analyst")
    def review_field(trade_id, field):
        db = database()
        action = request.form.get("action")
        note = request.form.get("note", "").strip()
        value = request.form.get("value", "").strip()
        if action not in {"accept", "propose"} or (action == "propose" and not (note and value)):
            abort(400, "Proposed corrections need a value and note")
        if action == "propose":
            try:
                matcher.normalize(field, value)
            except (ValueError, ArithmeticError):
                abort(400, "Invalid proposed value")
        with store.transaction(db):
            trade_for_user(trade_id)
            row = db.execute("SELECT * FROM fields WHERE trade_id=? AND path=?", (trade_id, field)).fetchone()
            if row is None:
                abort(404)
            if row["state"] not in {"open", "rejected"}:
                abort(409, "This field already has a review decision")
            db.execute("UPDATE trades SET assigned_to=? WHERE id=? AND assigned_to IS NULL", (g.user["id"], trade_id))
            if action == "accept":
                db.execute("UPDATE fields SET state='accepted' WHERE trade_id=? AND path=?", (trade_id, field))
                store.event(db, g.user["id"], "field_accepted", trade_id, field, note=note)
            else:
                db.execute("UPDATE fields SET state='pending',proposed=?,note=?,proposer=?,reason=NULL WHERE trade_id=? AND path=?",
                           (json.dumps(value), note, g.user["id"], trade_id, field))
                store.event(db, g.user["id"], "correction_proposed", trade_id, field, value=value, note=note)
            store.affirm_if_complete(db, trade_id, g.user["id"])
        return redirect(url_for("review", trade_id=trade_id))

    @app.get("/approvals")
    @roles("supervisor")
    def approvals():
        rows = database().execute(
            """SELECT fields.*, users.name AS analyst FROM fields JOIN users ON users.id=proposer
            WHERE state='pending' ORDER BY trade_id,path""")
        return render_template("approvals.html", pending=list(rows))

    @app.post("/trade/<trade_id>/field/<field>/approval")
    @roles("supervisor")
    def approval(trade_id, field):
        status = request.form.get("action")
        reason = request.form.get("reason", "").strip()
        if status not in {"approve", "reject"} or not reason:
            abort(400, "Approval decisions need a reason")
        db = database()
        with store.transaction(db):
            row = db.execute("SELECT * FROM fields WHERE trade_id=? AND path=?", (trade_id, field)).fetchone()
            if row is None:
                abort(404)
            if row["state"] != "pending":
                abort(409, "This proposal is no longer pending")
            if row["proposer"] == g.user["id"]:
                abort(403, "Self-approval is forbidden")
            approved = status == "approve"
            db.execute("UPDATE fields SET state=?,booking=?,reason=? WHERE trade_id=? AND path=?",
                       ("approved" if approved else "rejected", row["proposed"] if approved else row["booking"],
                        reason, trade_id, field))
            store.event(db, g.user["id"], "correction_approved" if approved else "correction_rejected",
                        trade_id, field, reason=reason, proposer=row["proposer"])
            store.affirm_if_complete(db, trade_id, g.user["id"])
        return redirect(url_for("approvals"))

    @app.post("/settings/lambda")
    @roles("supervisor")
    def change_lambda():
        try:
            value = float(request.form.get("lambda", ""))
        except ValueError:
            abort(400, "Invalid lambda")
        if not math.isfinite(value) or value < 0:
            abort(400, "Invalid lambda")
        db = database()
        with store.transaction(db):
            old = db.execute("SELECT value FROM settings WHERE name='lambda'").fetchone()[0]
            db.execute("UPDATE settings SET value=? WHERE name='lambda'", (value,))
            store.event(db, g.user["id"], "lambda_changed", old=old, new=value)
        return redirect(url_for("overview"))

    @app.get("/audit")
    @roles("auditor")
    def audit_log():
        rows = database().execute("SELECT audit.*,users.name FROM audit LEFT JOIN users ON actor=users.id ORDER BY audit.id")
        return render_template("audit.html", events=list(rows))

    return app
