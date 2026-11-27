"""Local desk web application and server-side authorization."""
from datetime import timedelta
from functools import wraps
import hmac
import json
import math
import os
import secrets

import bcrypt
from flask import Flask, Response, abort, g, redirect, render_template, request, session, url_for

from trustswap.desk import exports, imports, matcher, presentation, store


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

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(413)
    def friendly_error(error):
        return render_template("error.html", error=error), error.code

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
        username = ""
        if request.method == "POST":
            db = database()
            username = request.form.get("username", "")
            user = db.execute("SELECT * FROM users WHERE name=?", (username.strip(),)).fetchone()
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
        return render_template("login.html", error=error, username=username), (401 if error else 200)

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
        return render_overview()

    def render_overview(import_report=None, error=None, submitted_lambda=None):
        db = database()
        if g.user["role"] == "analyst":
            rows = db.execute("SELECT * FROM trades WHERE assigned_to IS NULL OR assigned_to=? ORDER BY id",
                              (g.user["id"],))
        else:
            rows = db.execute("SELECT * FROM trades ORDER BY id")
        threshold = db.execute("SELECT value FROM settings WHERE name='lambda'").fetchone()[0]
        return render_template("overview.html", trades=list(rows), threshold=threshold, import_report=import_report,
                               form_error=error, submitted_lambda=submitted_lambda)

    def trade_for_user(trade_id):
        trade = database().execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        if trade is None:
            abort(404)
        if g.user["role"] == "analyst" and trade["assigned_to"] not in (None, g.user["id"]):
            abort(403)
        return trade

    def threshold_now(db):
        return db.execute("SELECT value FROM settings WHERE name='lambda'").fetchone()[0]

    def comparison_rows(db, trade_id, threshold):
        rows = db.execute("""SELECT fields.*, field_evidence.raw, field_evidence.span FROM fields
                           LEFT JOIN field_evidence USING (trade_id,path)
                           WHERE trade_id=? ORDER BY path""", (trade_id,))
        return [presentation.comparison(row, threshold) for row in rows]

    def bulk_eligible(field):
        return not field["flagged"] and field["state"] in {"open", "rejected"}

    def render_review(trade_id, error=None, submitted=None):
        trade = trade_for_user(trade_id)
        db = database()
        threshold = threshold_now(db)
        fields = comparison_rows(db, trade_id, threshold)
        groups, segments = presentation.source_groups(trade["source"], fields)
        cleared = sum(field["state"] in {"accepted", "approved"} for field in fields)
        return render_template("review.html", trade=trade, fields=fields, groups=groups, segments=segments,
                               can_bulk_accept=any(bulk_eligible(field) for field in fields), threshold=threshold, cleared=cleared, form_error=error, submitted=submitted or {})

    @app.get("/trade/<trade_id>")
    @roles(*store.ROLES)
    def review(trade_id):
        return render_review(trade_id)

    @app.post("/trade/<trade_id>/accept-matched")
    @roles("analyst")
    def accept_matched(trade_id):
        db = database()
        with store.transaction(db):
            trade_for_user(trade_id)
            eligible = [field["path"] for field in comparison_rows(db, trade_id, threshold_now(db))
                        if bulk_eligible(field)]
            if eligible:
                db.execute("UPDATE trades SET assigned_to=? WHERE id=? AND assigned_to IS NULL",
                           (g.user["id"], trade_id))
            for path in eligible:
                db.execute("UPDATE fields SET state='accepted' WHERE trade_id=? AND path=?", (trade_id, path))
                store.event(db, g.user["id"], "field_accepted", trade_id, path, via="matched_low_risk")
            store.affirm_if_complete(db, trade_id, g.user["id"])
        return redirect(url_for("review", trade_id=trade_id))

    @app.post("/trade/<trade_id>/field/<field>/review")
    @roles("analyst")
    def review_field(trade_id, field):
        db = database()
        action = request.form.get("action")
        submitted = {"field": field, "note": request.form.get("note", ""), "value": request.form.get("value", "")}
        note, value = submitted["note"].strip(), submitted["value"].strip()
        trade_for_user(trade_id)
        if not db.execute("SELECT 1 FROM fields WHERE trade_id=? AND path=?", (trade_id, field)).fetchone():
            abort(404)
        if action not in {"accept", "propose"} or (action == "propose" and not (note and value)):
            return render_review(trade_id, "Proposed corrections need a value and note", submitted), 400
        if action == "propose":
            try:
                if matcher.normalize(field, value) is None:
                    raise ValueError("Missing term")
            except (ValueError, ArithmeticError):
                return render_review(trade_id, "Invalid proposed value", submitted), 400
        with store.transaction(db):
            trade_for_user(trade_id)
            row = db.execute("SELECT * FROM fields WHERE trade_id=? AND path=?", (trade_id, field)).fetchone()
            if row is None:
                abort(404)
            if row["state"] not in {"open", "rejected"}:
                return render_review(trade_id, "This field already has a review decision", submitted), 409
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
        return render_approvals()

    def render_approvals(error=None, submitted=None):
        rows = database().execute(
            """SELECT fields.*, users.name AS analyst FROM fields JOIN users ON users.id=proposer
            WHERE state='pending' ORDER BY trade_id,path""")
        return render_template("approvals.html", pending=list(rows), form_error=error, submitted=submitted or {})

    @app.post("/trade/<trade_id>/field/<field>/approval")
    @roles("supervisor")
    def approval(trade_id, field):
        status = request.form.get("action")
        raw_reason = request.form.get("reason", "")
        reason = raw_reason.strip()
        if status not in {"approve", "reject"} or not reason:
            return render_approvals("Approval decisions need a reason",
                                    {"trade": trade_id, "field": field, "reason": raw_reason}), 400
        db = database()
        with store.transaction(db):
            row = db.execute("SELECT * FROM fields WHERE trade_id=? AND path=?", (trade_id, field)).fetchone()
            if row is None:
                abort(404)
            if row["state"] != "pending":
                return render_approvals("This proposal is no longer pending"), 409
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
        submitted = request.form.get("lambda", "")
        try:
            value = float(submitted)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid lambda")
        except ValueError:
            return render_overview(error="Enter a finite nonnegative threshold", submitted_lambda=submitted), 400
        db = database()
        with store.transaction(db):
            old = db.execute("SELECT value FROM settings WHERE name='lambda'").fetchone()[0]
            db.execute("UPDATE settings SET value=? WHERE name='lambda'", (value,))
            store.event(db, g.user["id"], "lambda_changed", old=old, new=value)
        return redirect(url_for("overview"))

    @app.post("/bookings/import")
    @roles("supervisor")
    def upload_bookings():
        file = request.files.get("bookings")
        if file is None:
            return render_overview(error="Select a booking CSV"), 400
        try:
            report = imports.import_upload(database(), file.read(), g.user["id"])
        except (ValueError, UnicodeError, ArithmeticError):
            return render_overview(error="Invalid, ambiguous or already reviewed booking data"), 400
        return render_overview(import_report=report) if report.rejected else redirect(url_for("overview"))

    @app.get("/audit.<kind>")
    @roles("auditor", "supervisor")
    def download_audit(kind):
        if kind not in {"csv", "json"}:
            abort(404)
        data, content_type = exports.export(database(), kind)
        return Response(data, mimetype=content_type,
                        headers={"Content-Disposition": f"attachment; filename=desk-audit.{kind}"})

    @app.get("/audit")
    @roles("auditor", "supervisor")
    def audit_log():
        return render_template("audit.html", events=list(exports.audit_rows(database())))

    return app
