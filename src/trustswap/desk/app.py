"""Local desk web application and server-side authorization."""
from contextlib import closing
from datetime import timedelta
from functools import wraps
import hmac
import json
import math
import os
import secrets

import bcrypt
from flask import Flask, Response, abort, g, redirect, render_template, request, session, url_for

from trustswap.desk import exceptions, exports, imports, lifecycle, matcher, presentation, store, upload_review, uploads


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
    with closing(store.connect(path)) as db, store.transaction(db):
        exceptions.reconcile(db)
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
            if request.endpoint in {"upload_confirmation", "propose_lifecycle"}:
                request.max_content_length = uploads.MAX_BYTES + 64_000  # Allow multipart form overhead.
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

    def render_overview(import_report=None, error=None, submitted_lambda=None, upload_status=None, upload_id=""):
        db = database()
        query = """SELECT trades.*, CASE WHEN EXISTS (
                    SELECT 1 FROM desk_trade_versions WHERE trade_id=trades.id AND event_type='full')
                    THEN 'Terminated' ELSE state END AS display_state FROM trades"""
        if g.user["role"] == "analyst":
            rows = db.execute(query + " WHERE assigned_to IS NULL OR assigned_to=? ORDER BY id", (g.user["id"],))
        else:
            rows = db.execute(query + " ORDER BY id")
        threshold = threshold_now(db)
        cases = exceptions.queue(db, analyst=g.user["id"] if g.user["role"] == "analyst" else None)
        overdue_only = request.args.get("overdue") == "1"
        visible = [case for case in cases if case["bucket"] and (not overdue_only or case["bucket"] == "Overdue")]
        return render_template("overview.html", trades=list(rows), threshold=threshold, import_report=import_report,
                               form_error=error, submitted_lambda=submitted_lambda, upload_status=upload_status,
                               upload_id=upload_id, allow_external=upload_review.allow_external(),
                               cases=visible, case_metrics=exceptions.metrics(cases), overdue_only=overdue_only)

    def trade_for_user(trade_id):
        trade = database().execute("SELECT * FROM trades WHERE id=?", (trade_id,)).fetchone()
        if trade is None:
            abort(404)
        if g.user["role"] == "analyst" and trade["assigned_to"] not in (None, g.user["id"]):
            abort(403)
        return trade

    def require_active(trade_id):
        if lifecycle.terminated(database(), trade_id):
            abort(409, "The trade is terminated and read-only")

    def threshold_now(db):
        return db.execute("SELECT value FROM settings WHERE name='lambda'").fetchone()[0]

    def comparison_rows(db, trade_id, threshold):
        rows = db.execute("""SELECT fields.*, field_evidence.raw, field_evidence.span FROM fields
                           LEFT JOIN field_evidence USING (trade_id,path)
                           WHERE trade_id=? ORDER BY path""", (trade_id,))
        return [presentation.comparison(row, threshold) for row in rows]

    def bulk_eligible(field):
        return not field["flagged"] and field["state"] in {"open", "rejected"}

    def render_review(trade_id, error=None, submitted=None, lifecycle_input=None):
        trade = trade_for_user(trade_id)
        db = database()
        threshold = threshold_now(db)
        fields = comparison_rows(db, trade_id, threshold)
        groups, segments = presentation.source_groups(trade["source"], fields)
        upload = db.execute("SELECT kind FROM desk_uploads WHERE trade_id=?", (trade_id,)).fetchone()
        case = db.execute("SELECT desk_cases.*,users.name AS owner_name FROM desk_cases "
                          "LEFT JOIN users ON owner=users.id WHERE trade_id=?", (trade_id,)).fetchone()
        analysts = exceptions.analyst_options(db) if case and g.user["role"] == "supervisor" else []
        version = lifecycle.current(db, trade_id)
        readonly = lifecycle.terminated(db, trade_id)
        pending_change = db.execute("SELECT 1 FROM desk_lifecycle WHERE trade_id=? AND status='Pending approval'",
                                    (trade_id,)).fetchone() is not None
        cleared = sum(field["state"] in {"accepted", "approved"} for field in fields)
        return render_template("review.html", trade=trade, fields=fields, groups=groups, segments=segments,
                               can_bulk_accept=any(bulk_eligible(field) for field in fields), threshold=threshold,
                               cleared=cleared, form_error=error, submitted=submitted or {}, upload=upload,
                               case=case, analysts=analysts, readonly=readonly, version=version,
                               lifecycle_types=lifecycle.EVENTS, lifecycle_input=lifecycle_input or {},
                               pending_change=pending_change)

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
            require_active(trade_id)
            eligible = [field["path"] for field in comparison_rows(db, trade_id, threshold_now(db))
                        if bulk_eligible(field)]
            if eligible:
                db.execute("UPDATE trades SET assigned_to=? WHERE id=? AND assigned_to IS NULL",
                           (g.user["id"], trade_id))
            for path in eligible:
                db.execute("UPDATE fields SET state='accepted' WHERE trade_id=? AND path=?", (trade_id, path))
                store.event(db, g.user["id"], "field_accepted", trade_id, path, via="matched_low_risk")
            store.affirm_if_complete(db, trade_id, g.user["id"])
            exceptions.reconcile(db, trade_id, g.user["id"])
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
                imports._fictional(field, value)
                if matcher.normalize(field, value) is None:
                    raise ValueError("Missing term")
            except (ValueError, ArithmeticError):
                return render_review(trade_id, "Invalid proposed value", submitted), 400
        with store.transaction(db):
            trade_for_user(trade_id)
            require_active(trade_id)
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
            exceptions.reconcile(db, trade_id, g.user["id"])
        return redirect(url_for("review", trade_id=trade_id))

    @app.get("/approvals")
    @roles("supervisor")
    def approvals():
        return render_approvals()

    def render_approvals(error=None, submitted=None):
        rows = database().execute(
            """SELECT fields.*, users.name AS analyst FROM fields JOIN users ON users.id=proposer
            WHERE state='pending' ORDER BY trade_id,path""")
        return render_template("approvals.html", pending=list(rows), form_error=error,
                               submitted=submitted or {}, lifecycle_requests=lifecycle.requests(database()),
                               field_label=imports.field_label)

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
            require_active(trade_id)
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
            exceptions.reconcile(db, trade_id, g.user["id"])
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
            exceptions.reconcile(db, actor=g.user["id"])
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

    @app.post("/confirmations/upload")
    @roles("analyst", "supervisor")
    def upload_confirmation():
        file = request.files.get("confirmation")
        reference = request.form.get("trade_id", "").strip()
        if file is None or not file.filename or request.form.get("fictional") != "yes":
            return render_overview(error="Choose a confirmation and confirm that all parties are fictional",
                                   upload_id=reference), 400
        try:
            if reference:
                require_active(reference)
            upload = uploads.read_upload(file.read(uploads.MAX_BYTES + 1), file.filename)
            if upload.fields is None:
                return render_overview(upload_status="Saved locally. External extraction is disabled.",
                                       upload_id=reference), 202
            trade_id = uploads.attach(database(), upload, g.user["id"], reference)
        except ValueError as error:
            return render_overview(error=str(error), upload_id=reference), 400
        except OSError:
            return render_overview(error="The desk could not access its local upload directory",
                                   upload_id=reference), 400
        return redirect(url_for("review", trade_id=trade_id))

    @app.get("/trade/<trade_id>/export/<kind>")
    @roles(*store.ROLES)
    def export_confirmation(trade_id, kind):
        if kind not in {"fpml", "swift"}:
            abort(404)
        trade = trade_for_user(trade_id)
        require_active(trade_id)
        if trade["state"] != "Affirmed":
            abort(409, "Only affirmed trades can be exported")
        try:
            content = uploads.export(database(), trade_id, kind)
        except ValueError as error:
            return render_review(trade_id, str(error)), 409
        extension, media_type = ("xml", "application/xml") if kind == "fpml" else ("fin", "text/plain")
        return Response(content, mimetype=media_type, headers={
            "Content-Disposition": f"attachment; filename=desk-confirmation.{extension}",
        })

    @app.post("/trade/<trade_id>/case/assign")
    @roles("supervisor")
    def assign_case(trade_id):
        try:
            owner = int(request.form.get("owner", ""))
            db = database()
            with store.transaction(db):
                trade_for_user(trade_id)
                require_active(trade_id)
                exceptions.assign(db, trade_id, owner, g.user["id"])
        except ValueError as error:
            message = str(error) if request.form.get("owner", "").isdecimal() else "Select an analyst"
            return render_review(trade_id, message), 400
        return redirect(url_for("review", trade_id=trade_id))

    @app.post("/trade/<trade_id>/case/chase")
    @roles("analyst", "supervisor")
    def chase_case(trade_id):
        mode = request.form.get("mode")
        if mode not in {"draft", "download"}:
            return render_review(trade_id, "Choose to preview or download a draft"), 400
        db = database()
        try:
            with store.transaction(db):
                trade_for_user(trade_id)
                require_active(trade_id)
                message = exceptions.chase(db, trade_id, g.user["id"], g.user["role"])
        except ValueError as error:
            return render_review(trade_id, str(error)), 409
        if mode == "download":
            return Response(message.as_bytes(), mimetype="message/rfc822",
                            headers={"Content-Disposition": "attachment; filename=counterparty-draft.eml"})
        return render_template("chase.html", trade_id=trade_id, message=message)

    @app.post("/trade/<trade_id>/case/reply")
    @roles("analyst", "supervisor")
    def acknowledge_case(trade_id):
        db = database()
        try:
            with store.transaction(db):
                trade_for_user(trade_id)
                require_active(trade_id)
                exceptions.acknowledge(db, trade_id, g.user["id"], g.user["role"])
        except ValueError as error:
            return render_review(trade_id, str(error)), 409
        return redirect(url_for("review", trade_id=trade_id))

    @app.get("/trade/<trade_id>/history")
    @roles(*store.ROLES)
    def history(trade_id):
        trade = trade_for_user(trade_id)
        db = database()
        return render_template("history.html", trade=trade, versions=lifecycle.versions(db, trade_id),
                               requests=lifecycle.requests(db, trade_id), field_label=imports.field_label)

    @app.post("/trade/<trade_id>/lifecycle/propose")
    @roles("analyst")
    def propose_lifecycle(trade_id):
        trade_for_user(trade_id)
        require_active(trade_id)
        db = database()
        form = request.form.to_dict()
        event_type = form.get("event_type")
        file = request.files.get("confirmation")
        attachment = None
        try:
            lifecycle._validate_trade(db, trade_id)
            if file and file.filename:
                if event_type != "amendment":
                    raise ValueError("Attachments are supported for amendments")
                if form.get("fictional") != "yes":
                    raise ValueError("Confirm that the attachment and parties are fictional")
                attachment = uploads.read_upload(file.read(uploads.MAX_BYTES + 1), file.filename)
                if attachment.fields is None:
                    raise ValueError("The attachment was saved locally. External extraction is disabled.")
            with store.transaction(db):
                trade_for_user(trade_id)
                lifecycle.propose(db, trade_id, event_type, form, g.user["id"], attachment)
        except (OSError, ValueError) as error:
            message = str(error) if isinstance(error, ValueError) else "The attachment could not be read"
            return render_review(trade_id, message, lifecycle_input=form), 400
        return redirect(url_for("history", trade_id=trade_id))

    @app.post("/lifecycle/<int:event_id>/decision")
    @roles("supervisor")
    def decide_lifecycle(event_id):
        action = request.form.get("action")
        reason = request.form.get("reason", "")
        if action not in {"approve", "reject"}:
            return render_approvals("Choose an approval decision"), 400
        db = database()
        try:
            with store.transaction(db):
                lifecycle.decide(db, event_id, g.user["id"], action == "approve", reason)
        except PermissionError:
            abort(403, "Self-approval is forbidden")
        except ValueError as error:
            return render_approvals(str(error)), 400
        return redirect(url_for("approvals"))

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
