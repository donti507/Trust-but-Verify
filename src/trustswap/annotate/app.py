"""Loopback-only annotation UI with server-side assignment and role checks."""
from functools import wraps
import hmac
import json
import secrets

import bcrypt
from flask import Flask, Response, abort, g, jsonify, redirect, render_template, request, session, url_for

from trustswap.annotate import adjudicate, checker, labels, stats, store, transfer, workflow
from trustswap.extract.fields import PATHS

ANNOTATORS = ("A1", "A2")


def roles(*allowed):
    def guard(view):
        @wraps(view)
        def checked(*args, **kwargs):
            if g.user is None:
                return redirect(url_for("login"))
            if g.user["role"] not in allowed:
                abort(403)
            return view(*args, **kwargs)
        return checked
    return guard


def csrf():
    return session.setdefault("csrf", secrets.token_urlsafe(24))


def database():
    return g.db


def source_for(doc):
    source = workflow.document(database(), doc)
    if source is None:
        abort(404)
    return source


def assigned_doc(doc):
    assigned = workflow.assignment(database(), doc, g.user["role"])
    if assigned is None:
        abort(404)
    return assigned, source_for(doc)


def create_app(path="annotation.db", *, secret=None):
    if secret is not None and len(secret.encode()) < 32:
        raise ValueError("Use at least 32 bytes for ANNOTATION_SECRET")
    store.initialize(path)
    app = Flask(__name__)
    app.config.update(SECRET_KEY=secret or secrets.token_bytes(32), SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE="Strict", SESSION_COOKIE_NAME="trustswap_annotation", MAX_CONTENT_LENGTH=16 * 1024 * 1024,
                      TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])
    dummy_password = bcrypt.hashpw(b"local-inactive-account", bcrypt.gensalt())
    app.jinja_env.globals.update(csrf=csrf, field_label=labels.label, structures=labels.STRUCTURES,
                                 special={adjudicate.STRUCTURE: "Swap structure", adjudicate.FLAGS: "Document characteristics"})

    @app.before_request
    def authenticate():
        g.user = None
        if request.endpoint == "static":
            return
        g.db = store.connect(path)
        name = session.get("user")
        if name:
            g.user = g.db.execute("SELECT id,role FROM users WHERE id=?", (name,)).fetchone()
        if request.method == "POST":
            supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf", "")
            expected = session.get("csrf")
            if expected is None or not hmac.compare_digest(str(expected).encode(), supplied.encode()):
                abort(400, "Reload the page to refresh the form token.")

    @app.teardown_appcontext
    def close_db(_error):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    @app.after_request
    def private(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
            "form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(413)
    def request_error(error):
        return render_template("error.html", error=error.description), error.code

    @app.route("/login", methods=["GET", "POST"])
    def login():
        error, name = None, ""
        if request.method == "POST":
            name = request.form.get("username", "").strip()
            password = request.form.get("password", "").encode()
            user = database().execute("SELECT * FROM users WHERE id=?", (name,)).fetchone()
            if len(password) <= 72 and bcrypt.checkpw(password, user["password"] if user else dummy_password) and user:
                session.clear()
                session["user"] = user["id"]
                return redirect(url_for("queue"))
            error = "Invalid user ID or password."
        return render_template("login.html", error=error, name=name), 401 if error else 200

    @app.post("/logout")
    @roles(*store.ROLES)
    def logout():
        session.clear()
        return redirect(url_for("login"))

    def queue_page(message=None, error=None):
        db, role, user = database(), g.user["role"], g.user["id"]
        locked = store.sealed(db)
        if role == "ADJ":
            pending = adjudicate.pending_documents(db)
            ready = sum(1 for _ in adjudicate.gold(db))
            return render_template("queue.html", locked=locked, pending=pending, ready=ready, message=message, error=error)
        if role == "A1" and not locked:
            rows = list(db.execute("""SELECT documents.id,screening.doc AS screened FROM documents
                                      LEFT JOIN screening ON documents.id=screening.doc ORDER BY screened,id"""))
        else:
            rows = list(db.execute("""SELECT cohort.doc AS id,cohort.control,work.stage FROM cohort
                                      LEFT JOIN work ON cohort.doc=work.doc AND user=?
                                      WHERE ?='A1' OR cohort.overlap=1
                                      ORDER BY CASE stage WHEN 'review' THEN 0 WHEN 'draft' THEN 1
                                               WHEN 'final' THEN 3 ELSE 2 END, id""", (user, role)))
        return render_template("queue.html", locked=locked, rows=rows, summary=stats.progress(db, role),
                               message=message, error=error)

    @app.get("/")
    @roles(*store.ROLES)
    def queue():
        return queue_page()

    @app.route("/inclusion/<doc>", methods=["GET", "POST"])
    @roles("A1")
    def screening(doc):
        source = source_for(doc)
        db = database()
        if store.sealed(db):
            return redirect(url_for("queue"))
        saved = db.execute("SELECT record FROM screening WHERE doc=?", (doc,)).fetchone()
        form = json.loads(saved["record"]) if saved else {"answers": {}, "reason": "", "duplicate_of": ""}
        error = None
        if request.method == "POST":
            form = {"answers": {key: request.form.get(key) == "yes" for key, _ in labels.CHECKLIST
                                if request.form.get(key) in {"yes", "no"}},
                    "include": {"include": True, "exclude": False}.get(request.form.get("decision")),
                    "reason": request.form.get("reason", ""), "duplicate_of": request.form.get("duplicate_of", "")}
            try:
                with store.transaction(db):
                    workflow.screen_document(db, doc, g.user["id"], form)
            except ValueError as exception:
                error = str(exception)
            else:
                return redirect(url_for("queue"))
        known = [row["id"] for row in db.execute("SELECT id FROM documents WHERE id!=? ORDER BY id", (doc,))]
        return render_template("inclusion.html", source=source, form=form, checklist=labels.CHECKLIST,
                               error=error, known=known), 400 if error else 200

    @app.get("/documents/<doc>")
    @roles(*ANNOTATORS)
    def edit(doc):
        assigned, source = assigned_doc(doc)
        work = store.get_work(database(), doc, g.user["id"])
        if work is None:
            return render_template("start.html", source=source,
                                   may_prefill=g.user["role"] == "A1" and not assigned["control"])
        data = work["data"]
        prefill = work["stage"] == "draft" and data["mode"] == "prefill"
        suggestions = {}
        if prefill:
            suggestions = {path: answer for path in PATHS if (answer := checker.suggestion(path, source["text"]))}
        # This object never contains candidates for a blind assignment. Review
        # messages are only created after the first human submission.
        view = {"stage": work["stage"], "revision": work["revision"], "mode": data["mode"], "form": data["form"],
                "checks": data["checks"], "suggestions": suggestions,
                "original": data["submitted"] if work["stage"] == "review" else None,
                "paths": {key: labels.paths(key) for key in labels.STRUCTURES},
                "save": url_for("save", doc=doc), "selection": url_for("selection", doc=doc)}
        return render_template("editor.html", source=source, groups=labels.grouped(), view=view, flags=labels.FLAGS)

    @app.post("/documents/<doc>/start")
    @roles(*ANNOTATORS)
    def start(doc):
        assigned, _ = assigned_doc(doc)
        with store.transaction(database()):
            workflow.start(database(), doc, g.user["id"], assigned, g.user["role"], request.form.get("mode") == "prefill")
        return redirect(url_for("edit", doc=doc))

    @app.post("/documents/<doc>/save")
    @roles(*ANNOTATORS)
    def save(doc):
        _, source = assigned_doc(doc)
        payload = request.get_json(silent=True)
        if (not isinstance(payload, dict) or set(payload) != {"form", "revision", "delta", "submit"}
                or type(payload["submit"]) is not bool):
            return jsonify(error="The saved annotation has an invalid format."), 400
        db, user = database(), g.user["id"]
        try:
            with store.transaction(db):
                current = store.get_work(db, doc, user)
                revision, stage = workflow.commit_work(db, doc, user, source["text"], current, **payload)
        except workflow.RevisionError as error:
            return jsonify(error=str(error)), 409
        except ValueError as error:
            return jsonify(error=str(error)), 400
        return jsonify(revision=revision, stage=stage)

    def select_value(source):
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or payload.keys() != {"path", "span"} or payload["path"] not in PATHS:
            return jsonify(error="Choose a field and source span."), 400
        try:
            left, right = labels.span(payload["span"], source["text"])
            if right - left > 4000:
                raise ValueError("Select only the relevant text for this field.")
        except ValueError as error:
            return jsonify(error=str(error)), 400
        value = labels.normalize_selection(payload["path"], source["text"][left:right])
        return jsonify(value=value)

    @app.post("/documents/<doc>/selection")
    @roles(*ANNOTATORS)
    def selection(doc):
        _, source = assigned_doc(doc)
        current = store.get_work(database(), doc, g.user["id"])
        if not current or current["stage"] == "final":
            abort(409, "This document has no editable draft.")
        return select_value(source)

    def adjudication_page(doc, message=None, error=None):
        review = adjudicate.comparison(database(), doc)
        if not review or not review["differences"]:
            abort(404)
        return render_template("adjudication.html", source=source_for(doc), comparison=review, error=error, message=message)

    @app.get("/adjudication/<doc>")
    @roles("ADJ")
    def adjudication(doc):
        return adjudication_page(doc)

    @app.post("/adjudication/<doc>/selection")
    @roles("ADJ")
    def adjudication_selection(doc):
        review = adjudicate.comparison(database(), doc)
        payload = request.get_json(silent=True)
        field = payload.get("path") if isinstance(payload, dict) else None
        if not review or not isinstance(field, str) or field not in review["differences"]:
            abort(404)
        return select_value(source_for(doc))

    @app.post("/adjudication/<doc>/decide")
    @roles("ADJ")
    def decide(doc):
        source = source_for(doc)
        reason = request.form.get("reason", "")
        status = request.form.get("status", "")
        choice = request.form.get("choice")
        try:
            offset = [int(request.form.get(key, "")) for key in ("start", "end")] if (
                status == "explicit" and choice == "manual"
            ) else None
            manual = {"status": status, "value": request.form.get("value", "") if status == "explicit" else "",
                      "span": offset, "note": reason, "confirmed": True}
            with store.transaction(database()):
                adjudicate.decide(database(), doc, request.form.get("field"), g.user["id"],
                                  source["text"], choice, reason, manual)
        except ValueError as error:
            return adjudication_page(doc, error=str(error)), 400
        return redirect(url_for("adjudication", doc=doc))

    @app.get("/stats")
    @roles(*store.ROLES)
    def statistics():
        role = g.user["role"]
        summaries = [stats.progress(database(), item) for item in (ANNOTATORS if role == "ADJ" else (role,))]
        requested = request.args.get("mode")
        mode = requested if requested in workflow.MODES else None
        overlap, agreement = stats.field_agreement(database(), mode) if role == "ADJ" else (None, ())
        return render_template("stats.html", summaries=summaries, overlap=overlap, agreement=agreement, mode=mode)

    def download(filename, payload):
        return Response(payload, mimetype="application/x-ndjson",
                        headers={"Content-Disposition": f"attachment; filename={filename}"})

    @app.get("/exchange/export")
    @roles(*ANNOTATORS)
    def export_labels():
        user = g.user["id"]
        try:
            payload = transfer.export(database(), user, g.user["role"])
        except ValueError as error:
            return queue_page(error=str(error)), 409
        return download(f"labels_{user}.jsonl", payload)

    @app.post("/exchange/import")
    @roles(*store.ROLES)
    def import_labels():
        file = request.files.get("labels")
        if file is None or not file.filename:
            return queue_page(error="Choose a labels JSONL file."), 400
        try:
            count = transfer.import_packet(database(), file.read().decode("utf-8-sig"), g.user["id"])
        except UnicodeError:
            return queue_page(error="Labels files must use UTF-8."), 400
        except ValueError as error:
            return queue_page(error=str(error)), 400
        return queue_page(message=f"Imported {count} finalized documents.")

    @app.get("/gold.jsonl")
    @roles("ADJ")
    def export_gold():
        with database():
            database().execute("BEGIN")
            payload = "".join(record.model_dump_json() + "\n" for record in adjudicate.gold(database()))
        return download("gold.jsonl", payload)

    return app
