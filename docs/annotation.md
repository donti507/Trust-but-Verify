# Local annotation for Step 4

Install the annotation extra from the checkout.

    python -m pip install -e ".[annotate]"

Each laptop should use its own private database. The default is
`annotation.db`. Set `ANNOTATION_DB` to a local path to change it. Register the
same three user IDs and roles on each installation. The CLI prompts for each
bcrypt password. Use a different private password for each account and do not
share credentials.

    python -m trustswap.annotate create-user annotator-one A1
    python -m trustswap.annotate create-user annotator-two A2
    python -m trustswap.annotate create-user adjudicator ADJ

Import the same local raw files and manifest on each laptop, using the same
seed. By default the seed is 20261026. Use `--seed` on the import command to
choose a different seed before starting the study.

    python -m trustswap.annotate import data/real/raw --manifest data/real/manifest.csv
    python -m trustswap.annotate

Open **http://localhost:8001** in a local browser. The app binds to the loopback
interface. An optional `ANNOTATION_SECRET` of at least 32 bytes keeps sessions
valid across restarts. Drafts and assignments resume from SQLite with either
session setup.

The manifest needs `doc_id`. Use `raw_path`, `local_path`, `file_path`,
`filename` or `path` to point to a file under the given raw directory. Otherwise
the tool looks for one file named after the document ID. Other supported columns
include `source_url` (or `document_url`), `sha256`, `collection` and `split`.
Use the real collection and leave the split empty until assigned. The importer
skips manifest rows marked `real_test` and paths under a `real_test` directory.
It reads HTML, text and PDF text layers locally. Offsets count Unicode characters
in the clean text shown in the app. The database also retains the mapping to
visible text before whitespace normalization.

## Inclusion and labelling

1. **A1** screens every imported confirmation using the five questions. Record the
   inclusion or exclusion reason and identify any duplicate. Finish the cohort
   before labelling. The final inclusion decision locks assignments. A1 gets all
   included documents. A2 gets a seeded 35% sample, rounded to the nearest
   document. A separate seeded draw chooses up to 10 included documents for A1's
   blind control. The same seed and inclusion decisions give the same assignments.
2. Open a document and choose the swap structure. Each active field needs an
   explicit value with a selected source span, or an absent or ambiguous status
   with a note. Select text on the left to copy the term into the focused field.
   Dates, amounts, currencies and rates are normalized when the selected text
   supplies the necessary information. The value remains editable. If a rate is
   written without a unit, check the document and enter the decimal fraction or
   an explicitly qualified rate. Review the document characteristics before
   submitting.
3. From a field heading, **Tab** advances to the next field and **A** marks the
   current one absent. From a value input, **Enter** submits. The local autosave
   captures the last field, draft contents and foreground editor time. Return to
   the queue to resume.
4. A1 may choose rule-based pre-fill when beginning a non-control document. Every
   field still needs human confirmation. Suggestions are marked **Unverified**
   until confirmed. The audit records whether each candidate was accepted or
   changed. A2 always starts with an empty blind form. A1's 10 control documents
   have neither pre-fill nor checker guidance.
5. The checker runs only after the first submission on eligible documents.
   Re-check each flagged field. Every correction to a submitted value, span,
   status or note needs a reason. A finalized label is locked. The audit retains
   the original answer, final answer, reason, author and time.

## Daily exchange and adjudication

1. A1 exports `labels_<user>.jsonl` from the queue after finishing inclusion,
   even if no labels are ready yet. On a separate laptop, A2 and ADJ import that
   file to reproduce the cohort.
2. Each annotator exports their own finalized labels at the end of the day.
   Transfer these files through the approved private shared folder. ADJ imports
   both files using **Import labels**. Importing the same file twice is safe.
   The app verifies account IDs, corpus and document hashes, assignments and
   form structure. Conflicts are rejected and existing labels are retained.
   Drafts stay in the originating database.
3. ADJ sees only disagreements between the independently finalized overlap
   labels. Select A1, A2 or enter the final term and source span, then record
   the reason. Use **Export gold.jsonl** for the repository's gold schema,
   including raw and text hashes, source URL, annotator IDs and adjudication
   status. Overlap documents become eligible after both annotators finish and
   ADJ resolves any disagreements. A1-only documents are eligible once finalized.
4. Stats reports status-and-value agreement and Cohen’s κ for statuses in the
   completed overlap. ADJ can filter by A1 mode to inspect the blind controls.
   Agreement uses paired paths, and undefined κ is shown
   with a dash. Annotators see their own activity stats. The flagged rate uses
   checker-eligible submitted fields. The change rate uses flagged fields
   whose re-check was finalized.

Keep the documents and database local and private. Do not upload documents,
run them through a model, open another annotator’s label file, or commit the
database, exchanged labels, gold export or any document. The app uses the
stored document text and local rule-based parsing for this workflow.
