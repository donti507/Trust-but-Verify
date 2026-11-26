# Local Synthetic demo desk

Install the extra with `python -m pip install -e '.[desk]'`. Create local users with
`python -m trustswap.desk create-user USERNAME ROLE`. The role is `analyst`,
`supervisor`, or `auditor`; the CLI prompts for a bcrypt password. Set a random
session secret of at least 32 characters in `DESK_SECRET_KEY`, then run
`python -m trustswap.desk` and open `http://localhost:8000`. The app binds to
loopback. `DESK_DB` or `--db` selects the SQLite database.

Analysts accept fields or submit proposed booking changes with a note.
Supervisors approve or reject proposals with a reason and can adjust the
policy threshold. Every field must be accepted or approved before a trade is
affirmed. Auditors can inspect the read-only event log. The desk collection and
its SQLite state are isolated from the research pipeline.
