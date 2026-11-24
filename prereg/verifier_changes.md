# Verifier change log

After the `verifier-v1.0` tag, verifier rules and thresholds change only to fix
bugs. Log every change here **before** merging it, then update
`verifier_hash` in `prereg/verifier-v1.0.json` (recompute with
`python -c "from trustswap.verify.evaluate import verifier_hash; print(verifier_hash())"`).

| date | commit | rule | bug | fix | old hash → new hash |
| --- | --- | --- | --- | --- | --- |
