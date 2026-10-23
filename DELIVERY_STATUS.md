# Advisor handoff status — 2026-10-23

The current source/model snapshot and six-method results are under [submission/NW14_20261023](submission/NW14_20261023/README_TH.md). The old root demo remains historical.

- Source snapshots: 238 files checked against recorded SHA-256 bindings.
- Score grid: complete 360 selected cells; 5,400 per-flow rows. SAT and Mbps recomputed from those flow tables without new simulations.
- Raw packet/AP/queue archives: locally prepared for every selected cell. All members were read with ZIP CRC checks; packet/AP byte hashes match the frozen metadata for all 360 cells.
- Google Drive folder: [NW14_Advisor_20261023](https://drive.google.com/drive/folders/11CmXmEjSGi0BEo-YHGC2z49Rh-b2cGi1?usp=sharing). The user supplied this replacement destination. Its final sharing permissions have not yet been verified. **File upload is pending: the browser did not recognize file access and then stopped responding. A folder link alone does not mean files have been delivered.**
- Reproduction boundary: supplied-data verification is portable; a clean native ns-3 rebuild and complete from-scratch training are not verified by this export.

The six-method presentation scope was selected after viewing an original 11-method campaign. All conditions, loads and seeds of the six selected methods remain included; this is not a claim of superiority over every evaluated method.

All 360 supplied packet/AP cells were additionally recomputed with the portable verifier: SAT, Mbps, delivered-only p95 and exact delivered UID/time AP joins passed. All 360 queue byte hashes match the frozen audit table. This remains supplied-evidence verification, not new simulation.
