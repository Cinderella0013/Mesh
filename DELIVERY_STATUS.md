# Advisor handoff status — 2026-10-23

The current source/model snapshot and six-method results are under [submission/NW14_20261023](submission/NW14_20261023/README_TH.md). The old root demo remains historical.

- Source snapshots: 238 files checked against recorded SHA-256 bindings.
- Score grid: complete 360 selected cells; 5,400 per-flow rows. SAT and Mbps recomputed from those flow tables without new simulations.
- Raw packet/AP/queue archives: locally prepared for every selected cell. All members were read with ZIP CRC checks; packet/AP byte hashes match the frozen metadata for all 360 cells.
- Google Drive folder: [NW14_Advisor_20261023](https://drive.google.com/drive/folders/1FaiH6qkqzQTR62555kB9jTMIzKq7zxaB). General access was set to Anyone with the link / Viewer. **File upload still pending browser file-access permission recognition; a folder link alone does not mean files have been delivered.**
- Reproduction boundary: supplied-data verification is portable; a clean native ns-3 rebuild and complete from-scratch training are not verified by this export.

The six-method presentation scope was selected after viewing an original 11-method campaign. All conditions, loads and seeds of the six selected methods remain included; this is not a claim of superiority over every evaluated method.
