# Export verification

The following checks passed on the cloud x86_64 Linux machine before export:

* Portable installation using only bundle-relative paths.
* Independent executor: four test groups, all ten primitives covered.
* Original data agreement:548282 pairs across28 files, zero mismatches.
* Tiny Transformer training and beam-five decoding: four/four exact matches.
* Checkpoint loading, one resumed update and saved step301.
* Building the included Linux/amd64 Docker image with TLS verification enabled.
* Repeating the complete data/training/decoding/resume smoke inside that image.

The Mac workflow uses that same Linux/amd64 image through Docker Desktop.
Physical Intel Mac and Apple Silicon execution were not tested here. Resource
limits and emulation performance depend on your Mac. No full seed or full-run
evaluation has completed. No pretrained weights are included.
