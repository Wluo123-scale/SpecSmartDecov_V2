# SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378
#!/bin/bash
# Fit the two maps of the reference study (paths are examples; data are not part of this repository).
# The map shape is detected automatically; each run ends with shortlist.md (lines to re-measure at higher resolution).
set -euo pipefail
fastdecov fit "${PILLAR6:-pillar6_validation/output3-good}" out/pillar6 --figures --name pillar6
fastdecov fit "${PILLAR9:-pillar9_training/txt_output}"     out/pillar9 --figures --name pillar9
fastdecov fit "${PILLAR9:-pillar9_training/txt_output}"     out/pillar9_legacy --method legacy --name pillar9-legacy
cat out/pillar6/shortlist.md out/pillar9/shortlist.md
