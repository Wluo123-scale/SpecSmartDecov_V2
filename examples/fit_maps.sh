#!/bin/bash
# Fit the two maps of the reference study (paths are examples; data are not part of this repository).
set -euo pipefail
fastdecov fit "${PILLAR6:-pillar6_validation/output3-good}" out/pillar6 --grid 11x11 --figures --name pillar6
fastdecov fit "${PILLAR9:-pillar9_training/txt_output1}"  out/pillar9 --grid 21x15 --figures --name pillar9
fastdecov fit "${PILLAR9:-pillar9_training/txt_output1}"  out/pillar9_legacy --method legacy --name pillar9-legacy
