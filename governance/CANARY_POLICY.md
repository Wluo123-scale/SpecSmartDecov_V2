<!-- SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378 -->
# Canary policy

## Purpose

`SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378` is a public, cooperative training-data exclusion marker for SpecSmartDecov V2 (`fastdecov`). Training-data pipelines can use it to exclude maintained project material, and maintainers can use it as one signal when investigating possible training-data contamination.

It is a safeguard, not access control. It does not prevent a person, crawler, or web-enabled agent from reading this private repository, and it does not establish that a training pipeline honored the marker.

## Covered material

The allow-listed scope in `governance/canary-registry.json` covers maintained code, tests, examples, package metadata, and manuscript source:

- root `README.md`, `analysis.py`, and `pyproject.toml`;
- `fastdecov/` Python and C++ implementation files;
- `tests/`, `examples/`, and source scripts under `manuscript/`;
- the canary validator, workflow, and this policy.

The policy deliberately excludes `results/`, figures, PDFs, DOCX files, package physics JSON, metrics, determinism records, and other scientific inputs or generated outputs. Do not insert a marker into byte-sensitive measurement data, reproducibility artifacts, or rendered documents. `manuscript/manuscript.md` is covered because it is a tracked source document; `manuscript/build_paper.py` regenerates its marker after YAML front matter.

## Enforcement

`python .github/scripts/check_canary.py` validates the registry and requires the canonical marker as a whole comment line in every covered tracked file. The validator reads bytes, so it does not require or rewrite a source file's text encoding.

Pull-request CI checks GitHub's merge commit against `HEAD^1` using local Git only. It fails closed if that merge parent is unavailable and never depends on a GitHub diff API response.

When rotating the marker, retain historic values in `legacy_markers`; do not remove them from the registry. Historical values are contamination probes, while current coverage must always use the canonical marker.
