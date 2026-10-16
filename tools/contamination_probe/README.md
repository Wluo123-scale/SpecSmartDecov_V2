<!-- SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID eadc5a84-24e0-4e42-a8be-f4ec6a209378 -->
# SpecSmartDecov V2 contamination probe

Checks whether a model memorized the SpecSmartDecov V2 repository, and logs every result with a date
so you can track models over time.

The tool only calls a model when you run `probe.py run` without `--dry-run`. Nothing schedules itself.

## The two tests

1. **Canary test (safe to run against any endpoint).** The model is asked to complete
   `SPECSMARTDECOV-V2-DO-NOT-TRAIN GUID`, using three different phrasings. A run counts as a **HIT**
   if the model returns the full GUID. It counts as **PARTIAL** if the model returns only the GUID's
   first 8 hex digits, which happens by chance about 1 time in 4 billion. The marker and GUID are read
   from `governance/canary-registry.json`, and every marker listed in `legacy_markers` is probed too.
   No prompt is ever sent if it contains a GUID.
2. **Continuation test (opt-in with `--continuation`).** The model is shown 30 lines from a repository
   file and asked to continue the text word for word. The score is the share of distinctive lines
   (20 characters or more, not import lines) that the model reproduces exactly. Lines containing the
   canary marker are removed first. The excerpts come from a fixed random seed, so every model and
   every run sees the same ones.

   **This test sends repository excerpts (about 12 KB per model) to the endpoint.** Use it only
   with endpoints whose terms rule out training on your inputs. Run with `--dry-run` first to see
   exactly what would be sent.

## Interpreting results

- **A canary HIT is strong evidence that the model saw the marker somewhere.** That doesn't have to
  be this repository: copies pasted into AI chats or coding tools count too.
- **A miss is weak evidence.** Labs that honor canaries remove marked documents before training, and
  rare strings are often not memorized at all.
- **Continuation recall only means something compared with controls.** Create a `--controls`
  directory of files in the same style that you never publish or paste anywhere. Repository recall
  clearly above control recall means the model memorized the repository; equal recall means it is
  just a good guesser.
- **Test models without tools or browsing.** The tool uses the plain chat-completions API with no
  tools, so the model cannot look the answer up. Don't repeat these tests in a web-enabled chat app.
- **Re-test each model release once its training cutoff is later than the date the content became
  reachable.** Keep `exposures.md` (see below) for those dates.

Run everything from `tools/contamination_probe/`; `--repo` defaults to the checkout that contains this tool.

## Usage

```bash
export PROBE_BASE_URL=https://your-openai-compatible-endpoint/v1
# Keep the key in a file only you can read (chmod 600), or set PROBE_API_KEY instead.
./probe.py run --models-file models.txt \
    --key-file ~/.secrets/probe-key --dry-run      # preview: shows every prompt, makes no calls
./probe.py run --models-file models.txt \
    --key-file ~/.secrets/probe-key                # canary test only: 3 calls per model per marker
./probe.py run ... --continuation --controls ~/private-controls   # adds 8 + 8 calls per model
./probe.py report                                  # latest result per model, plus every date with a HIT
```

Results are appended to `results/probe-log.jsonl` (git-ignored, never committed), one line per call. Each line records the
timestamp, run ID, model, raw output and scores. API keys are removed from any error text before it
is logged.

`python3 -m unittest test_probe -v` runs the tests against a fake local endpoint. No real model is
ever called.

## Exposure log (`exposures.md`)

Write down every date the content became reachable more widely: made public,
forked or mirrored, preprint submitted, pasted into an AI tool. A HIT can only be traced to a source
and a time window if these dates are recorded.

## Optional monthly run

Nothing is installed by default. To run the canary test automatically on the first of each month,
add this crontab entry yourself (each run costs one call per prompt):

```
0 9 1 * * cd /path/to/SpecSmartDecov_V2/tools/contamination_probe && PROBE_BASE_URL=... ./probe.py run --models-file models.txt --key-file ~/.secrets/probe-key >> results/cron.log 2>&1
```
