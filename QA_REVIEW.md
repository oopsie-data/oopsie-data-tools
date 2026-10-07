# QA review of an OopsieData bundle

You have received a zip such as `qa_bundle_01.zip` with about 100 robot episodes. For each episode, watch the videos, check the instruction and existing annotations, and write a QA note. Nothing else is edited.

## 1. Install

Requires Python ≥ 3.8 and [uv](https://docs.astral.sh/uv/) (or pip).

```bash
uv tool install "git+https://github.com/oopsie-data/oopsie-data-tools@qa_tool"
# or: pip install "git+https://github.com/oopsie-data/oopsie-data-tools@qa_tool"
```

## 2. Start

```bash
oopsie-data qa qa_bundle_01.zip --reviewer <your name>
```

| Option | Default | Meaning |
|---|---|---|
| `BUNDLE` | required | The zip, or its already unpacked folder |
| `--reviewer` | prompted | Your name, stored with every note |
| `--out` | folder of the zip | Where to unpack |
| `--port` | `5001` | Local port; change if already in use |
| `--no-browser` | off | Do not open the browser; go to `http://localhost:<port>/` yourself |

The zip is unpacked once, and a browser tab opens. Stop with `Ctrl-C`. Running the same command again resumes, and your notes are kept.

## 3. Review

For every episode in the left-hand list:

1. Watch the videos.
2. Check the task instruction, the episode details, and the existing annotations (read-only).
3. Write anything wrong or suspicious in **QA notes**, for example: missing or broken videos, a wrong instruction, a wrong success/failure label, a bad failure description, or odd robot data. If everything is fine, write `ok`.
4. Click **Save QA notes** (`Ctrl/Cmd+S`).

Navigation:
- `←` / `→` or `p` / `n` move between episodes.
- **Next without notes ▶▶** jumps to the next episode without a note.
- ✓✓ in the list marks episodes with a saved note.
- Tick "On save: next unannotated" to advance automatically after saving.

## 4. Send back

Send back only this file:

```
qa_bundle_01/qa_manifest.json
```

It sits next to the zip unless you used `--out`. Do not send the episodes.
