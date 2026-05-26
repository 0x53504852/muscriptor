# Sanity-check transcription

Use this to catch silent regressions in the audio / mel-spectrogram /
inference pipeline. `piano-guitar.wav` and the reference `piano-guitar.mid`
live at the repo root and are kept around for exactly this purpose.

```bash
uv run muscriptor transcribe \
    --instruments acoustic_piano,acoustic_guitar \
    --model ./muscriptor_weights_8f59580c_750.safetensors \
    piano-guitar.wav -o /tmp/check.mid
```

Then compare `/tmp/check.mid` against `piano-guitar.mid`. The `note_on` /
`note_off` event stream (type, channel, pitch, velocity, delta-time) should
match exactly — any drift means something in the pipeline changed behavior.

```python
from mido import MidiFile

def events(path):
    return [
        (m.type, m.channel, m.note, m.velocity, m.time)
        for tr in MidiFile(path).tracks
        for m in tr
        if m.type in ("note_on", "note_off")
    ]

assert events("/tmp/check.mid") == events("piano-guitar.mid")
```
