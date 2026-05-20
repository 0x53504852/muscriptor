# muscriptor

Audio-to-MIDI transcription using a transformer language model.

## Installation

```bash
pip install -e .
```

## Usage

```python
from muscriptor import TranscriptionModel

model = TranscriptionModel.load_model("muscriptor_weights.safetensors")
notes = model.transcribe("audio.wav", return_notes=True)
model.transcribe("audio.wav", output_midi="out.mid")
```

## CLI

```bash
muscriptor transcribe audio.wav -o out.mid
```
