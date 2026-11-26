# MuScriptor V2 "Best + 13-decode vote": annotating your own songs

This page shows how to turn your own recordings into MIDI with our best MuScriptor V2 model. You can use a
single decode (fast) or the 13-decode consensus vote (more accurate).

## The model

MuScriptor transcribes mixed audio (a full band, a choir, a piano, ...) into notes: pitch, onset, offset and
instrument, drums included. It is a 24-layer decoder-only transformer (medium size, ~300M parameters). It reads
5 s log-mel windows of mono 16 kHz audio and writes MT3-style note tokens. The V2 "best" checkpoint
(`muscriptor_v2_remix_rl_20261126`, step 5000) was trained on the MuScriptor V2 dataset, then fine-tuned with RL.

There are two ways to decode:

- **single decode**: greedy, 5 s windows every 4 s. Each window keeps the notes of its central part.
- **ensemble (vote)**: 13 decodes of every window: different guidance strengths, window grids and sampling
  temperatures. A note is kept when at least 4 of the 13 decodes agree on it (same pitch, onsets within 50 ms). It
  takes the median onset and offset. Recipe **A** is for decoding with an instrument list, recipe **B** for decoding
  without one.

## Weights

```
/lustre/scwpod02/client/kyutai/music/models/muscriptor_v2_best_20261126/model.safetensors
```

This is the training repository's export, used as is (1.2 GB, float32). The package reads the model size and the
compact vocabulary from the weights. You do not need any conversion.

## Installation (uv)

```bash
git clone git@github.com:0x53504852/muscriptor.git
cd muscriptor
git checkout v2-best-ensemble
uv sync
uv run muscriptor list-instruments
```

- **WAV** files work out of the box.
- **mp3 / m4a / opus / webm** files are decoded by `ffmpeg`, the same way as the training data. Put `ffmpeg` on your
  `PATH`, or point `MUSCRIPTOR_FFMPEG` to its binary (`ffprobe` must sit next to it).
- **FLAC / OGG** files are read by `soundfile` if it is installed, and by ffmpeg otherwise.

## Running on the cluster (GPU)

Use a GPU node. The examples below set `MODEL` to the weights:

```bash
MODEL=/lustre/scwpod02/client/kyutai/music/models/muscriptor_v2_best_20261126/model.safetensors
```

A SLURM job for a folder of songs (partition `kyutai`, 1 GPU), saved as `transcribe.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=muscriptor
#SBATCH --partition=kyutai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=muscriptor-%j.out
MODEL=/lustre/scwpod02/client/kyutai/music/models/muscriptor_v2_best_20261126/model.safetensors
cd ~/muscriptor
uv run muscriptor transcribe --model $MODEL --ensemble auto \
    --instruments electric_bass,distorted_electric_guitar,drums,voice \
    ~/my_songs/ -o ~/my_songs_midi/ --save-notes csv
```

```bash
sbatch transcribe.sbatch
```

For a quick interactive run:

```bash
srun -p kyutai --gres=gpu:1 -c 8 --mem=64G --pty bash
uv run muscriptor transcribe --model $MODEL song.wav -o out/
```

**CPU fallback.** `--device cpu` works, but it is very slow. A sparse 25 s Bach chorale took 19 s on 16 cores; a
dense band song takes several times its own length. Use the single decode only on CPU, never the ensemble:

```bash
uv run muscriptor transcribe --model $MODEL --device cpu clip.wav -o out_cpu/
```

## Single decode or ensemble?

All numbers below were measured on one H100 with this package, on 8 songs: 5 songs of our test set (rock band songs
and two orchestral / folk pieces) and 3 Bach10 chorales, 14.9 minutes of audio in total. The times include audio
reading but not the model load (~1-10 s).

| decoding                      | command                                    | time per minute of audio | onset F1 (8 songs) |
|-------------------------------|--------------------------------------------|--------------------------|--------------------|
| single decode, instruments    | `--instruments ...`                        | 0.6-0.7 s                | 72.6               |
| single decode, no instruments | (none)                                     | 0.5 s                    | 70.8               |
| ensemble A (instruments)      | `--ensemble auto --instruments ...`        | 3.2 s                    | 75.7               |
| ensemble B (no instruments)   | `--ensemble auto`                          | 4.0 s                    | 74.3               |

- **Quality.** On our larger benchmarks, the vote adds about **+1.8 onset F1 on band songs** and about **+2.5 on the
  paper composite** (8 public test sets). It also gains on offsets and frames. On the 8 songs above it adds +3.0
  (prompted) and +3.5 (unprompted).
- **Cost.** The 13 decodes run in one batched pass. The windows of several songs are decoded together, the shared
  audio prefix of a window is computed once, and finished windows leave the batch. So the ensemble costs about
  **5x a single decode** with this package, instead of 13x. Run as 13 separate decodes with the training code, the
  same 8 songs take about 13 x 20 s = 260 s, against 48 s here.
- **One song at a time**, the GPU is far from full. Each 5 s window is decoded token by token, and a song takes
  as many steps as its longest window. The ensemble then costs about as much as the single decode. For example, a
  2:42 rock song took 5-10 s for the single decode and 11 s for ensemble A. An 18-minute file took 50 s with
  ensemble A.
- Use the **ensemble** for annotations you will keep, and the **single decode** for a quick look.

## Instruments: when and how

`--instruments` gives the model the list of instruments present in the song (the "prompt"). Give it when you
know the instrumentation:

- the model then writes notes only for these instruments, and assigns them better;
- with `--ensemble auto`, a list selects recipe A (prompted), and no list selects recipe B (unprompted). You can also
  choose with `--ensemble A` or `--ensemble B`. Recipe A was tuned with a list and recipe B without one.

Only list the instruments that really play. A wrong list costs notes. If you are unsure, leave the list out.

The names accepted by `--instruments` (`uv run muscriptor list-instruments` prints them all):

```
acoustic_piano electric_piano chromatic_percussion organ acoustic_guitar clean_electric_guitar
distorted_electric_guitar acoustic_bass electric_bass violin viola cello contrabass harp timpani strings
synth_strings voice orchestra_hit trumpet trombone tuba french_horn brass high_sax tenor_sax baritone_sax oboe
english_horn bassoon clarinet pipe synth_lead synth_pad fx_brightness fx_goblins drums
```

The list also contains rarer General MIDI groups (sitar, banjo, koto, steel_drums, ...). Some notes on the groups:

- `voice` covers singing, lead and choir;
- `strings` is a string ensemble;
- `pipe` covers flute, piccolo, recorder and pan flute;
- `high_sax` covers soprano and alto sax.

Examples:

```bash
# a rock song: ensemble A
uv run muscriptor transcribe --model $MODEL --ensemble auto \
    --instruments electric_bass,distorted_electric_guitar,drums,voice song.wav -o out/

# unknown instrumentation: ensemble B
uv run muscriptor transcribe --model $MODEL --ensemble auto song.wav -o out/
```

## Outputs

For every input `name.wav`, the output folder gets:

- `name.mid`, a type-1 MIDI file:
  - one named track per instrument (`acoustic_piano`, `electric_bass`, `voice`, `drums`, ...), each with its program
    change;
  - drums on channel 10, written as General MIDI drum notes;
  - tempo 120 bpm, so MIDI seconds are audio seconds;
  - velocity 100 everywhere: the model does not estimate loudness.
- with `--save-notes csv` and/or `--save-notes json`: `name.notes.csv` / `name.notes.json`, one row per note with
  `onset`, `offset` (seconds), `pitch`, `instrument`, `program` (General MIDI program of the group) and `is_drum`.

A note's program is the first program of its instrument group (e.g. 0 for `acoustic_piano`, 33 for `electric_bass`,
52 for `voice`).

**Drums.** The model writes one canonical pitch per drum class:

| class                  | pitch      |
|------------------------|------------|
| kick                   | 36         |
| snare                  | 38         |
| side stick             | 37         |
| hi-hat closed / pedal / open | 42 / 44 / 46 |
| toms low / mid / high  | 43 / 45 / 48 |
| crash                  | 49         |
| ride                   | 51         |
| ride bell              | 53         |
| china                  | 52         |
| splash                 | 55         |

These are General MIDI notes already. Guitar Pro articulation numbers (rim shot, chokes, ...) are mapped to their
parent General MIDI drum. Two hits that differ only by a GM variant (e.g. crash 1 and crash 2) are both written as
the canonical one.

`-f json` / `-f jsonl` write the note events instead of MIDI, and `--notes` prints the notes.

## Known limits

- **Choir and soft attacks**: our onsets come about 20 ms earlier than some references (Bach10, Dagstuhl
  ChoirSet), which mark the onset later in a soft attack. The notes are found, but early.
- **Note ends on piano**: the model ends a piano note where the key is released, as MAESTRO-style labels do. It does
  not wait for the sustain pedal. References that include the pedal have longer notes.
- **No lyrics, no velocities, no techniques** (bends, slides, mutes): only pitch, timing and instrument.
- Mono only: the model hears the channel mean.
- The instrument names are groups (e.g. `distorted_electric_guitar` covers overdrive and distortion). Two guitars of
  the same group come out on one track.

## Many songs

Give several files or folders. They are batched together, and the MIDI files are written as songs finish:

```bash
uv run muscriptor transcribe --model $MODEL --ensemble auto songs/ -o midi/ --save-notes csv,json
```

- A folder is read for audio files (`.wav .flac .ogg .mp3 .m4a .aac .opus .webm .aif .aiff`). Add `--recursive` to
  include subfolders.
- `-o DIR/` (with a trailing `/`, or an existing directory) writes `DIR/<name>.mid` for every input.
- One `--instruments` list applies to every song of the run. Group songs with the same instrumentation into one
  run.

From Python:

```python
from pathlib import Path
from muscriptor import TranscriptionModel

model = TranscriptionModel.load_model(MODEL)          # GPU if available
songs = ["a.wav", "b.mp3"]
for i, notes in model.transcribe_notes(songs, instruments=["electric_bass", "drums", "voice"], ensemble="auto"):
    Path(songs[i]).with_suffix(".mid").write_bytes(model.notes_to_midi(notes))
    print(songs[i], len(notes), "notes")
```

- `song_instruments=[[...], None, ...]` sets one list per song.
- `ensemble=None` runs the single decode.
- `return_decodes=True` also returns the 13 individual decodes.

## Troubleshooting

- **Out of GPU memory.** The package sizes its batches from the free GPU memory. If a batch does not fit anyway, it
  backs off by itself:
  - songs (or windows) still decoding are moved to a later batch;
  - a log line `out of GPU memory ... decoded again later` tells you when this happens.

  The output is unchanged. If it happens often (smaller GPU, or a GPU shared with another job), cap the number of
  window decodes held at once, for example `--batch-size 512`, or 256 on a 40 GB GPU.
- **Long songs** are fine: a song is cut into batches of 64 windows (about 4 minutes). On an H100, an 18-minute file
  decoded with ensemble A without any setting (50 s), and with ensemble B with `--batch-size 512` (76 s).
- **Slow**: check that the job got a GPU (`[muscriptor]` log lines show the time per input). On CPU only use the
  single decode.
- **`cannot read ...: install ffmpeg`**: the file is not WAV. Install ffmpeg or convert the file to WAV.
- **Unknown instrument name**: run `uv run muscriptor list-instruments`.
- **Reproducibility**: the sampled decodes of the ensemble are seeded from the song's path. The same file at the
  same path gives the same notes on the same GPU. On another GPU or batch, the results can differ slightly, at the
  level of bf16 rounding.
