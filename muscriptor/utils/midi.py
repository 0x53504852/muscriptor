"""MIDI output utilities."""

from pathlib import Path

from muscriptor.tokenizer.notes import Note, note2note_event, note_event2midi


def notes_to_midi(notes: list[Note], velocity: int = 100, tempo_bpm: int = 120):
    """Convert a list of Note objects to a mido MidiFile."""
    note_events = note2note_event(notes)
    tempo_us = int(60_000_000 / tempo_bpm)
    return note_event2midi(
        note_events, output_file=None, velocity=velocity, tempo=tempo_us
    )


def save_midi(
    notes: list[Note],
    path: str | Path,
    velocity: int = 100,
    tempo_bpm: int = 120,
) -> None:
    """Save a list of Note objects as a MIDI file."""
    midi = notes_to_midi(notes, velocity=velocity, tempo_bpm=tempo_bpm)
    midi.save(str(path))


def notes_to_multitrack_midi(
    notes: list[Note],
    instrument_name,
    drum_map=None,
    velocity: int = 100,
    tempo_bpm: int = 120,
    ticks_per_beat: int = 480,
):
    """Notes -> a type-1 mido MidiFile with one named track per instrument.

    Track 0 holds the tempo; then one track per (is_drum, program), pitched instruments by program then drums, named
    `instrument_name(note)`, with a program change (drums: channel 10). drum_map(pitch) maps the drum pitches (e.g.
    muscriptor.drums.to_general_midi). The notes are first cleaned as the MIDI writer of the released models does
    (validate_notes, then a note of a (program, pitch) is cut at the next onset of the same (program, pitch)), so
    note-on / note-off pairs never overlap on a channel.
    """
    import copy

    from mido import Message, MetaMessage, MidiFile, MidiTrack, bpm2tempo, second2tick

    from muscriptor.tokenizer.notes import trim_overlapping_notes, validate_notes

    notes = [copy.copy(n) for n in notes]
    if drum_map is not None:
        for n in notes:
            if n.is_drum:
                n.pitch = int(drum_map(n.pitch))
    notes = trim_overlapping_notes(validate_notes(notes, fix=True), sort=True)
    tempo = bpm2tempo(tempo_bpm)
    midi = MidiFile(type=1, ticks_per_beat=ticks_per_beat)
    conductor = MidiTrack()
    conductor.append(MetaMessage("set_tempo", tempo=tempo, time=0))
    midi.tracks.append(conductor)
    by_inst: dict[tuple[bool, int], list[Note]] = {}
    for n in notes:
        by_inst.setdefault((n.is_drum, n.program), []).append(n)
    channels = [c for c in range(16) if c != 9]
    for k, key in enumerate(sorted(by_inst)):
        is_drum, program = key
        inst_notes = by_inst[key]
        channel = 9 if is_drum else channels[min(k, len(channels) - 1)]
        track = MidiTrack()
        track.append(
            MetaMessage("track_name", name=instrument_name(inst_notes[0]), time=0)
        )
        track.append(
            Message(
                "program_change",
                program=0 if is_drum else min(program, 127),
                channel=channel,
                time=0,
            )
        )
        events = []  # (tick, 0 = off / 1 = on, pitch)
        for n in inst_notes:
            events.append(
                (round(second2tick(n.onset, ticks_per_beat, tempo)), 1, n.pitch)
            )
            events.append(
                (round(second2tick(n.offset, ticks_per_beat, tempo)), 0, n.pitch)
            )
        events.sort()
        now = 0
        for tick, on, pitch in events:
            track.append(
                Message(
                    "note_on" if on else "note_off",
                    note=pitch,
                    velocity=velocity if on else 0,
                    channel=channel,
                    time=tick - now,
                )
            )
            now = tick
        midi.tracks.append(track)
    return midi
