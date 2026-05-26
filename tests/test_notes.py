"""Tests for muscriptor/tokenizer/notes.py."""

import pytest

from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    MINIMUM_NOTE_DURATION_SEC,
    Note,
    NoteEvent,
    Event,
    EventTokenizer,
    NoteEventTokenizer,
    sort_notes,
    sort_note_events,
    validate_notes,
    trim_overlapping_notes,
    note_event2event,
    event2note_event,
    note_event2note,
    note2note_event,
)


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------


def test_sort_notes_by_onset():
    notes = [
        Note(False, 0, 2.0, 3.0, 60),
        Note(False, 0, 0.5, 1.5, 62),
        Note(False, 0, 1.0, 2.0, 64),
    ]
    sort_notes(notes)
    onsets = [n.onset for n in notes]
    assert onsets == sorted(onsets)


def test_sort_note_events_by_time():
    evs = [
        NoteEvent(False, 0, 3.0, 1, 60),
        NoteEvent(False, 0, 1.0, 1, 62),
        NoteEvent(False, 0, 2.0, 0, 60),
    ]
    sort_note_events(evs)
    times = [e.time for e in evs]
    assert times == sorted(times)


# ---------------------------------------------------------------------------
# validate_notes
# ---------------------------------------------------------------------------


def test_validate_notes_fixes_short_duration():
    note = Note(False, 0, 1.0, 1.001, 60)  # offset - onset < 0.01
    result = validate_notes([note], fix=True)
    assert result[0].offset >= result[0].onset + MINIMUM_NOTE_DURATION_SEC


def test_validate_notes_fixes_inverted():
    note = Note(False, 0, 2.0, 1.0, 60)  # onset > offset
    result = validate_notes([note], fix=True)
    assert result[0].offset >= result[0].onset


def test_validate_notes_removes_none_onset():
    note = Note(False, 0, None, 1.0, 60)  # type: ignore[arg-type]
    result = validate_notes([note], fix=True)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# trim_overlapping_notes
# ---------------------------------------------------------------------------


def test_trim_overlapping_notes_no_overlap():
    notes = [
        Note(False, 0, 0.0, 1.0, 60),
        Note(False, 0, 1.5, 2.5, 60),
    ]
    result = trim_overlapping_notes(notes)
    assert result[0].offset <= result[1].onset


def test_trim_overlapping_notes_overlap():
    notes = [
        Note(False, 0, 0.0, 2.0, 60),
        Note(False, 0, 1.0, 3.0, 60),
    ]
    result = trim_overlapping_notes(notes)
    assert result[0].offset == result[1].onset


def test_trim_overlapping_notes_different_pitch():
    notes = [
        Note(False, 0, 0.0, 2.0, 60),
        Note(False, 0, 0.5, 2.5, 62),  # different pitch, no trimming
    ]
    result = trim_overlapping_notes(notes)
    assert len(result) == 2
    assert result[0].offset == 2.0
    assert result[1].offset == 2.5


# ---------------------------------------------------------------------------
# EventTokenizer
# ---------------------------------------------------------------------------


def _make_event_tokenizer():
    return EventTokenizer(special_tokens=["PAD", "EOS", "UNK"], max_shift_steps=206)


def test_event_tokenizer_num_tokens():
    et = _make_event_tokenizer()
    # 3 special + 206 shift + 128 pitch + 2 velocity + 1 tie + 130 program + 128 drum
    assert et.num_tokens == 3 + 206 + 128 + 2 + 1 + 130 + 128


def test_event_tokenizer_roundtrip_shift():
    et = _make_event_tokenizer()
    for value in [0, 50, 100, 205]:
        event = Event("shift", value)
        idx = et.encode_event(event)
        decoded = et.decode_event_index(idx)
        assert decoded.type == "shift"
        assert decoded.value == value


def test_event_tokenizer_roundtrip_pitch():
    et = _make_event_tokenizer()
    for pitch in [0, 60, 127]:
        event = Event("pitch", pitch)
        idx = et.encode_event(event)
        decoded = et.decode_event_index(idx)
        assert decoded == event


def test_event_tokenizer_roundtrip_velocity():
    et = _make_event_tokenizer()
    for vel in [0, 1]:
        event = Event("velocity", vel)
        idx = et.encode_event(event)
        assert et.decode_event_index(idx) == event


def test_event_tokenizer_roundtrip_program():
    et = _make_event_tokenizer()
    event = Event("program", 0)
    idx = et.encode_event(event)
    assert et.decode_event_index(idx) == event


def test_event_tokenizer_roundtrip_drum():
    et = _make_event_tokenizer()
    event = Event("drum", 36)
    idx = et.encode_event(event)
    assert et.decode_event_index(idx) == event


def test_event_tokenizer_unknown_type_raises():
    et = _make_event_tokenizer()
    with pytest.raises((ValueError, KeyError)):
        et.encode_event(Event("bad_type", 0))


def test_event_tokenizer_out_of_range_index_raises():
    et = _make_event_tokenizer()
    with pytest.raises(ValueError):
        et.decode_event_index(et.num_tokens + 1)


# ---------------------------------------------------------------------------
# note_event2event / event2note_event roundtrip
# ---------------------------------------------------------------------------


def _simple_piano_note_events():
    """Onset at 0.10s, offset at 0.20s for middle C on piano."""
    return [
        NoteEvent(is_drum=False, program=0, time=0.10, velocity=1, pitch=60),
        NoteEvent(is_drum=False, program=0, time=0.20, velocity=0, pitch=60),
    ]


def test_note_event2event_produces_tie():
    evs = note_event2event(_simple_piano_note_events())
    types = [e.type for e in evs]
    assert "tie" in types


def test_note_event2event_shift_correct():
    evs = note_event2event(_simple_piano_note_events(), frame_rate=100)
    shift_values = [e.value for e in evs if e.type == "shift"]
    # onset at frame 10 from start 0 → shift 10; offset at frame 20 → shift 20
    assert 10 in shift_values
    assert 20 in shift_values


def test_event_roundtrip_piano_note():
    note_events = _simple_piano_note_events()
    events = note_event2event(note_events, frame_rate=100)
    recovered_ne, recovered_tie, _, err = event2note_event(
        events, start_time=0.0, frame_rate=100
    )
    assert not err
    assert len(recovered_tie) == 0
    times = {(ne.velocity, ne.pitch): ne.time for ne in recovered_ne}
    assert abs(times[(1, 60)] - 0.10) < 1e-6
    assert abs(times[(0, 60)] - 0.20) < 1e-6


def test_event_roundtrip_drum_note():
    note_events = [
        NoteEvent(is_drum=True, program=DRUM_PROGRAM, time=0.05, velocity=1, pitch=36)
    ]
    events = note_event2event(note_events, frame_rate=100)
    recovered_ne, _, _, err = event2note_event(events, start_time=0.0, frame_rate=100)
    assert not err
    assert len(recovered_ne) == 1
    assert recovered_ne[0].is_drum
    assert recovered_ne[0].pitch == 36


# ---------------------------------------------------------------------------
# NoteEventTokenizer
# ---------------------------------------------------------------------------


def _make_tokenizer():
    return NoteEventTokenizer(max_length=1024, frame_rate=100, max_shift_steps=206)


def test_note_event_tokenizer_num_tokens():
    tok = _make_tokenizer()
    et = EventTokenizer(special_tokens=["PAD", "EOS", "UNK"], max_shift_steps=206)
    assert tok.num_tokens == et.num_tokens


def test_note_event_tokenizer_encode_returns_ints():
    tok = _make_tokenizer()
    tokens = tok.encode(_simple_piano_note_events())
    assert all(isinstance(t, int) for t in tokens)
    assert all(0 <= t < tok.num_tokens for t in tokens)


def test_note_event_tokenizer_encode_decode_roundtrip():
    tok = _make_tokenizer()
    note_events = _simple_piano_note_events()
    tokens = tok.encode(note_events)
    recovered_ne, _, _, err = tok.decode(tokens)
    assert not err
    by_vel = {ne.velocity: ne for ne in recovered_ne}
    assert abs(by_vel[1].time - 0.10) < 1e-6
    assert abs(by_vel[0].time - 0.20) < 1e-6


def test_note_event_tokenizer_encode_decode_multi_note():
    tok = _make_tokenizer()
    note_events = [
        NoteEvent(False, 0, 0.10, 1, 60),
        NoteEvent(False, 0, 0.10, 1, 64),
        NoteEvent(False, 0, 0.20, 0, 60),
        NoteEvent(False, 0, 0.20, 0, 64),
    ]
    tokens = tok.encode(note_events)
    recovered_ne, _, _, err = tok.decode(tokens)
    assert not err
    onsets = [ne for ne in recovered_ne if ne.velocity == 1]
    assert len(onsets) == 2


# ---------------------------------------------------------------------------
# note_event2note / note2note_event
# ---------------------------------------------------------------------------


def test_note_event2note_basic():
    note_events = _simple_piano_note_events()
    notes, err = note_event2note(note_events)
    assert not err
    assert len(notes) == 1
    n = notes[0]
    assert n.pitch == 60
    assert n.program == 0
    assert abs(n.onset - 0.10) < 1e-6
    assert abs(n.offset - 0.20) < 1e-6


def test_note2note_event_roundtrip():
    original = [Note(False, 0, 0.10, 0.50, 60)]
    note_events = note2note_event(original)
    assert len(note_events) == 2  # onset + offset
    by_vel = {ne.velocity: ne for ne in note_events}
    assert abs(by_vel[1].time - 0.10) < 1e-6
    assert abs(by_vel[0].time - 0.50) < 1e-6


def test_note_event2note_drum():
    note_events = [
        NoteEvent(is_drum=True, program=DRUM_PROGRAM, time=0.05, velocity=1, pitch=36)
    ]
    notes, err = note_event2note(note_events)
    assert not err
    assert len(notes) == 1
    assert notes[0].is_drum
    assert notes[0].pitch == 36
