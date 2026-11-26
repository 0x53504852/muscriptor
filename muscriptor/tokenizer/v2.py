"""Window tokens -> notes as the MuScriptor V2 training code decodes them, plus the V2 vocabulary facts.

Ported from the training repository (fast_muscriptor, branch gabriel/v2-fast-vote: amt/tokenizer.py and amt/notes.py), so
that the inference package turns the same window tokens into the same notes:

- `decode_track_overlap`: 5 s windows starting every `hop` s; each window owns the events of its central region (from the
  midpoint between its centre and the previous window's centre to the midpoint with the next one).
- `decode_track`: back-to-back windows.
- `COMPACT_IDS`: the 830 tokenizer ids a compact-vocabulary model predicts (its output class c is token COMPACT_IDS[c]).
- `PREFIX_GROUPS`: the instrument groups of the `instrument_group` condition (group id = index).

The token vocabulary (1393 ids) is the one of `muscriptor.tokenizer.mt3.MT3Tokenizer` (PAD, EOS, UNK, shift 0..1000,
pitch 0..127, velocity 0..1, tie, program 0..129, drum 0..127, in this order).
"""

from collections import Counter

from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    MINIMUM_NOTE_DURATION_SEC,
    Event,
    Note,
    NoteEvent,
    TieNoteEvent,
)

PAD, EOS = 0, 1
SPECIAL_TOKENS = ("PAD", "EOS", "UNK")
MAX_SHIFT_STEPS = 1001
FRAME_RATE = 100
SEGMENT_SEC = 5.0

# MT3_FULL_PLUS instrument groups (YourMT3+), as muscriptor.tokenizer.mt3.get_group_program_map("MT3_FULL_PLUS").
MT3_FULL_PLUS: list[list[int]] = [
    [0, 1, 3, 6, 7],
    [2, 4, 5],
    list(range(8, 16)),
    list(range(16, 24)),
    [24, 25],
    [26, 27, 28],
    [29, 30, 31],
    [32, 35],
    [33, 34, 36, 37, 38, 39],
    [40],
    [41],
    [42],
    [43],
    [46],
    [47],
    [48, 49, 44, 45],
    [50, 51],
    [52, 53, 54],
    [55],
    [56, 59],
    [57],
    [58],
    [60],
    [61, 62, 63],
    [64, 65],
    [66],
    [67],
    [68],
    [69],
    [70],
    [71],
    list(range(72, 80)),
    list(range(80, 88)),
    list(range(88, 96)),
    [100],
    [101],
]
# Groups of the instrument_group condition: MT3_FULL_PLUS, then singletons for drums, program 129 and the unassigned
# programs. The order of the singletons is the one the models were trained with (their embeddings depend on it).
PREFIX_GROUPS: list[list[int]] = MT3_FULL_PLUS + [
    [p] for p in [DRUM_PROGRAM, 129, 96, 97, 98, 99, *range(102, 128)]
]
PROGRAM_TO_PREFIX_GROUP = {
    p: g for g, programs in enumerate(PREFIX_GROUPS) for p in programs
}
DRUM_GROUP = PROGRAM_TO_PREFIX_GROUP[DRUM_PROGRAM]  # 36
# Program token remap: every program of a group is emitted as the group's first program.
PROGRAM_TOKEN = {p: programs[0] for programs in MT3_FULL_PLUS for p in programs}


def _build_vocab() -> list[Event]:
    ranges = [(t, 0, 0) for t in SPECIAL_TOKENS] + [
        ("shift", 0, MAX_SHIFT_STEPS - 1),
        ("pitch", 0, 127),
        ("velocity", 0, 1),
        ("tie", 0, 0),
        ("program", 0, 129),
        ("drum", 0, 127),
    ]
    return [Event(t, v) for t, lo, hi in ranges for v in range(lo, hi + 1)]


VOCAB = _build_vocab()
VOCAB_SIZE = len(VOCAB)  # 1393
TOKEN_ID = {(e.type, e.value): i for i, e in enumerate(VOCAB)}
# The 1393 ids minus UNK, shifts past 5 s (500 ticks) and the programs that never start a group.
COMPACT_IDS = [TOKEN_ID[("PAD", 0)], TOKEN_ID[("EOS", 0)]] + [
    i
    for i, e in enumerate(VOCAB)
    if (e.type == "shift" and e.value <= 5 * FRAME_RATE)
    or e.type in ("pitch", "velocity", "tie", "drum")
    or (e.type == "program" and PROGRAM_TOKEN.get(e.value, e.value) == e.value)
]
assert len(COMPACT_IDS) == 830


# ---------------------------------------------------------------------------
# Notes / note events (amt/notes.py)
# ---------------------------------------------------------------------------


def sort_notes(notes: list[Note]):
    notes.sort(
        key=lambda n: (n.onset, n.is_drum, n.program, n.velocity, n.pitch, n.offset)
    )


def sort_note_events(note_events: list[NoteEvent]):
    note_events.sort(key=lambda e: (e.time, e.is_drum, e.program, e.velocity, e.pitch))


def sort_tie_note_events(tie_note_events: list[TieNoteEvent]):
    tie_note_events.sort(key=lambda e: (e.program, e.pitch))


def event2note_event(
    events: list[Event], start_time: float = 0.0, frame_rate: int = FRAME_RATE
) -> tuple[list[NoteEvent], list[TieNoteEvent], list[tuple[int, int]], Counter]:
    """Events of one window -> (note_events, ties, last_activity, errors)."""
    tie_index = None
    program_state = None
    ties = []
    last_activity: set[tuple[int, int]] = set()
    err_cnt: Counter = Counter()
    for i, e in enumerate(events):
        if e.type == "tie":
            tie_index = i
            break
        if e.type == "shift":
            break
        if e.type == "program":
            program_state = e.value
        elif e.type == "pitch":
            if program_state is None:
                err_cnt["Err/Missing prg in tie"] += 1
                continue
            ties.append(TieNoteEvent(program_state, e.value))
            last_activity.add((program_state, e.value))
    if tie_index is None:
        err_cnt["Err/Missing tie"] += 1
        return [], [], [], err_cnt
    events = events[tie_index + 1 :]

    note_events = []
    velocity_state = None
    start_tick = round(start_time * frame_rate)
    tick_state = start_tick
    for e in events:
        if e.type == "shift":
            if e.value <= 0:
                err_cnt["Err/Negative shift"] += 1
                continue
            prev = tick_state
            tick_state = start_tick + e.value
            if tick_state <= prev:
                err_cnt["Err/Shift not strictly monotonic"] += 1
        elif e.type == "drum":
            note_events.append(
                NoteEvent(True, DRUM_PROGRAM, tick_state / frame_rate, 1, e.value)
            )
        elif e.type == "program":
            program_state = e.value
        elif e.type == "velocity":
            velocity_state = e.value
        elif e.type == "pitch":
            if program_state is None:
                err_cnt["Err/Missing prg"] += 1
                continue
            if velocity_state is None:
                err_cnt["Err/Missing vel"] += 1
                continue
            if velocity_state > 0:
                last_activity.add((program_state, e.value))
            elif (program_state, e.value) in last_activity:
                last_activity.remove((program_state, e.value))
            else:
                err_cnt["Err/Note off without note on"] += 1
                continue
            note_events.append(
                NoteEvent(
                    False,
                    program_state,
                    tick_state / frame_rate,
                    velocity_state,
                    e.value,
                )
            )
        elif e.type == "EOS":
            break
        elif e.type in ("PAD", "UNK"):
            continue
        elif e.type == "tie":
            err_cnt["Err/Multi-tie"] += 1
    sort_note_events(note_events)
    sort_tie_note_events(ties)
    return note_events, ties, list(last_activity), err_cnt


def validate_notes(
    notes: list[Note], minimum_offset: float = MINIMUM_NOTE_DURATION_SEC
) -> list[Note]:
    for note in list(notes):
        if note.onset is None:
            notes.remove(note)
        elif note.offset is None:
            note.offset = note.onset + minimum_offset
        elif note.onset > note.offset:
            note.offset = max(note.offset, note.onset + minimum_offset)
        elif not note.is_drum and note.offset - note.onset < 0.01:
            note.offset = note.onset + minimum_offset
    return notes


def trim_overlapping_notes(notes: list[Note]) -> list[Note]:
    """Within each (program, pitch, is_drum) channel, cut a note at the next onset; drop empty notes."""
    if len(notes) <= 1:
        return notes
    channels: dict[tuple[int, int, bool], list[Note]] = {}
    for n in notes:
        channels.setdefault((n.program, n.pitch, n.is_drum), []).append(n)
    trimmed = []
    for channel_notes in channels.values():
        channel_notes.sort(key=lambda n: n.onset)
        for prev, cur in zip(channel_notes, channel_notes[1:]):
            if prev.offset > cur.onset:
                prev.offset = cur.onset
        trimmed.extend(n for n in channel_notes if n.onset < n.offset)
    sort_notes(trimmed)
    return trimmed


def note_event2note(
    note_events: list[NoteEvent],
    tie_note_events: list[TieNoteEvent] | None = None,
    force_offset_past_segment_end: float | None = None,
    force_onset_before_segment_start: float | None = None,
    shorten_notes_above_n_sec: float = 10,
) -> tuple[list[Note], Counter]:
    notes = []
    active: dict[tuple[int, int], NoteEvent | TieNoteEvent] = {}
    err_cnt: Counter = Counter()
    for tne in tie_note_events or []:
        active[(tne.program, tne.pitch)] = tne
    sort_note_events(note_events)
    for ne in note_events:
        if ne.is_drum:
            if ne.velocity == 1:
                notes.append(
                    Note(
                        True,
                        DRUM_PROGRAM,
                        ne.time,
                        ne.time + MINIMUM_NOTE_DURATION_SEC,
                        ne.pitch,
                    )
                )
            continue
        prev = active.pop((ne.program, ne.pitch), None)
        if ne.velocity == 0 and prev is None:
            err_cnt["Err/onset not found"] += 1
            continue
        if isinstance(prev, NoteEvent):
            notes.append(
                Note(False, prev.program, prev.time, ne.time, prev.pitch, prev.velocity)
            )
        elif isinstance(prev, TieNoteEvent):
            notes.append(
                Note(
                    False,
                    prev.program,
                    force_onset_before_segment_start,
                    ne.time,
                    prev.pitch,
                )
            )
        if ne.velocity == 1:
            active[(ne.program, ne.pitch)] = ne
    # Notes still on at the end; tie notes never closed in the segment are dropped.
    for ne in active.values():
        if isinstance(ne, NoteEvent) and ne.velocity == 1:
            offset = (
                ne.time + MINIMUM_NOTE_DURATION_SEC
                if force_offset_past_segment_end is None
                else force_offset_past_segment_end
            )
            notes.append(Note(False, ne.program, ne.time, offset, ne.pitch))
    for n in notes:
        if n.offset - n.onset > shorten_notes_above_n_sec:
            n.offset = n.onset + MINIMUM_NOTE_DURATION_SEC
            err_cnt[f"Err/long note > {shorten_notes_above_n_sec}s"] += 1
    notes = trim_overlapping_notes(validate_notes(notes))
    sort_notes(notes)
    return notes, err_cnt


def merge_segments_to_notes(
    segments: list[
        tuple[list[NoteEvent], list[TieNoteEvent], list[tuple[int, int]], float]
    ],
) -> tuple[list[Note], Counter]:
    """Stitch consecutive, non-overlapping decoded segments into one note list. A note active at the end of a segment
    but missing from the next segment's tie section is closed at the next segment's start."""
    merged: list[NoteEvent] = []
    err_cnt: Counter = Counter()
    prev_last_activity = None
    for note_events, ties, last_activity, start_time in segments:
        if prev_last_activity is not None:
            tied = {(t.program, t.pitch) for t in ties}
            for program, pitch in prev_last_activity:
                if (program, pitch) not in tied:
                    merged.append(NoteEvent(False, program, start_time, 0, pitch))
                    err_cnt["Err/merging segment tie"] += 1
        merged += note_events
        prev_last_activity = last_activity
    notes, errs = note_event2note(merged)
    err_cnt.update(errs)
    return notes, err_cnt


# ---------------------------------------------------------------------------
# Track decoding (amt/tokenizer.py)
# ---------------------------------------------------------------------------


def _decode_events(tokens: list[int], seek_time: float):
    events = [VOCAB[t] for t in tokens if 0 <= t < VOCAB_SIZE]
    return event2note_event(events, start_time=seek_time, frame_rate=FRAME_RATE)


def decode_track(
    segment_tokens: list[list[int]], seek_times: list[float]
) -> tuple[list[Note], Counter]:
    """Notes of a full track from its consecutive, non-overlapping segments."""
    segments, err_cnt = [], Counter()
    for tokens, seek_time in zip(segment_tokens, seek_times, strict=True):
        note_events, ties, last_activity, errs = _decode_events(tokens, seek_time)
        segments.append((note_events, ties, last_activity, seek_time))
        err_cnt += errs
    notes, errs = merge_segments_to_notes(segments)
    return notes, err_cnt + errs


def decode_track_overlap(
    segment_tokens: list[list[int]],
    seek_times: list[float],
    duration: float = SEGMENT_SEC,
) -> tuple[list[Note], Counter]:
    """Notes of a full track from overlapping segments of `duration` s starting at `seek_times` (increasing).

    Each segment owns the events (onsets and offsets) of its central region: from the midpoint between its centre and the
    previous segment's centre to the midpoint with the next one (the first segment from the start, the last one to the
    end). The regions tile the track and decode as consecutive segments: each region gets as ties the notes its own
    segment has sounding at the region start, and merge_segments_to_notes continues a note held across a region boundary
    when both segments have it sounding there, and closes it at the boundary when the next segment does not."""
    segments, err_cnt = [], Counter()
    ticks = [round(s * FRAME_RATE) for s in seek_times]
    half = round(duration * FRAME_RATE) / 2
    bounds = (
        [None] + [(a + b) / 2 + half for a, b in zip(ticks, ticks[1:])] + [None]
    )  # region bounds, ticks
    for k, (tokens, seek_time) in enumerate(
        zip(segment_tokens, seek_times, strict=True)
    ):
        note_events, ties, _, errs = _decode_events(tokens, seek_time)
        err_cnt += errs
        lo, hi = bounds[k], bounds[k + 1]
        active = {(t.program, t.pitch) for t in ties}
        start_active, region = None, []
        for ne in note_events:  # sorted by time, offsets before onsets
            tick = round(ne.time * FRAME_RATE)
            if hi is not None and tick >= hi:
                break
            if start_active is None and (lo is None or tick >= lo):
                start_active = set(active)
            if start_active is not None:
                region.append(ne)
            if not ne.is_drum:
                if ne.velocity > 0:
                    active.add((ne.program, ne.pitch))
                else:
                    active.discard((ne.program, ne.pitch))
        if start_active is None:
            start_active = set(active)
        start_ties = [TieNoteEvent(p, q) for p, q in start_active]
        sort_tie_note_events(start_ties)
        start = seek_time if lo is None else lo / FRAME_RATE
        segments.append((region, start_ties, list(active), start))
    notes, errs = merge_segments_to_notes(segments)
    return notes, err_cnt + errs
