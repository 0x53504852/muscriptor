"""Note and NoteEvent types, tokenizer utilities.

Adapted from YourMT3+ (https://github.com/mimbres/YourMT3).
"""

import os
import warnings
from collections import Counter
from dataclasses import dataclass

from mido import Message, MidiFile, MidiTrack, second2tick

DRUM_PROGRAM = 128
MINIMUM_NOTE_DURATION_SEC = 0.01


@dataclass
class Note:
    is_drum: bool
    program: int  # MIDI program number (0-127); 128 for drum
    onset: float  # onset time in seconds
    offset: float  # offset time in seconds (== onset for drums)
    pitch: int  # MIDI note number (0-127)


@dataclass
class NoteEvent:
    is_drum: bool
    program: int  # [0, 127], 128 for drum (ignored in tokenizer)
    time: float  # absolute time in seconds
    velocity: int  # 1 for onset, 0 for offset; drum has no offset
    pitch: int  # MIDI pitch


@dataclass
class TieNoteEvent:
    program: int  # [0, 127], 128 for drum (ignored in tokenizer)
    pitch: int  # MIDI pitch


@dataclass
class EventRange:
    type: str
    min_value: int
    max_value: int  # inclusive


@dataclass
class Event:
    type: str
    value: int


def sort_notes(notes: list[Note]):
    if len(notes) > 0:
        notes.sort(key=lambda n: (n.onset, n.is_drum, n.program, n.pitch, n.offset))


def sort_note_events(note_events: list[NoteEvent]):
    if len(note_events) > 0:
        note_events.sort(
            key=lambda n: (n.time, n.is_drum, n.program, n.velocity, n.pitch)
        )


def sort_tie_note_events(tie_note_events: list[TieNoteEvent]):
    if len(tie_note_events) > 0:
        tie_note_events.sort(key=lambda n: (n.program, n.pitch))


def validate_notes(
    notes: list[Note],
    minimum_offset: float | None = MINIMUM_NOTE_DURATION_SEC,
    fix: bool = True,
) -> list[Note]:
    if len(notes) > 0:
        for note in list(notes):
            if note.onset is None and fix:
                notes.remove(note)
            elif note.offset is None and fix:
                note.offset = note.onset + minimum_offset
            elif note.onset > note.offset:
                if fix:
                    note.offset = max(note.offset, note.onset + minimum_offset)
            elif note.is_drum is False and note.offset - note.onset < 0.01 and fix:
                note.offset = note.onset + minimum_offset
    return notes


def trim_overlapping_notes(notes: list[Note], sort: bool = True) -> list[Note]:
    if len(notes) <= 1:
        return notes
    trimmed_notes = []
    channels = set((note.program, note.pitch, note.is_drum) for note in notes)
    for program, pitch, is_drum in channels:
        channel_notes = [
            n
            for n in notes
            if n.pitch == pitch and n.program == program and n.is_drum == is_drum
        ]
        sorted_notes = sorted(channel_notes, key=lambda n: n.onset)
        for i in range(1, len(sorted_notes)):
            if sorted_notes[i - 1].offset > sorted_notes[i].onset:
                sorted_notes[i - 1].offset = sorted_notes[i].onset
        valid_notes = [n for n in sorted_notes if n.onset < n.offset]
        trimmed_notes.extend(valid_notes)
    if sort:
        sort_notes(trimmed_notes)
    return trimmed_notes


def _create_inverse_vocab(in_vocab):
    out_vocab = {}
    for key, values in in_vocab.items():
        for value in values:
            out_vocab[value] = values[0]
    return out_vocab


class EventTokenizer:
    """Fast encoding and decoding of Event objects."""

    def __init__(
        self,
        special_tokens: list[str],
        max_shift_steps: int,
        program_vocabulary: dict | None = None,
        drum_vocabulary: dict | None = None,
    ):
        self.special_tokens = special_tokens
        _special_token_ranges = [EventRange(token, 0, 0) for token in special_tokens]
        _shift_range = EventRange(
            type="shift", min_value=0, max_value=max_shift_steps - 1
        )
        other_event_ranges = [
            EventRange("pitch", min_value=0, max_value=127),
            EventRange("velocity", min_value=0, max_value=1),
            EventRange("tie", min_value=0, max_value=0),
            EventRange("program", min_value=0, max_value=129),
            EventRange("drum", min_value=0, max_value=127),
        ]
        self._event_ranges = _special_token_ranges + [_shift_range] + other_event_ranges
        assert len(self._event_ranges) == len(
            set([er.type for er in self._event_ranges])
        )

        self._decode_dict = {}
        self._encode_dict = {}
        self._event_type_range_dict = {}
        idx = 0
        for er in self._event_ranges:
            start_idx = idx
            for value in range(er.min_value, er.max_value + 1):
                self._decode_dict[idx] = Event(type=er.type, value=value)
                self._encode_dict[(er.type, value)] = idx
                idx += 1
            end_idx = idx - 1
            self._event_type_range_dict[er.type] = (start_idx, end_idx)
        self.num_tokens: int = idx

        self.inverse_vocab_program = None
        if program_vocabulary is not None:
            self.inverse_vocab_program = _create_inverse_vocab(program_vocabulary)

        self.inverse_vocab_drum = None
        if drum_vocabulary is not None:
            self.inverse_vocab_drum = _create_inverse_vocab(drum_vocabulary)

    def encode_event(self, event: Event) -> int:
        if (event.type, event.value) not in self._encode_dict:
            raise ValueError(
                f"Unknown event type: {event.type} or value: {event.value}"
            )
        if event.type == "program" and self.inverse_vocab_program is not None:
            _event_value = self.inverse_vocab_program.get(event.value, event.value)
            return self._encode_dict[(event.type, _event_value)]
        elif event.type == "drum" and self.inverse_vocab_drum is not None:
            _event_value = self.inverse_vocab_drum.get(event.value, event.value)
            return self._encode_dict[(event.type, _event_value)]
        else:
            return self._encode_dict[(event.type, event.value)]

    def decode_event_index(self, index: int) -> Event:
        if index < 0 or index >= self.num_tokens:
            raise ValueError(f"Unknown event index: {index}")
        decoded_event = self._decode_dict[index]
        return Event(type=decoded_event.type, value=decoded_event.value)


def note_event2event(
    note_events: list[NoteEvent],
    tie_note_events: list[TieNoteEvent] | None = None,
    start_time: float = 0.0,
    frame_rate: int = 100,
) -> list[Event]:
    if tie_note_events is not None:
        sort_tie_note_events(tie_note_events)
    note_events.sort(
        key=lambda n: (
            round(n.time * frame_rate),
            n.is_drum,
            n.program,
            n.velocity,
            n.pitch,
        )
    )

    events = []
    start_tick = round(start_time * frame_rate)
    tick_state = start_tick
    program_state = None

    if tie_note_events:
        for tne in tie_note_events:
            if tne.program != program_state:
                events.append(Event(type="program", value=tne.program))
                program_state = tne.program
            events.append(Event(type="pitch", value=tne.pitch))

    events.append(Event(type="tie", value=0))

    velocity_state = None
    for ne in note_events:
        if ne.is_drum and ne.velocity == 0:
            continue

        ne_tick = round(ne.time * frame_rate)
        if ne_tick > tick_state:
            shift_ticks = ne_tick - start_tick
            events.append(Event(type="shift", value=shift_ticks))
            tick_state = ne_tick
        elif ne_tick == tick_state:
            pass
        else:
            raise ValueError(
                f"NoteEvent tick {ne_tick} at time {ne.time} is smaller than tick_state {tick_state}."
            )

        if ne.is_drum and ne.velocity == 1:
            if velocity_state != 1 or velocity_state is None:
                events.append(Event(type="velocity", value=1))
                velocity_state = 1
            events.append(Event(type="drum", value=ne.pitch))
        else:
            if ne.program != program_state or program_state is None:
                events.append(Event(type="program", value=ne.program))
                program_state = ne.program
            if ne.velocity != velocity_state or velocity_state is None:
                events.append(Event(type="velocity", value=ne.velocity))
                velocity_state = ne.velocity
            events.append(Event(type="pitch", value=ne.pitch))

    return events


def event2note_event(
    events: list[Event], start_time: float = 0.0, frame_rate: int = 100
) -> tuple[list[NoteEvent], list[TieNoteEvent], list[tuple[int, ...]], Counter]:
    assert start_time >= 0.0

    tie_index = None
    program_state = None
    tie_note_events = []
    last_activity = set()
    err_cnt: Counter = Counter()

    for i, e in enumerate(events):
        try:
            if e.type == "tie":
                tie_index = i
                break
            if e.type == "shift":
                break
            elif e.type == "program":
                program_state = e.value
            elif e.type == "pitch":
                if program_state is None:
                    raise ValueError("Err/Missing prg in tie")
                tie_note_events.append(
                    TieNoteEvent(program=program_state, pitch=e.value)
                )
                last_activity.add((program_state, e.value))
        except ValueError as ve:
            err_cnt[str(ve)] += 1

    try:
        if tie_index is None:
            raise ValueError("Err/Missing tie")
        else:
            events = events[tie_index + 1 :]
    except ValueError as ve:
        err_cnt[str(ve)] += 1
        return [], [], [], err_cnt

    note_events = []
    velocity_state = None
    start_tick = round(start_time * frame_rate)
    tick_state = start_tick
    for e in events:
        try:
            if e.type == "shift":
                if e.value <= 0:
                    raise ValueError("Err/Negative shift")
                prev_tick_state = tick_state
                tick_state = start_tick + e.value
                if tick_state <= prev_tick_state:
                    raise ValueError("Err/Shift not strictly monotonic")
            elif e.type == "drum":
                note_events.append(
                    NoteEvent(
                        is_drum=True,
                        program=DRUM_PROGRAM,
                        time=tick_state / frame_rate,
                        velocity=1,
                        pitch=e.value,
                    )
                )
            elif e.type == "program":
                program_state = e.value
            elif e.type == "velocity":
                velocity_state = e.value
            elif e.type == "pitch":
                if program_state is None:
                    raise ValueError("Err/Missing prg")
                elif velocity_state is None:
                    raise ValueError("Err/Missing vel")
                if velocity_state > 0:
                    last_activity.add((program_state, e.value))
                elif velocity_state == 0 and (program_state, e.value) in last_activity:
                    last_activity.remove((program_state, e.value))
                else:
                    raise ValueError("Err/Note off without note on")
                note_events.append(
                    NoteEvent(
                        is_drum=False,
                        program=program_state,
                        time=tick_state / frame_rate,
                        velocity=velocity_state,
                        pitch=e.value,
                    )
                )
            elif e.type == "EOS":
                break
            elif e.type in ("PAD", "UNK"):
                continue
            elif e.type == "tie":
                if tick_state == start_tick:
                    raise ValueError("Err/Multi-tie type 1")
                else:
                    raise ValueError("Err/Multi-tie type 2")
            else:
                raise ValueError("Err/Unknown event")
        except ValueError as ve:
            err_cnt[str(ve)] += 1

    sort_note_events(note_events)
    sort_tie_note_events(tie_note_events)
    return note_events, tie_note_events, list(last_activity), err_cnt


class NoteEventTokenizer:
    """Encodes and decodes note events to/from token sequences."""

    def __init__(
        self,
        max_length: int = 1024,
        frame_rate: int = 100,
        extra_tokens: list[str] = [],
        max_shift_steps: int = 206,
        program_vocabulary: dict | None = None,
        drum_vocabulary: dict | None = None,
        modulo_max_shift_steps: bool = False,
    ) -> None:
        special_tokens: list[str] = ["PAD", "EOS", "UNK"]
        self.event_tokenizer = EventTokenizer(
            special_tokens=special_tokens + extra_tokens,
            max_shift_steps=max_shift_steps,
            program_vocabulary=program_vocabulary,
            drum_vocabulary=drum_vocabulary,
        )
        self.max_length = max_length
        self.frame_rate = frame_rate
        self.modulo_max_shift_steps = modulo_max_shift_steps
        self.max_shift_steps = max_shift_steps
        self.eos_id = self.event_tokenizer.special_tokens.index("EOS")
        self.pad_id = self.event_tokenizer.special_tokens.index("PAD")

    @property
    def num_tokens(self) -> int:
        return self.event_tokenizer.num_tokens

    def encode(
        self,
        note_events: list[NoteEvent],
        tie_note_events: list[TieNoteEvent] | None = None,
        start_time: float = 0.0,
    ) -> list[int]:
        events = note_event2event(
            note_events=note_events,
            tie_note_events=tie_note_events,
            start_time=start_time,
            frame_rate=self.frame_rate,
        )
        if self.modulo_max_shift_steps:
            for e in events:
                if e.type == "shift":
                    e.value = e.value % self.max_shift_steps
        return [self.event_tokenizer.encode_event(e) for e in events]

    def decode(
        self, tokens: list[int], start_time: float = 0.0
    ) -> tuple[list[NoteEvent], list[TieNoteEvent], list[tuple[int, ...]], Counter]:
        events = [self.event_tokenizer.decode_event_index(idx) for idx in tokens]
        return event2note_event(events, start_time, self.frame_rate)

    def decode_segments(
        self, list_tokens: list[list[int]], list_start_times: list[float]
    ):
        assert len(list_tokens) == len(list_start_times)
        zipped = []
        total_err_cnt: Counter = Counter()
        for tokens, start_time in zip(list_tokens, list_start_times):
            note_events, tie_note_events, last_activity, err_cnt = self.decode(
                tokens, start_time
            )
            zipped.append((note_events, tie_note_events, last_activity, start_time))
            total_err_cnt += err_cnt
        return zipped, total_err_cnt


def slice_note_events_and_ties(
    note_events: list[NoteEvent], start_time: float, end_time: float
) -> tuple[list[NoteEvent], list[TieNoteEvent]]:
    if start_time > end_time:
        raise ValueError(
            f"slice_note_events: start_time {start_time} > end_time {end_time}"
        )
    elif len(note_events) == 0:
        warnings.warn("slice_note_events: empty note_events as input")
        return [], []

    start_index, end_index = None, None
    tie_activity = set()
    for i, ne in enumerate(note_events):
        if start_index is None and start_time <= ne.time < end_time:
            start_index = i
        if start_index is None and not ne.is_drum:
            if ne.velocity > 0:
                tie_activity.add((ne.program, ne.pitch))
            elif ne.velocity == 0 and (ne.program, ne.pitch) in tie_activity:
                tie_activity.remove((ne.program, ne.pitch))
        if ne.time >= end_time:
            end_index = i
            break

    if start_index is None and (end_index is None or end_index == 0):
        tie_note_events = []
    else:
        tie_note_events = [TieNoteEvent(program=p, pitch=pi) for p, pi in tie_activity]
    sort_tie_note_events(tie_note_events)

    sliced_note_events = (
        [] if start_index is None else note_events[start_index:end_index]
    )
    sort_note_events(sliced_note_events)
    return sliced_note_events, tie_note_events


def note_event2note(
    note_events: list[NoteEvent],
    tie_note_events: list[TieNoteEvent] | None = None,
    shorten_notes_above_n_sec: int = 10,
    fix_broken_notes: bool = True,
    trim_overlap: bool = True,
    force_offset_past_segment_end: float | None = None,
    force_onset_before_segment_start: float | None = None,
) -> tuple[list[Note], Counter]:
    notes: list[Note] = []
    active_note_events: dict[tuple[int, int], NoteEvent | TieNoteEvent] = {}
    err_cnt: Counter = Counter()

    if tie_note_events is not None:
        for ne in tie_note_events:
            active_note_events[(ne.program, ne.pitch)] = ne

    sort_note_events(note_events)
    for ne in note_events:
        try:
            if ne.time is None:
                continue
            elif ne.is_drum:
                if ne.velocity == 1:
                    notes.append(
                        Note(
                            is_drum=True,
                            program=DRUM_PROGRAM,
                            onset=ne.time,
                            offset=ne.time + MINIMUM_NOTE_DURATION_SEC,
                            pitch=ne.pitch,
                        )
                    )
                else:
                    continue
            else:
                active_ne = active_note_events.pop((ne.program, ne.pitch), None)
                if ne.velocity == 0 and active_ne is None:
                    raise ValueError("Err/onset not found")
                if active_ne is not None:
                    if type(active_ne) is NoteEvent:
                        notes.append(
                            Note(
                                is_drum=False,
                                program=active_ne.program,
                                onset=active_ne.time,
                                offset=ne.time,
                                pitch=active_ne.pitch,
                            )
                        )
                    else:  # TieNoteEvent
                        notes.append(
                            Note(
                                is_drum=False,
                                program=active_ne.program,
                                onset=force_onset_before_segment_start,
                                offset=ne.time,
                                pitch=active_ne.pitch,
                            )
                        )
                if ne.velocity == 1:
                    active_note_events[(ne.program, ne.pitch)] = ne
        except ValueError as ve:
            err_cnt[str(ve)] += 1

    for ne in active_note_events.values():
        try:
            if type(ne) is NoteEvent and ne.velocity == 1:
                if ne.program is None or ne.pitch is None:
                    raise ValueError("Err/active ne incomplete")
                elif ne.time is None:
                    continue
                else:
                    notes.append(
                        Note(
                            is_drum=False,
                            program=ne.program,
                            onset=ne.time,
                            offset=ne.time + MINIMUM_NOTE_DURATION_SEC
                            if force_offset_past_segment_end is None
                            else force_offset_past_segment_end,
                            pitch=ne.pitch,
                        )
                    )
        except ValueError as ve:
            err_cnt[str(ve)] += 1

    if shorten_notes_above_n_sec > 0:
        for n in list(notes):
            try:
                if n.offset - n.onset > shorten_notes_above_n_sec:
                    n.offset = n.onset + MINIMUM_NOTE_DURATION_SEC
                    raise ValueError(f"Err/long note > {shorten_notes_above_n_sec}s")
            except ValueError as ve:
                err_cnt[str(ve)] += 1
    if fix_broken_notes:
        notes = validate_notes(notes, fix=True)
    if trim_overlap:
        notes = trim_overlapping_notes(notes, sort=True)
    else:
        sort_notes(notes)
    return notes, err_cnt


def note2note_event(notes: list[Note]) -> list[NoteEvent]:
    note_events = []
    for note in notes:
        if note.program == 1024:
            note.is_drum = True
        note_events.append(
            NoteEvent(note.is_drum, note.program, note.onset, 1, note.pitch)
        )
        if not note.is_drum:
            note_events.append(
                NoteEvent(note.is_drum, note.program, note.offset, 0, note.pitch)
            )
    sort_note_events(note_events)
    return note_events


def merge_zipped_note_events_and_ties_to_notes(
    zipped_note_events_and_ties,
    segment_duration,
    force_note_off_missing_tie: bool = True,
    fix_broken_notes: bool = True,
) -> tuple[list[Note], Counter]:
    merged_note_events = []
    prev_last_activity = None
    seg_merge_err_cnt: Counter = Counter()

    for i, (nes, tie_nes, last_activity, start_time) in enumerate(
        zipped_note_events_and_ties
    ):
        this_segment_end_time = start_time + segment_duration
        next_segment_start_time = (
            zipped_note_events_and_ties[i + 1][3]
            if i < len(zipped_note_events_and_ties) - 1
            else None
        )
        if (
            next_segment_start_time is not None
            and next_segment_start_time != this_segment_end_time
        ):
            nes = [ne for ne in nes if ne.time < next_segment_start_time]
            last_activity = set([(ne.program, ne.pitch) for ne in tie_nes])
            for ne in nes:
                if ne.velocity == 0 and (ne.program, ne.pitch) in last_activity:
                    last_activity.remove((ne.program, ne.pitch))
                if ne.velocity > 1:
                    last_activity.add((ne.program, ne.pitch))
        if prev_last_activity is not None and force_note_off_missing_tie:
            prog_pitch_tie = set([(ne.program, ne.pitch) for ne in tie_nes])
            for prog_pitch_pla in prev_last_activity:
                if prog_pitch_pla not in prog_pitch_tie:
                    merged_note_events.append(
                        NoteEvent(
                            is_drum=False,
                            program=prog_pitch_pla[0],
                            time=start_time,
                            velocity=0,
                            pitch=prog_pitch_pla[1],
                        )
                    )
                    seg_merge_err_cnt["Err/merging segment tie"] += 1
        merged_note_events += nes
        prev_last_activity = last_activity

    notes, err_cnt = note_event2note(
        merged_note_events, tie_note_events=None, fix_broken_notes=fix_broken_notes
    )
    err_cnt.update(seg_merge_err_cnt)
    return notes, err_cnt


def note_event2midi(
    note_events: list[NoteEvent],
    output_file: str | os.PathLike | None = None,
    velocity: int = 100,
    ticks_per_beat: int = 480,
    tempo: int = 500000,
) -> MidiFile:
    """Convert NoteEvent list to a MIDI file."""
    midi = MidiFile(ticks_per_beat=ticks_per_beat, type=0)
    track = MidiTrack()
    midi.tracks.append(track)

    drum_offset_events = []
    for ne in note_events:
        if ne.is_drum:
            drum_offset_events.append(
                NoteEvent(
                    is_drum=True,
                    program=ne.program,
                    time=ne.time + 0.01,
                    pitch=ne.pitch,
                    velocity=0,
                )
            )
    note_events = list(note_events) + drum_offset_events
    sort_note_events(note_events)

    program_to_channel: dict[int, int] = {}
    available_channels = list(range(0, 9)) + list(range(10, 16))
    drums_initialized = False
    current_tick = 0
    for ne in note_events:
        absolute_tick = round(second2tick(ne.time, ticks_per_beat, tempo))
        if absolute_tick == current_tick:
            delta_tick = 0
        elif absolute_tick < current_tick:
            raise ValueError(
                f"at ne.time {ne.time}, absolute_tick {absolute_tick} < current_tick {current_tick}"
            )
        else:
            delta_tick = absolute_tick - current_tick
            current_tick += delta_tick

        if ne.program in program_to_channel:
            ne_channel = program_to_channel[ne.program]
        elif ne.program == DRUM_PROGRAM or ne.is_drum:
            ne_channel = 9
            if not drums_initialized:
                track.append(
                    Message(
                        "program_change", program=0, time=delta_tick, channel=ne_channel
                    )
                )
                delta_tick = 0
                drums_initialized = True
        else:
            try:
                program_to_channel[ne.program] = available_channels.pop(0)
            except IndexError:
                program_to_channel[ne.program] = 15
            ne_channel = program_to_channel[ne.program]
            track.append(
                Message(
                    "program_change",
                    program=ne.program,
                    time=delta_tick,
                    channel=ne_channel,
                )
            )
            delta_tick = 0

        msg_note = "note_on" if ne.velocity > 0 else "note_off"
        msg_velocity = velocity if ne.velocity > 0 else 0
        track.append(
            Message(
                msg_note,
                note=ne.pitch,
                velocity=msg_velocity,
                time=delta_tick,
                channel=ne_channel,
            )
        )

    if output_file is not None:
        midi.save(output_file)
    return midi
