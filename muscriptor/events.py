"""Public streaming events and per-chunk event builder.

`TranscriptionModel.transcribe` is a generator that yields these dataclasses
one at a time. Every :class:`NoteStartEvent` is guaranteed to be followed by
exactly one matching :class:`NoteEndEvent` (same `index`) later in the stream.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from muscriptor.tokenizer.notes import (
    MINIMUM_NOTE_DURATION_SEC,
    NoteEvent,
    TieNoteEvent,
)


_DRUM_INSTRUMENT = "drums"


@dataclass
class NoteStartEvent:
    pitch: int
    start_time: float
    index: int
    instrument: str


@dataclass
class NoteEndEvent:
    end_time: float
    start_event: NoteStartEvent

    @property
    def start_event_index(self) -> int:
        return self.start_event.index


class _StreamingEventBuilder:
    """Turns per-chunk (NoteEvent, TieNoteEvent) lists into a stream of
    NoteStartEvent / NoteEndEvent.

    The builder is stateful across chunks: a note opened in chunk N but not
    closed there stays open in `self.open_notes` and is only emitted as a
    NoteEnd either when its `velocity=0` event arrives in a later chunk or
    when chunk N+1's tie events show it has been dropped.
    """

    def __init__(self, instrument_for_program: Callable[[int], str]):
        self.open_notes: dict[tuple[int, int], NoteStartEvent] = {}
        self._next_index = 0
        self._instrument_for_program = instrument_for_program

    def process_chunk(
        self,
        note_events: list[NoteEvent],
        tie_note_events: list[TieNoteEvent],
        seek_time: float,
        next_seek_time: float | None,
    ) -> Iterator[NoteStartEvent | NoteEndEvent]:
        # Tie-stitch: any prior-chunk open note that is NOT in this chunk's
        # tie set must end at the boundary (start time of this chunk).
        tie_set = {(t.program, t.pitch) for t in tie_note_events}
        for key in list(self.open_notes.keys()):
            if key not in tie_set:
                yield NoteEndEvent(
                    end_time=seek_time,
                    start_event=self.open_notes.pop(key),
                )

        # Events past the next chunk's start belong to the next chunk —
        # the model occasionally emits shifts that overshoot the window.
        if next_seek_time is not None:
            note_events = [ne for ne in note_events if ne.time < next_seek_time]

        for ne in note_events:
            if ne.is_drum:
                start = self._mint_start(ne.pitch, ne.time, _DRUM_INSTRUMENT)
                yield start
                yield NoteEndEvent(
                    end_time=ne.time + MINIMUM_NOTE_DURATION_SEC,
                    start_event=start,
                )
            elif ne.velocity == 1:
                key = (ne.program, ne.pitch)
                if key in self.open_notes:
                    yield NoteEndEvent(
                        end_time=ne.time, start_event=self.open_notes.pop(key)
                    )
                start = self._mint_start(
                    ne.pitch, ne.time, self._instrument_for_program(ne.program)
                )
                self.open_notes[key] = start
                yield start
            else:  # velocity == 0
                key = (ne.program, ne.pitch)
                if key in self.open_notes:
                    yield NoteEndEvent(
                        end_time=ne.time, start_event=self.open_notes.pop(key)
                    )
                # orphan offsets are silently dropped

    def flush(self) -> Iterator[NoteEndEvent]:
        """Close any still-open notes after the final chunk.

        Uses the same minimum-duration fallback as the legacy decoder
        (matches `note_event2note` behavior when no segment-end is known).
        """
        for ev in self.open_notes.values():
            yield NoteEndEvent(
                end_time=ev.start_time + MINIMUM_NOTE_DURATION_SEC,
                start_event=ev,
            )
        self.open_notes.clear()

    def _mint_start(
        self, pitch: int, start_time: float, instrument: str
    ) -> NoteStartEvent:
        ev = NoteStartEvent(
            pitch=pitch,
            start_time=start_time,
            index=self._next_index,
            instrument=instrument,
        )
        self._next_index += 1
        return ev
