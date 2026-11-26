"""MT3 MIDI tokenizer.

Adapted from YourMT3+ (https://github.com/mimbres/YourMT3) and audiocraft_trans.
"""

import logging

from muscriptor.tokenizer.notes import (
    DRUM_PROGRAM,
    NoteEventTokenizer,
    Event,
)

logger = logging.getLogger(__name__)


def _get_drum_group_note_map(drum_vocabulary: str) -> dict[int, list[int]]:
    if drum_vocabulary == "FULL":
        return {i: [i] for i in range(128)}
    elif drum_vocabulary == "GM_DRUM_NOTES":
        ret = {
            0: [36, 35],
            1: [37, 2],
            2: [38, 40],
            3: [42, 44, 22],
            4: [46, 26],
            5: [56],
            6: [43],
            7: [41],
            8: [45],
            9: [47],
            10: [48],
            11: [50],
            12: [58],
            13: [51],
            14: [53],
            15: [59],
            16: [52],
            17: [49, 57],
            18: [55],
        }
    elif drum_vocabulary == "KICK_SNARE_HIHAT":
        ret = {0: [36, 35], 1: [38, 40], 2: [42, 44, 46, 22, 26]}
    else:
        assert False, drum_vocabulary
    notes_not_assigned = set(range(128)) - set([x for xs in ret.values() for x in xs])
    for n in notes_not_assigned:
        ret[len(ret)] = [n]
    return ret


def get_group_program_map(
    instrument_vocabulary: str,
    misc_programs: str,
    is_mt3: bool = False,
    include_drums: bool = False,
) -> dict[int, list[int]]:
    if instrument_vocabulary == "ONLY_PIANO":
        ret = {0: list(range(128))}
    elif instrument_vocabulary == "FULL":
        ret = {i: [i] for i in range(128)}
    elif instrument_vocabulary == "MT3_MIDI_PLUS":
        ret = {
            0: list(range(8)),
            1: list(range(8, 16)),
            2: list(range(16, 24)),
            3: list(range(24, 32)),
            4: list(range(32, 40)),
            5: list(range(40, 56)),
            6: list(range(56, 64)),
            7: list(range(64, 72)),
            8: list(range(72, 80)),
            9: list(range(80, 88)),
            10: list(range(88, 96)),
            11: list(range(100, 102)),
        }
    elif instrument_vocabulary == "MT3_FULL_PLUS":
        ret = {
            0: [0, 1, 3, 6, 7],
            1: [2, 4, 5],
            2: list(range(8, 16)),
            3: list(range(16, 24)),
            4: [24, 25],
            5: [26, 27, 28],
            6: [29, 30, 31],
            7: [32, 35],
            8: [33, 34, 36, 37, 38, 39],
            9: [40],
            10: [41],
            11: [42],
            12: [43],
            13: [46],
            14: [47],
            15: [48, 49, 44, 45],
            16: [50, 51],
            17: [52, 53, 54],
            18: [55],
            19: [56, 59],
            20: [57],
            21: [58],
            22: [60],
            23: [61, 62, 63],
            24: [64, 65],
            25: [66],
            26: [67],
            27: [68],
            28: [69],
            29: [70],
            30: [71],
            31: list(range(72, 80)),
            32: list(range(80, 88)),
            33: list(range(88, 96)),
            34: [100],
            35: [101],
        }
    elif instrument_vocabulary == "OURS_INSTRUMENT_GROUPS":
        ret = {
            0: list(range(8)),
            1: list(range(24, 32)),
            2: list(range(32, 40)),
            3: list(range(40, 56)),
            4: list(range(56, 64)),
            5: list(range(16, 24)) + list(range(64, 80)),
            6: list(range(80, 96)),
            7: list(range(8, 16)) + list(range(112, 119)),
        }
    else:
        assert False, instrument_vocabulary

    if instrument_vocabulary == "MT3_FULL_PLUS" and not is_mt3:
        not_assigned = set(range(130)) - set([v for vs in ret.values() for v in vs])
    else:
        not_assigned = set(range(128)) - set([v for vs in ret.values() for v in vs])
    if include_drums:
        not_assigned = not_assigned.union({DRUM_PROGRAM})
    if misc_programs == "ONE_GROUP":
        ret[len(ret)] = list(not_assigned)
    elif misc_programs == "SINGLETON_GROUPS":
        for p in not_assigned:
            ret[len(ret)] = [p]
    else:
        assert misc_programs == "OMIT", misc_programs
    return ret


# Human-readable names for the MT3_FULL_PLUS instrument groups (see
# get_group_program_map). Used by the CLI's --expected-instruments option.
MT3_FULL_PLUS_GROUP_NAMES: dict[str, int] = {
    "acoustic_piano": 0,
    "electric_piano": 1,
    "chromatic_percussion": 2,
    "organ": 3,
    "acoustic_guitar": 4,
    "clean_electric_guitar": 5,
    "distorted_electric_guitar": 6,
    "acoustic_bass": 7,
    "electric_bass": 8,
    "violin": 9,
    "viola": 10,
    "cello": 11,
    "contrabass": 12,
    "harp": 13,
    "timpani": 14,
    "strings": 15,
    "synth_strings": 16,
    "voice": 17,
    "orchestra_hit": 18,
    "trumpet": 19,
    "trombone": 20,
    "tuba": 21,
    "french_horn": 22,
    "brass": 23,
    "high_sax": 24,
    "tenor_sax": 25,
    "baritone_sax": 26,
    "oboe": 27,
    "english_horn": 28,
    "bassoon": 29,
    "clarinet": 30,
    "pipe": 31,
    "synth_lead": 32,
    "synth_pad": 33,
    "fx_brightness": 34,
    "fx_goblins": 35,
}


# Names of the other groups of the instrument_group condition (muscriptor.tokenizer.v2.PREFIX_GROUPS: after the 36
# MT3_FULL_PLUS groups come singletons for drums, program 129 (group 37, unnamed) and the General MIDI programs outside
# MT3_FULL_PLUS).
EXTRA_GROUP_NAMES: dict[str, int] = {
    "drums": 36,
    "fx_rain": 38,
    "fx_soundtrack": 39,
    "fx_crystal": 40,
    "fx_atmosphere": 41,
    "fx_echoes": 42,
    "fx_sci_fi": 43,
    "sitar": 44,
    "banjo": 45,
    "shamisen": 46,
    "koto": 47,
    "kalimba": 48,
    "bagpipe": 49,
    "fiddle": 50,
    "shanai": 51,
    "tinkle_bell": 52,
    "agogo": 53,
    "steel_drums": 54,
    "woodblock": 55,
    "taiko_drum": 56,
    "melodic_tom": 57,
    "synth_drum": 58,
    "reverse_cymbal": 59,
    "guitar_fret_noise": 60,
    "breath_noise": 61,
    "seashore": 62,
    "bird_tweet": 63,
    "telephone_ring": 64,
    "helicopter": 65,
    "applause": 66,
    "gunshot": 67,
}
# Every name accepted by --instruments: group name -> group id of the instrument_group condition.
INSTRUMENT_GROUP_NAMES: dict[str, int] = {
    **MT3_FULL_PLUS_GROUP_NAMES,
    **EXTRA_GROUP_NAMES,
}


def group_program_map_to_amt_vocab_dict(instrument_group_to_program_map):
    midi_instrument_names = [
        "Acoustic Grand Piano",
        "Bright Acoustic Piano",
        "Electric Grand Piano",
        "Honky-tonk Piano",
        "Electric Piano 1",
        "Electric Piano 2",
        "Harpsichord",
        "Clavinet",
        "Celesta",
        "Glockenspiel",
        "Music Box",
        "Vibraphone",
        "Marimba",
        "Xylophone",
        "Tubular Bells",
        "Dulcimer",
        "Drawbar Organ",
        "Percussive Organ",
        "Rock Organ",
        "Church Organ",
        "Reed Organ",
        "Accordion",
        "Harmonica",
        "Tango Accordion",
        "Acoustic Guitar (nylon)",
        "Acoustic Guitar (steel)",
        "Electric Guitar (jazz)",
        "Electric Guitar (clean)",
        "Electric Guitar (muted)",
        "Overdriven Guitar",
        "Distortion Guitar",
        "Guitar Harmonics",
        "Acoustic Bass",
        "Electric Bass (finger)",
        "Electric Bass (pick)",
        "Fretless Bass",
        "Slap Bass 1",
        "Slap Bass 2",
        "Synth Bass 1",
        "Synth Bass 2",
        "Violin",
        "Viola",
        "Cello",
        "Contrabass",
        "Tremolo Strings",
        "Pizzicato Strings",
        "Orchestral Harp",
        "Timpani",
        "String Ensemble 1",
        "String Ensemble 2",
        "Synth Strings 1",
        "Synth Strings 2",
        "Choir Aahs",
        "Voice Oohs",
        "Synth Choir",
        "Orchestra Hit",
        "Trumpet",
        "Trombone",
        "Tuba",
        "Muted Trumpet",
        "French Horn",
        "Brass Section",
        "Synth Brass 1",
        "Synth Brass 2",
        "Soprano Sax",
        "Alto Sax",
        "Tenor Sax",
        "Baritone Sax",
        "Oboe",
        "English Horn",
        "Bassoon",
        "Clarinet",
        "Piccolo",
        "Flute",
        "Recorder",
        "Pan Flute",
        "Blown bottle",
        "Shakuhachi",
        "Whistle",
        "Ocarina",
        "Lead 1 (square)",
        "Lead 2 (sawtooth)",
        "Lead 3 (calliope)",
        "Lead 4 chiff",
        "Lead 5 (charang)",
        "Lead 6 (voice)",
        "Lead 7 (fifths)",
        "Lead 8 (bass + lead)",
        "Pad 1 (new age)",
        "Pad 2 (warm)",
        "Pad 3 (polysynth)",
        "Pad 4 (choir)",
        "Pad 5 (bowed)",
        "Pad 6 (metallic)",
        "Pad 7 (halo)",
        "Pad 8 (sweep)",
        "FX 1 (rain)",
        "FX 2 (soundtrack)",
        "FX 3 (crystal)",
        "FX 4 (atmosphere)",
        "FX 5 (brightness)",
        "FX 6 (goblins)",
        "FX 7 (echoes)",
        "FX 8 (sci-fi)",
        "Sitar",
        "Banjo",
        "Shamisen",
        "Koto",
        "Kalimba",
        "Bagpipe",
        "Fiddle",
        "Shanai",
        "Tinkle Bell",
        "Agogo",
        "Steel Drums",
        "Woodblock",
        "Taiko Drum",
        "Melodic Tom",
        "Synth Drum",
        "Reverse Cymbal",
        "Guitar Fret Noise",
        "Breath Noise",
        "Seashore",
        "Bird Tweet",
        "Telephone Ring",
        "Helicopter",
        "Applause",
        "Gunshot",
    ]
    return {
        midi_instrument_names[v[0]]: v for v in instrument_group_to_program_map.values()
    }


class MT3Tokenizer:
    def __init__(
        self,
        instrument_vocabulary: str = "FULL",
        drum_vocabulary: str = "FULL",
        max_shift_steps: int = 1001,
        add_instrument_section: bool = False,
        modulo_max_shift_steps: bool = False,
        frame_rate: int = 100,
    ):
        self.group_program_map = get_group_program_map(
            instrument_vocabulary, misc_programs="SINGLETON_GROUPS", is_mt3=True
        )
        drum_group_note_map = _get_drum_group_note_map(drum_vocabulary)
        self.tokenizer = NoteEventTokenizer(
            max_shift_steps=max_shift_steps,
            frame_rate=frame_rate,
            extra_tokens=["<INST_SEC_START>", "<INST_SEC_END>"]
            if add_instrument_section
            else [],
            program_vocabulary=group_program_map_to_amt_vocab_dict(
                self.group_program_map
            ),
            drum_vocabulary=drum_group_note_map,
            modulo_max_shift_steps=modulo_max_shift_steps,
        )
        self.add_instrument_section = add_instrument_section

        logger.info(f"MT3Tokenizer: {self.tokenizer.num_tokens} tokens")

    def _drop_instrument_section(self, tokens):
        if not self.add_instrument_section:
            return tokens
        try:
            drop_until = next(
                i
                for i in reversed(range(len(tokens)))
                if tokens[i]
                == self.tokenizer.event_tokenizer.encode_event(
                    Event("<INST_SEC_END>", 0)
                )
            )
        except StopIteration:
            drop_until = 0
        return tokens[drop_until:]
