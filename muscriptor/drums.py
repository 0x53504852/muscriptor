"""Drum pitches of the MuScriptor V2 models -> General MIDI drum notes.

The V2 models were trained on drum targets renumbered to one canonical pitch per drum class (training repo:
amt/drum_canon.py, "drum_canon 2 (24 Nov 2026) | renumber"). The canonical pitches are General MIDI numbers already
(36 kick, 38 snare, 42 / 44 / 46 hi-hat closed / pedal / open, 43 / 45 / 48 low / mid / high tom, 49 crash, 51 ride, ...),
so for them the map is the identity. The training data also hold Guitar Pro articulation numbers that are not General MIDI
drums (29, 31, 91-127: rim shot, half-open hi-hat, ride edge, chokes, ...); a model that writes one gets the General MIDI
number of its parent instrument (alphaTab's PercussionMapper names, as amt/drum_canon.py GP_TO_GM), then the canonical
class. Any other pitch is kept as written.

    to_general_midi(57) -> 49 (crash 2 -> crash 1)      to_general_midi(91) -> 38 (rim shot -> snare)
"""

# canonical GM pitch: (class, GM pitches that map onto it)  (amt/drum_canon.py CLASSES)
CLASSES = {
    36: ("kick", (35, 36)),
    38: ("snare", (38, 40)),
    37: ("side_stick", (37,)),
    39: ("clap", (39,)),
    42: ("hihat_closed", (42,)),
    44: ("hihat_pedal", (44,)),
    46: ("hihat_open", (46,)),
    43: ("tom_floor", (41, 43)),
    45: ("tom_mid", (45, 47)),
    48: ("tom_high", (48, 50)),
    49: ("crash", (49, 57)),
    51: ("ride", (51, 59)),
    53: ("ride_bell", (53,)),
    52: ("china", (52,)),
    55: ("splash", (55,)),
    54: ("tambourine", (54,)),
    56: ("cowbell", (56,)),
    58: ("vibraslap", (58,)),
    60: ("bongo_high", (60,)),
    61: ("bongo_low", (61,)),
    62: ("conga_high_mute", (62,)),
    63: ("conga_high_open", (63,)),
    64: ("conga_low", (64,)),
    65: ("timbale_high", (65,)),
    66: ("timbale_low", (66,)),
    67: ("agogo_high", (67,)),
    68: ("agogo_low", (68,)),
    69: ("cabasa", (69,)),
    70: ("maracas", (70,)),
    71: ("whistle_short", (71,)),
    72: ("whistle_long", (72,)),
    73: ("guiro_short", (73,)),
    74: ("guiro_long", (74,)),
    75: ("claves", (75,)),
    76: ("woodblock_high", (76,)),
    77: ("woodblock_low", (77,)),
    78: ("cuica_mute", (78,)),
    79: ("cuica_open", (79,)),
    80: ("triangle_mute", (80,)),
    81: ("triangle_open", (81,)),
    82: ("shaker", (82,)),
    83: ("jingle_bell", (83,)),
    84: ("belltree", (84,)),
    85: ("castanets", (85,)),
    86: ("surdo", (86, 87)),
}
# Guitar Pro articulation number -> General MIDI number of its parent instrument (amt/drum_canon.py GP_TO_GM)
GP_TO_GM = {
    29: 59,
    31: 37,
    91: 38,
    92: 46,
    93: 51,
    94: 51,
    95: 55,
    96: 52,
    97: 49,
    98: 57,
    99: 56,
    100: 56,
    101: 56,
    102: 56,
    103: 56,
    104: 60,
    105: 60,
    106: 61,
    107: 61,
    108: 64,
    109: 64,
    110: 63,
    111: 54,
    112: 54,
    113: 54,
    114: 36,
    115: 49,
    116: 49,
    117: 69,
    118: 70,
    119: 70,
    120: 70,
    122: 82,
    123: 84,
    124: 62,
    125: 62,
    126: 59,
    127: 53,
}
MAP = {src: can for can, (_, srcs) in CLASSES.items() for src in srcs}


def to_general_midi(pitch: int) -> int:
    """General MIDI drum note of a drum pitch written by the model (see the module docstring)."""
    p = GP_TO_GM.get(int(pitch), int(pitch))
    return MAP.get(p, p)


def drum_name(pitch: int) -> str:
    """Class name of a drum pitch ('other_<pitch>' outside the table)."""
    p = to_general_midi(pitch)
    return CLASSES[p][0] if p in CLASSES else f"other_{p}"
