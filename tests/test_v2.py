"""MuScriptor V2 pieces on CPU with a tiny random model: weight loading, the batched ensemble decoder against
LMModel.generate, the vote, the window decoding, the drum map, the resampler and the multi-track MIDI writer."""

import io
import math
import random

import pytest
import torch
from mido import MidiFile
from safetensors.torch import save_file

from muscriptor import ensemble as ens
from muscriptor.drums import to_general_midi
from muscriptor.modules.conditioners import ConditioningAttributes, WavCondition
from muscriptor.tokenizer import v2 as v2tok
from muscriptor.tokenizer.notes import Note
from muscriptor.transcription_model import (
    TranscriptionModel,
    _build_model,
    _ModelConfig,
)
from muscriptor.utils.resample import resample_frac
from muscriptor.vote import note_rows, vote, vote_notes

SR = 16000
SEG = 5 * SR


def _tiny_lm(seed: int = 1, eos_scale: float = 2.0):
    """A random compact-vocabulary model (2 layers, dim 64); the EOS row scaled up so that windows end at various
    steps."""
    torch.manual_seed(seed)
    lm = _build_model(
        torch.device("cpu"), _ModelConfig(dim=64, num_heads=4, num_layers=2, card=830)
    )
    lm.token_of_class = torch.tensor(v2tok.COMPACT_IDS)
    lm.autocast_conditions = True
    lm.verbose = False
    lm.condition_provider.conditioners["self_wav"].verbose = False
    with torch.no_grad():
        for p in lm.parameters():
            if p.dim() == 2:
                p.normal_(0, 0.05)
        lm.linears[0].weight[1] *= eos_scale
    return lm.eval()


def _windows(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, SEG, generator=g) * 0.1


def _plain(
    lm, wav: torch.Tensor, prompt: str | None, cfg: float, max_gen_len: int
) -> list[list[int]]:
    """LMModel.generate (greedy) on the windows of one batch, as token lists cut after the first EOS."""
    conds = [
        ConditioningAttributes(
            wav={
                "self_wav": WavCondition(
                    w[None, None], torch.tensor([SEG]), [SR], [None], [0.0]
                )
            },
            text={"instrument_group": prompt, "dataset_name": None},
        )
        for w in wav
    ]
    out = lm.generate(
        conditions=conds,
        max_gen_len=max_gen_len,
        use_sampling=False,
        cfg_coef=cfg,
        early_stop_on_token=v2tok.EOS,
    )
    rows = []
    for seq in lm.classes_to_tokens(out[:, 0]).tolist():
        rows.append(seq[: seq.index(v2tok.EOS) + 1] if v2tok.EOS in seq else seq)
    return rows


@pytest.fixture(scope="module")
def lm():
    return _tiny_lm(seed=1, eos_scale=2.0)


@pytest.mark.parametrize("cfg", [1.0, 1.5])
def test_generate_calls_matches_generate(lm, cfg):
    """Greedy rows of the batched decoder = separate LMModel.generate calls (float32 CPU), with batches of two prompt
    widths in one wave and split over waves."""
    batches = [
        ens.WindowBatch(_windows(3, 1), "0 4", None),
        ens.WindowBatch(_windows(2, 2), None, None),
        ens.WindowBatch(_windows(2, 3), "36", None),
    ]
    ref = [_plain(lm, b.wav, b.instrument_group, cfg, 40) for b in batches]
    calls = [ens.Call(i, cfg_coef=cfg) for i in range(len(batches))]
    for max_rows in (None, 2 * 3):
        out = ens.generate_calls(
            lm,
            batches,
            calls,
            max_gen_len=40,
            max_rows=max_rows,
            max_slots=max_rows,
            backend="torch",
            eos_check_every=4,
        )
        assert out == ref
    lengths = {len(r) for rows in ref for r in rows}
    assert (
        len(lengths) > 1
    )  # rows finished at different steps (the compaction was exercised)


def test_shared_prefix_rows(lm):
    """Several calls on one window batch (greedy, CFG, sampled) share its slots; each greedy call still equals its
    stand-alone generate()."""
    b = ens.WindowBatch(_windows(2, 4), "0", None)
    g = torch.Generator().manual_seed(7)
    calls = [
        ens.Call(0),
        ens.Call(0, cfg_coef=1.25),
        ens.Call(0, temp=0.7, generator=g),
    ]
    out = ens.generate_calls(lm, [b], calls, max_gen_len=30, backend="torch")
    assert out[0] == _plain(lm, b.wav, "0", 1.0, 30)
    assert out[1] == _plain(lm, b.wav, "0", 1.25, 30)


def test_sampled_rows_follow_their_generator(lm):
    """A sampled call draws one [n, card] Exp(1) block per step until its last row's EOS: the same seed gives the same
    tokens wherever the call runs (alone or with others), and the generator ends where `end` draws leave it."""
    b0 = ens.WindowBatch(_windows(2, 5), "0", None)
    b1 = ens.WindowBatch(_windows(3, 6), None, None)

    def run(batches, idx, temp=0.8):
        g = torch.Generator().manual_seed(11)
        calls = [
            ens.Call(
                i,
                temp=temp,
                generator=g if i == idx else torch.Generator().manual_seed(3),
            )
            for i in range(len(batches))
        ]
        return ens.generate_calls(
            lm, batches, calls, max_gen_len=30, backend="torch", eos_check_every=8
        )[idx], g

    alone, g_alone = run([b0], 0)
    mixed, g_mixed = run([b1, b0], 1)
    assert alone == mixed
    end = max(len(r) for r in alone)
    g = torch.Generator().manual_seed(11)
    block = torch.empty(2, 830)
    for _ in range(end):
        block.exponential_(1, generator=g)
    assert torch.equal(g.get_state(), g_alone.get_state())
    assert torch.equal(g.get_state(), g_mixed.get_state())


def test_consecutive_batches_of_a_generator_go_to_later_waves():
    b = [ens.WindowBatch(torch.zeros(2, 10)), ens.WindowBatch(torch.zeros(2, 10))]
    g = torch.Generator()
    calls = [
        ens.Call(0, temp=0.5, generator=g),
        ens.Call(1, temp=0.5, generator=g),
        ens.Call(1),
    ]
    waves = ens.plan_waves(b, calls, max_rows=100, max_slots=100)
    assert waves == [[0, 2], [1]]


def test_recipes():
    a, b = ens.RECIPES["A"], ens.RECIPES["B"]
    assert len(a) == len(b) == 13
    assert [v.name for v in a][:5] == [
        "cfg1.25_p",
        "base_p",
        "ov2.5_p",
        "ov3.0_p",
        "ov3.5_p",
    ]
    assert [(v.cfg_coef, v.hop) for v in b[:5]] == [
        (1.0, 4.0),
        (1.25, 4.0),
        (1.25, 3.0),
        (1.25, 2.5),
        (1.0, 3.0),
    ]
    assert sum(v.temp == 0.5 for v in a) == sum(v.temp == 1.0 for v in b) == 4
    assert ens.parse_variants("cfg=1.25;hop=2.5;temp=0.5,seed=1") == (
        ens.Variant("cfg1.25_t0_s-1_hop4", 1.25),
        ens.Variant("cfg1_t0_s-1_hop2.5", hop=2.5),
        ens.Variant("cfg1_t0.5_s1_hop4", temp=0.5, seed=1),
    )
    with pytest.raises(ValueError):
        ens.parse_variants("cfg=1;foo=2")


# ---------------------------------------------------------------------------
# vote
# ---------------------------------------------------------------------------


def _n(onset, pitch, offset=None, program=0, drum=False):
    return Note(
        drum,
        128 if drum else program,
        onset,
        onset + 0.5 if offset is None else offset,
        pitch,
    )


def test_vote_keeps_supported_clusters_with_medians():
    decs = [
        [_n(1.00, 60, 1.5)],
        [_n(1.02, 60, 1.6)],
        [_n(1.04, 60, 1.7)],
        [_n(2.0, 62)],
        [],
    ]
    voted = vote_notes(
        vote([note_rows(d) for d in decs], 3, tol=0.05, median="onoff", dedup=0.05)
    )
    assert len(voted) == 1
    n = voted[0]
    assert (n.pitch, n.onset, n.offset) == (60, 1.02, 1.6)
    assert vote([note_rows(d) for d in decs], 4, tol=0.05) == []


def test_vote_one_note_per_decode_and_tolerance():
    decs = [[_n(1.0, 60), _n(1.01, 60)], [_n(1.0, 60)], [_n(1.2, 60)]]
    rows = vote([note_rows(d) for d in decs], 2, tol=0.05)
    assert [(r["pitch"], r["onset"]) for r in rows] == [("60", "1.0")]


def test_vote_dedup_keeps_the_better_supported():
    a, b = _n(1.00, 60), _n(1.06, 60)
    decs = [[a], [a], [b], [b], [b]]
    rows = vote([note_rows(d) for d in decs], 2, tol=0.05, median="on", dedup=0.1)
    assert len(rows) == 1 and rows[0]["onset"] == "1.0600"


# ---------------------------------------------------------------------------
# window decoding, drums, resampling, MIDI
# ---------------------------------------------------------------------------

T = v2tok.TOKEN_ID


def _tok(*events):
    return [T[e] for e in events] + [v2tok.EOS]


def test_decode_track_overlap_central_regions():
    """Windows at 0 and 4 s overlap on 4-5 s; the boundary is 4.5 s: the first window owns events before it, the
    second those after. A note held across the boundary in both windows is continued."""
    w0 = _tok(
        ("tie", 0),
        ("shift", 100),
        ("program", 0),
        ("velocity", 1),
        ("pitch", 60),
        ("shift", 420),
        ("pitch", 64),
        ("shift", 470),
        ("pitch", 67),
    )
    w1 = _tok(
        ("program", 0),
        ("pitch", 60),
        ("tie", 0),
        ("shift", 30),
        ("program", 0),
        ("velocity", 1),
        ("pitch", 67),
        ("shift", 60),
        ("velocity", 0),
        ("pitch", 60),
    )
    notes, _ = v2tok.decode_track_overlap([w0, w1], [0.0, 4.0])
    got = sorted((n.pitch, round(n.onset, 2), round(n.offset, 2)) for n in notes)
    # 60: onset 1.0 (window 0), held, ended at 4.6 by window 1; 64: onset 4.2 (window 0, region < 4.5), not sounding
    # in window 1 -> closed at 4.5; 67: window 0's 4.7 onset is not in its region, window 1's at 4.3 is not in its own
    # either (before 4.5) -> no note 67.
    assert got == [(60, 1.0, 4.6), (64, 4.2, 4.5)]


def test_decode_track_back_to_back():
    w0 = _tok(
        ("tie", 0), ("shift", 100), ("program", 32), ("velocity", 1), ("pitch", 40)
    )
    w1 = _tok(
        ("program", 32),
        ("pitch", 40),
        ("tie", 0),
        ("shift", 50),
        ("velocity", 0),
        ("pitch", 40),
    )
    notes, _ = v2tok.decode_track([w0, w1], [0.0, 5.0])
    assert [(n.program, n.pitch, n.onset, n.offset) for n in notes] == [
        (32, 40, 1.0, 5.5)
    ]


def test_compact_ids():
    assert len(v2tok.COMPACT_IDS) == 830 and v2tok.COMPACT_IDS[:2] == [0, 1]
    assert (
        T[("shift", 500)] in v2tok.COMPACT_IDS
        and T[("shift", 501)] not in v2tok.COMPACT_IDS
    )
    assert v2tok.PREFIX_GROUPS[v2tok.DRUM_GROUP] == [128]


def test_drum_map():
    for p in (36, 38, 42, 44, 46, 43, 45, 48, 49, 51):
        assert to_general_midi(p) == p  # canonical pitches are GM notes already
    assert (
        to_general_midi(35) == 36
        and to_general_midi(57) == 49
        and to_general_midi(91) == 38
    )
    assert to_general_midi(10) == 10  # outside every table: unchanged


def test_resample_frac():
    x = torch.ones(44100)
    y = resample_frac(x, 44100, 16000)
    assert y.shape == (16000,)
    assert torch.allclose(y, torch.ones_like(y), atol=1e-4)
    t = torch.arange(48000) / 48000
    y = resample_frac(torch.sin(2 * math.pi * 440 * t), 48000, 16000)
    ref = torch.sin(2 * math.pi * 440 * torch.arange(16000) / 16000)
    assert (y[200:-200] - ref[200:-200]).abs().max() < 1e-3


def test_multitrack_midi():
    from muscriptor.utils.midi import notes_to_multitrack_midi

    notes = [
        Note(False, 0, 0.0, 1.0, 60),
        Note(False, 0, 0.5, 1.5, 60),
        Note(False, 33, 0.0, 2.0, 40),
        Note(True, 128, 0.25, 0.26, 57),
    ]
    names = {0: "acoustic_piano", 33: "electric_bass", 128: "drums"}
    midi = notes_to_multitrack_midi(notes, lambda n: names[n.program], to_general_midi)
    assert midi.type == 1 and len(midi.tracks) == 4
    track_names = [
        m.name for tr in midi.tracks[1:] for m in tr if m.type == "track_name"
    ]
    assert track_names == ["acoustic_piano", "electric_bass", "drums"]
    drum = [m for m in midi.tracks[3] if m.type == "note_on"]
    assert drum[0].channel == 9 and drum[0].note == 49
    piano = [
        (m.type, m.note, m.time)
        for m in midi.tracks[1]
        if m.type in ("note_on", "note_off")
    ]
    assert (
        piano[0][0] == "note_on" and piano[1][0] == "note_off"
    )  # the overlapping 60s are cut, never nested


# ---------------------------------------------------------------------------
# loading V2 weights + the note-level API
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v2_model(tmp_path_factory):
    """A tiny random model saved with the training repository's key names, loaded by TranscriptionModel."""
    lm = _tiny_lm(seed=0, eos_scale=4.0)
    state = {}
    for k, v in lm.state_dict().items():
        k = k.replace("emb.0.", "emb.").replace("linears.0.", "linear.")
        state[k] = v.contiguous()
    path = tmp_path_factory.mktemp("w") / "tiny_v2.safetensors"
    save_file(state, str(path))
    return TranscriptionModel.load_model(path, device="cpu")


def test_v2_weights_load(v2_model):
    m = v2_model
    assert m.is_v2 and m._model.card == 830 and m._model.autocast_conditions
    assert m._model.token_of_class.tolist() == v2tok.COMPACT_IDS
    assert m.instrument_prompt(["drums", "acoustic_piano", 33]) == [0, 33, 36]
    assert (
        m.resolve_ensemble("auto", True) == "A"
        and m.resolve_ensemble("auto", False) == "B"
    )
    assert m.resolve_ensemble("none", True) is None


def test_v2_transcribe_notes_single_and_ensemble(v2_model):
    """Two inputs (one shorter than a window), single decode and a small ensemble; the ensemble's greedy hop-4 decode
    equals the single decode."""
    m = v2_model
    random.seed(0)
    a = (torch.randn(SR * 9) * 0.1, SR)
    b = (torch.randn(SR * 3) * 0.1, 22050)
    single = dict(
        m.transcribe_notes(
            [a, b], instruments=["acoustic_piano"], max_gen_len=24, verbose=False
        )
    )
    assert set(single) == {0, 1}
    plain = dict(
        m.transcribe_notes(
            [a, b],
            instruments=["acoustic_piano"],
            max_gen_len=24,
            verbose=False,
            decoder="generate",
        )
    )
    assert (
        plain == single
    )  # the batched decoder's single decode = LMModel.generate on the windows
    spec = (
        "name=g,cfg=1,hop=4;name=c,cfg=1.25,hop=4;name=h,hop=2.5;name=s,temp=0.7,seed=1"
    )
    out = dict(
        m.transcribe_notes(
            [a, b],
            instruments=["acoustic_piano"],
            ensemble=spec,
            max_gen_len=24,
            return_decodes=True,
            verbose=False,
            backend="torch",
        )
    )
    for i in (0, 1):
        _, decodes = out[i]
        assert len(decodes) == 4
        assert decodes[0] == single[i]
    midi = MidiFile(
        file=io.BytesIO(m.notes_to_midi(single[0] + [Note(True, 128, 0.5, 0.51, 35)]))
    )
    assert midi.type == 1
    events = list(m.notes_to_events(single[0]))
    starts = [e for e in events if hasattr(e, "pitch")]
    assert len(events) == 2 * len(starts) == 2 * len(single[0])


def test_v2_old_api_routes_to_notes(v2_model):
    m = v2_model
    audio = (torch.randn(SR * 6) * 0.1, SR)
    events = list(m.transcribe(audio, instrument_group="0"))
    assert len(events) % 2 == 0
    assert m.transcribe_to_midi(audio, instrument_group="0")[:4] == b"MThd"


def test_out_of_memory_mid_wave_evicts_and_redoes_calls(monkeypatch):
    """An out-of-memory error when the generated-key cache grows: the last calls leave the wave and are decoded
    again in a later wave (sampled ones from their generator's wave-start state): the same tokens as without it."""
    lm = _tiny_lm(
        seed=0, eos_scale=0.0
    )  # rows rarely end: the cache has to grow past its first size
    batches = [
        ens.WindowBatch(_windows(2, 8), "0", None),
        ens.WindowBatch(_windows(1, 9), None, None),
        ens.WindowBatch(_windows(2, 10), "36", None),
    ]

    def calls():
        return [
            ens.Call(0),
            ens.Call(1, temp=0.9, generator=torch.Generator().manual_seed(5)),
            ens.Call(2, cfg_coef=1.5),
        ]

    ref = ens.generate_calls(lm, batches, calls(), max_gen_len=150, backend="torch")
    real_empty = torch.empty
    state = {"armed": True}

    def flaky_empty(*shape, **kw):
        dims = (
            shape[0]
            if len(shape) == 1 and isinstance(shape[0], (tuple, list))
            else shape
        )
        if state["armed"] and len(dims) == 4 and dims[1] > ens.CHUNK:
            state["armed"] = False
            raise torch.cuda.OutOfMemoryError("simulated")
        return real_empty(*shape, **kw)

    monkeypatch.setattr(torch, "empty", flaky_empty)
    stats = {}
    out = ens.generate_calls(
        lm, batches, calls(), max_gen_len=150, backend="torch", stats=stats
    )
    assert not state["armed"] and stats.get("evicted", 0) >= 1
    assert out == ref


def test_out_of_memory_for_a_single_call_splits_its_windows(monkeypatch):
    """A wave of one call that does not fit: the call's windows are decoded in two halves (greedy: the same tokens)."""
    lm = _tiny_lm(seed=1, eos_scale=2.0)
    batches = [ens.WindowBatch(_windows(4, 12), "0", None)]
    ref = ens.generate_calls(
        lm, batches, [ens.Call(0)], max_gen_len=40, backend="torch"
    )
    real = ens._run_wave
    state = {"n": 0}

    def flaky(lm_, device, backend, batches_, calls_, *a, **k):
        state["n"] += 1
        if batches_[calls_[0].batch].n == 4:
            raise torch.cuda.OutOfMemoryError("simulated")
        return real(lm_, device, backend, batches_, calls_, *a, **k)

    monkeypatch.setattr(ens, "_run_wave", flaky)
    stats = {}
    out = ens.generate_calls(
        lm, batches, [ens.Call(0)], max_gen_len=40, backend="torch", stats=stats
    )
    assert (
        stats["splits"] == 1 and state["n"] == 2
    )  # the two halves then fit in one wave
    assert out == ref
