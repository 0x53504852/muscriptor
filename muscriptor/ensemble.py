"""Ensemble decoding: many window decodes of one model run together as the rows of one batch, then voted.

Port of the training repository's amt/ensemble.py (branch gabriel/v2-fast-vote, commit 9309218) in plain PyTorch.

`generate_calls(lm, batches, calls)` returns, for each `Call(batch, cfg_coef, temp, generator)`, the tokens that
`lm.generate(conditions of batches[batch], cfg_coef=cfg_coef, temp=temp)` returns when its sampling noise comes from
`generator` (one [n, card] Exp(1) draw per step, sampled class = argmax(softmax(logits / temp) / noise), the
training code's `sample_one`), and leaves the generator where that stand-alone call would leave it.

- Shared prefix: the audio / dataset / instruments prefix + BOS of a window is run once per condition (conditional, plus
  unconditional when a call on it uses classifier-free guidance) into a "slot"; every row decoding that window reads the
  slot's keys / values. On CUDA the slot attention is one variable-length flash-attention call per layer in which the
  rows of a slot are the queries of one sequence, so a slot is read once per step for all its rows.
- Rows: a call of n windows gives n rows (+ n unconditional twins with CFG: a twin reads the unconditional slot, is fed
  its row's tokens, and its logits guide its row's: uncond + (cond - uncond) * cfg_coef). Each row has its own cfg /
  temperature / noise.
- Finished rows (EOS emitted, their twins with them) leave the batch at the host checks (every `eos_check_every` steps):
  the rows still decoding are compacted, with their generated keys / values.
- Several songs per batch: the calls are planned into waves of at most `max_rows` rows and `max_slots` slots; calls that
  share a generator (one variant's consecutive window batches of a song) go to successive waves, so that each draws the
  random stream of its stand-alone run.

Numerics: the same per-row computation as LMModel.generate under the same autocast; in bf16 the logits can differ in the
last bits (other batch shapes, a two-part attention merged by log-sum-exp), so greedy rows match stand-alone runs up to
bf16 rounding and sampled rows follow the same noise until a rounding flips a draw. In float32 on CPU the tokens match.

Attention backends: "flash" (CUDA, bf16 / fp16, compute capability >= 8.0: PyTorch's built-in flash-attention kernels,
the default where available) and "torch" (anywhere: slot keys gathered per row + masked scaled_dot_product_attention;
slower on GPU, used on CPU).
"""

import math
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from muscriptor.modules.conditioners import ConditioningAttributes, WavCondition
from muscriptor.modules.streaming import init_states
from muscriptor.modules.transformer import create_sin_embedding
from muscriptor.tokenizer.v2 import EOS, VOCAB_SIZE

CHUNK = 128  # first size of the generated-key cache (then doubled as needed; CUDA graphs are captured per size)
GROW_MAX = 256  # the cache doubles up to 256 positions, then grows by 256
GRAPH_MAX_ROWS = 256  # CUDA graphs for buckets of at most this many rows (larger ones keep the GPU busy eagerly)
GRAPH_AFTER = 8  # ... once a bucket / cache size has run this many steps


# ---------------------------------------------------------------------------
# Recipes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Variant:
    """One decode of an ensemble: greedy (temp 0) or sampled at `temp` with seed `seed` (-1: unseeded), classifier-free
    guidance `cfg_coef`, 5 s windows every `hop` s."""

    name: str
    cfg_coef: float = 1.0
    temp: float = 0.0
    seed: int = -1
    hop: float = 4.0


def _recipe_a() -> tuple[Variant, ...]:
    return (
        Variant("cfg1.25_p", 1.25),
        Variant("base_p"),
        Variant("ov2.5_p", hop=2.5),
        Variant("ov3.0_p", hop=3.0),
        Variant("ov3.5_p", hop=3.5),
        *(Variant(f"t0.5s{s}_p", temp=0.5, seed=s) for s in range(1, 5)),
        *(Variant(f"t1.0s{s}_p", temp=1.0, seed=s) for s in range(1, 5)),
    )


def _recipe_b() -> tuple[Variant, ...]:
    return (
        Variant("base_n"),
        Variant("cfg1.25_n", 1.25),
        Variant("c125ov3_n", 1.25, hop=3.0),
        Variant("c125ov25_n", 1.25, hop=2.5),
        Variant("ov3.0_n", hop=3.0),
        *(Variant(f"t0.5s{s}_n", temp=0.5, seed=s) for s in range(1, 5)),
        *(Variant(f"t1.0s{s}_n", temp=1.0, seed=s) for s in range(1, 5)),
    )


# The frozen candidates of 25 Nov 2026 (decode_vs_dettato CANDIDATES_FROZEN.txt), in vote order (first = primary):
#   A (prompted, the instruments are given): 13 decodes, cfg 1.25 hop 4; cfg 1 hop 4 / 2.5 / 3 / 3.5; temp 0.5 and temp 1.0
#     with seeds 1-4 at hop 4;
#   B (unprompted, null instrument prompt): cfg 1 hop 4; cfg 1.25 hop 4 / 3 / 2.5; cfg 1 hop 3; temp 0.5 and 1.0, seeds 1-4.
# Both: window batches of 64, vote k 4, tolerance 0.05 s, median onset and offset, dedup 0.05 s.
RECIPES: dict[str, tuple[Variant, ...]] = {"A": _recipe_a(), "B": _recipe_b()}
RECIPE_PROMPTED = {"A": True, "B": False}
VOTE_K, VOTE_TOL, VOTE_MEDIAN, VOTE_DEDUP = 4.0, 0.05, "onoff", 0.05
ENSEMBLE_BATCH = 64  # windows per call (the batch size of the stand-alone runs behind the recipes: random streams)


def parse_variants(spec: str) -> tuple[Variant, ...]:
    """A recipe name (RECIPES) or variants separated by ';', each comma-separated key=value with keys name, cfg, temp,
    seed, hop (defaults: cfg 1, temp 0, seed -1, hop 4), e.g. "cfg=1.25;hop=2.5;temp=0.5,seed=1"."""
    if spec in RECIPES:
        return RECIPES[spec]
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        kv = dict(part.split("=", 1) for part in item.split(","))
        unknown = set(kv) - {"name", "cfg", "temp", "seed", "hop"}
        if unknown:
            raise ValueError(f"variant {item!r}: unknown keys {sorted(unknown)}")
        cfg, temp = float(kv.get("cfg", 1.0)), float(kv.get("temp", 0.0))
        seed, hop = int(kv.get("seed", -1)), float(kv.get("hop", 4.0))
        name = kv.get("name") or f"cfg{cfg:g}_t{temp:g}_s{seed}_hop{hop:g}"
        out.append(Variant(name, cfg, temp, seed, hop))
    if not out:
        raise ValueError(f"no variants in {spec!r}")
    if len({v.name for v in out}) != len(out):
        raise ValueError(f"variant names must differ: {[v.name for v in out]}")
    return tuple(out)


# ---------------------------------------------------------------------------
# Batched decoding
# ---------------------------------------------------------------------------


@dataclass
class WindowBatch:
    """The conditions of n windows (one stand-alone generate() batch): audio [n, samples] at 16 kHz, the
    instrument_group / dataset_name condition strings (as ConditioningAttributes.text; None = null)."""

    wav: torch.Tensor
    instrument_group: str | None = None
    dataset_name: str | None = None

    @property
    def n(self) -> int:
        return self.wav.shape[0]

    @property
    def width(self) -> int:
        """Instrument positions of the prefix."""
        return (
            1 if self.instrument_group is None else len(self.instrument_group.split())
        )


@dataclass
class Call:
    """One generate() call to reproduce on batches[batch]; temp > 0 draws its noise from `generator`."""

    batch: int
    cfg_coef: float = 1.0
    temp: float = 0.0
    generator: torch.Generator | None = None


def call_rows(batch: WindowBatch, call: Call) -> int:
    return 2 * batch.n if call.cfg_coef != 1.0 else batch.n


def plan_waves(
    batches: list[WindowBatch], calls: list[Call], max_rows: int, max_slots: int
) -> list[list[int]]:
    """Calls -> waves (call indices). A wave holds at most max_rows rows and max_slots slots (a batch costs its windows
    once, + once more for the unconditional slots of CFG calls); a call drawing from a generator that an earlier call
    uses goes to a later wave than that call."""
    waves: list[list[int]] = []
    rows: list[int] = []
    slots: list[dict[int, int]] = []
    gen_wave: dict[int, int] = {}
    for i, call in enumerate(calls):
        b = batches[call.batch]
        need_rows = call_rows(b, call)
        need_slots = need_rows
        if need_rows > max_rows or need_slots > max_slots:
            raise ValueError(
                f"call {i}: {need_rows} rows > max_rows {max_rows} or max_slots {max_slots}"
            )
        g = id(call.generator) if call.temp > 0 and call.generator is not None else None
        w = gen_wave[g] + 1 if g is not None and g in gen_wave else 0
        while True:
            if w == len(waves):
                waves.append([])
                rows.append(0)
                slots.append({})
            have = slots[w].get(call.batch, 0)
            extra = max(need_slots, have) - have
            if (
                rows[w] + need_rows <= max_rows
                and sum(slots[w].values()) + extra <= max_slots
            ):
                break
            w += 1
        waves[w].append(i)
        rows[w] += need_rows
        slots[w][call.batch] = max(slots[w].get(call.batch, 0), need_slots)
        if g is not None:
            gen_wave[g] = w
    return [sorted(w) for w in waves if w]


def flash_available(device: torch.device, dtype: torch.dtype | None) -> bool:
    if device.type != "cuda" or dtype not in (torch.bfloat16, torch.float16):
        return False
    major, _ = torch.cuda.get_device_capability(device)
    return major >= 8 and hasattr(torch.ops.aten, "_flash_attention_forward")


def _autocast_dtype(lm) -> torch.dtype | None:
    ac = lm.autocast
    return ac.dtype if getattr(ac, "enabled", False) else None


def _conditions(batch: WindowBatch, uncond: bool) -> list[ConditioningAttributes]:
    """ConditioningAttributes of the batch's windows; uncond: every condition dropped (audio length 0 keeps the 501 mel
    positions, the null instrument rows keep the prompt width), as generate's unconditional rows."""
    out = []
    n_samples = batch.wav.shape[-1]
    for i in range(batch.n):
        length = torch.tensor([0 if uncond else n_samples], device=batch.wav.device)
        wav = WavCondition(
            wav=batch.wav[i][None, None],
            length=length,
            sample_rate=[16000],
            path=[None],
            seek_time=[0.0],
        )
        if uncond:
            text = {
                "instrument_group": " ".join(["-1"] * batch.width),
                "dataset_name": None,
            }
        else:
            text = {
                "instrument_group": batch.instrument_group,
                "dataset_name": batch.dataset_name,
            }
        out.append(ConditioningAttributes(wav={"self_wav": wav}, text=text))
    return out


@torch.no_grad()
def generate_calls(
    lm,
    batches: list[WindowBatch],
    calls: list[Call],
    max_gen_len: int = 1920,
    eos_check_every: int = 8,
    max_rows: int | None = None,
    max_slots: int | None = None,
    backend: str | None = None,
    prefill_batch: int = 64,
    stats: dict | None = None,
    verbose: bool = False,
    cuda_graphs: bool | None = None,
) -> list[list[list[int]]]:
    """See the module docstring. Returns per call, per window, the tokens up to and including the first EOS (all
    max_gen_len tokens when no EOS). Runs under lm.autocast. max_rows / max_slots: None = from the free GPU memory."""
    if not calls:
        return []
    device = lm.emb[0].weight.device
    dtype = _autocast_dtype(lm)
    backend = backend or ("flash" if flash_available(device, dtype) else "torch")
    if backend == "flash" and not flash_available(device, dtype):
        raise ValueError(
            "the flash backend needs CUDA, compute capability >= 8.0 and a bf16 / fp16 model"
        )
    if cuda_graphs is None:
        cuda_graphs = os.environ.get("MUSCRIPTOR_CUDA_GRAPHS", "0") == "1"
    stats = stats if stats is not None else {}
    for key in (
        "prefill_s",
        "steps_s",
        "steps",
        "row_steps",
        "waves",
        "splits",
        "compactions",
    ):
        stats.setdefault(key, 0)
    auto_rows, auto_slots = _capacity(lm, device, dtype, max_gen_len)
    max_rows = max_rows or auto_rows
    max_slots = max_slots or auto_slots
    max_rows = max(max_rows, max(call_rows(batches[c.batch], c) for c in calls))
    max_slots = max(max_slots, max(call_rows(batches[c.batch], c) for c in calls))
    n_orig = len(calls)
    calls, batches = list(calls), list(batches)
    key = [
        (i,) for i in range(n_orig)
    ]  # call order (a split call's halves take its place)
    children: dict[int, list[int]] = {}
    results: list = [None] * n_orig

    def replan(indices):
        rest = sorted(
            indices, key=lambda i: key[i]
        )  # in call order: a generator's calls stay in order
        return [
            [rest[j] for j in w]
            for w in plan_waves(batches, [calls[i] for i in rest], max_rows, max_slots)
        ]

    queue = plan_waves(batches, calls, max_rows, max_slots)
    while queue:
        wave = queue.pop(0)
        gens = {
            id(c.generator): c.generator
            for c in (calls[i] for i in wave)
            if c.temp > 0 and c.generator is not None
        }
        saved = {k: g.get_state() for k, g in gens.items()}
        oom = None
        try:
            out = _run_wave(
                lm,
                device,
                backend,
                batches,
                [calls[i] for i in wave],
                max_gen_len,
                eos_check_every,
                prefill_batch,
                stats,
                verbose,
                cuda_graphs,
            )
        except torch.cuda.OutOfMemoryError as e:
            oom = str(e).split("\n")[0][:160]
        if (
            oom is not None
        ):  # outside the except block: the failed wave's tensors are released with the traceback
            import gc

            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for k, g in gens.items():
                g.set_state(saved[k])
            stats["splits"] += 1
            if len(wave) > 1:  # half the calls now, half later
                half = len(wave) // 2
                queue[:0] = [wave[:half], wave[half:]]
                what = f"wave of {len(wave)} calls split in two"
            else:  # one call alone does not fit: its windows in two halves (a sampled call then draws other noise)
                i = wave[0]
                c, b = calls[i], batches[calls[i].batch]
                if b.n == 1:
                    raise torch.cuda.OutOfMemoryError(
                        f"one 5 s window does not fit in GPU memory ({oom}); free the GPU or use a larger one"
                    )
                h = b.n // 2
                children[i] = []
                for j, part in enumerate((b.wav[:h], b.wav[h:])):
                    batches.append(
                        WindowBatch(part, b.instrument_group, b.dataset_name)
                    )
                    calls.append(
                        Call(len(batches) - 1, c.cfg_coef, c.temp, c.generator)
                    )
                    key.append(key[i] + (j,))
                    children[i].append(len(calls) - 1)
                    results.append(None)
                queue = replan(children[i] + [k for w in queue for k in w])
                what = f"a call of {b.n} windows split in two"
            if verbose:
                print(f"[muscriptor] out of GPU memory, {what}: {oom}", file=sys.stderr)
            continue
        again = []
        for i, r in zip(wave, out):
            if (
                r is None
            ):  # evicted mid-wave: its generator back to the wave start, decoded again later
                again.append(i)
                c = calls[i]
                if c.temp > 0 and c.generator is not None:
                    c.generator.set_state(saved[id(c.generator)])
            else:
                results[i] = r
        stats["waves"] += 1
        if again:
            queue = replan(again + [i for w in queue for i in w])

    def result(i):
        return (
            results[i]
            if i not in children
            else result(children[i][0]) + result(children[i][1])
        )

    return [result(i) for i in range(n_orig)]


def _capacity(lm, device, dtype, max_gen_len: int) -> tuple[int, int]:
    """Rows and slots per wave from the free GPU memory. Rows finish over time and the generated-key cache only holds
    the rows still decoding (compacted before it grows), so a row is budgeted at 512 positions; a wave that does not
    fit anyway is split and run again."""
    if device.type != "cuda":
        return 256, 256
    layers = len(lm.transformer.layers)
    elem = torch.empty((), dtype=dtype or torch.float32).element_size()
    per_pos = layers * 2 * lm.dim * elem
    free, _ = torch.cuda.mem_get_info(device)
    free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(
        device
    )  # cached by PyTorch: reusable
    total = torch.cuda.get_device_properties(device).total_memory
    budget = max(
        free - max(6e9, 0.15 * total), 1e9
    )  # margin: activations, prefill, allocator slack
    slot_bytes = 520 * per_pos
    row_bytes = min(max_gen_len, 512) * per_pos
    slots = int(min(512, budget * 0.35 // slot_bytes))
    rows = int(min(1536, (budget - slots * slot_bytes) // row_bytes))
    return max(rows, 16), max(slots, 16)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


BUCKETS = (
    8,
    16,
    32,
    64,
    96,
    128,
    192,
    256,
    320,
    384,
    448,
    512,
    640,
    768,
    896,
    1024,
    1280,
    1536,
    1792,
    2048,
    2560,
    3072,
    4096,
)


def bucket(n: int) -> int:
    for b in BUCKETS:
        if b >= n:
            return b
    return n


class _RowState:
    """Static per-position buffers of a wave (positions 0..R-1 decode, R..Rb-1 are inactive padding)."""

    def __init__(self, size: int, n_slots: int, card: int, device):
        z = dict(device=device)
        self.cls_in = torch.zeros(size, dtype=torch.long, **z)
        self.done = torch.ones(size, dtype=torch.bool, **z)
        self.slot = torch.zeros(size, dtype=torch.long, **z)
        self.plen = torch.ones(size, dtype=torch.long, **z)
        self.cfg = torch.ones(size, **z)
        self.temp = torch.ones(size, **z)
        self.sampled = torch.zeros(size, dtype=torch.bool, **z)
        self.uncond = torch.zeros(size, dtype=torch.bool, **z)
        self.has_cfg = torch.zeros(size, dtype=torch.bool, **z)
        self.twin = torch.arange(size, **z)
        self.gid = torch.zeros(size, dtype=torch.long, **z)
        self.noise_src = torch.zeros(size, dtype=torch.long, **z)
        self.lse_index = torch.zeros(
            size, 1, dtype=torch.long, **z
        )  # [size, H] once the heads are known
        self.live = torch.zeros(
            size, dtype=torch.long, **z
        )  # 1: decoding; 0: done or padding (reads no generated keys)
        self.cu_q = torch.zeros(n_slots + 1, dtype=torch.int32, **z)
        self.seq_t = torch.zeros(size, dtype=torch.int32, **z)
        self.t = torch.zeros(1, dtype=torch.long, **z)
        self.max_q = 1
        self.rb = size


def _run_wave(
    lm,
    device,
    backend: str,
    batches: list[WindowBatch],
    calls: list[Call],
    max_gen_len: int,
    eos_check_every: int,
    prefill_batch: int,
    stats: dict,
    verbose: bool,
    cuda_graphs: bool,
) -> list:
    check_every = max(1, int(eos_check_every))
    layers = lm.transformer.layers
    H = layers[0].self_attn.num_heads
    D = lm.dim
    Dh = D // H
    card = lm.card
    t0 = time.perf_counter()

    # ---- slots: prefill each window batch (cond, + uncond when a CFG call uses it) ----
    used = list(dict.fromkeys(c.batch for c in calls))
    cfg_batches = {c.batch for c in calls if c.cfg_coef != 1.0}
    slot_specs = []  # (batch, uncond)
    for b in used:
        slot_specs.append((b, False))
        if b in cfg_batches:
            slot_specs.append((b, True))
    slot_base: dict[tuple[int, bool], int] = {}
    n_slots = 0
    for b, unc in slot_specs:
        slot_base[(b, unc)] = n_slots
        n_slots += batches[b].n
    p_max = max(
        batches[b].wav.shape[-1] // 160 + 1 + 1 + batches[b].width + 1 for b in used
    )
    kv_dtype = None
    slot_k: list[torch.Tensor] = []
    slot_v: list[torch.Tensor] = []
    slot_logits = torch.empty(n_slots, card, device=device)
    slot_plen = torch.empty(n_slots, dtype=torch.int32)
    # Prefill in groups of windows with the same prompt width (a batch never mixes widths: no extra padding positions).
    by_width: dict[int, list] = {}
    for b, unc in slot_specs:
        for j in range(batches[b].n):
            by_width.setdefault(batches[b].width, []).append(
                (slot_base[(b, unc)] + j, b, j, unc)
            )
    for items in by_width.values():
        for start in range(0, len(items), prefill_batch):
            part = items[start : start + prefill_batch]
            conds = []
            for s, b, j, unc in part:
                one = WindowBatch(
                    batches[b].wav[j : j + 1],
                    batches[b].instrument_group,
                    batches[b].dataset_name,
                )
                conds += _conditions(one, unc)
            k_list, v_list, logits, plen = _prefill(lm, conds)
            if not slot_k:
                kv_dtype = k_list[0].dtype
                for _ in layers:
                    slot_k.append(
                        torch.zeros(
                            n_slots, p_max, H, Dh, device=device, dtype=kv_dtype
                        )
                    )
                    slot_v.append(
                        torch.zeros(
                            n_slots, p_max, H, Dh, device=device, dtype=kv_dtype
                        )
                    )
            idx = torch.tensor([s for s, *_ in part], device=device)
            for i in range(len(layers)):
                slot_k[i][idx, :plen] = k_list[i]
                slot_v[i][idx, :plen] = v_list[i]
            slot_logits[idx] = logits
            slot_plen[idx.cpu()] = plen
            del k_list, v_list
    _sync(device)
    stats["prefill_s"] += time.perf_counter() - t0

    # ---- rows: host attributes per row id; main rows get output rows 0..N-1 in call order ----
    r_slot, r_cfg, r_temp, r_sampled, r_uncond, r_twin, r_call, r_out = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    call_main: list[list[int]] = []
    n_out = 0
    for ci, call in enumerate(calls):
        b = batches[call.batch]
        mains = []
        for j in range(b.n):
            rid = len(r_slot)
            mains.append(rid)
            r_slot.append(slot_base[(call.batch, False)] + j)
            r_cfg.append(float(call.cfg_coef))
            r_temp.append(float(call.temp) if call.temp > 0 else 1.0)
            r_sampled.append(call.temp > 0)
            r_uncond.append(False)
            r_twin.append(rid)
            r_call.append(ci)
            r_out.append(n_out)
            n_out += 1
        if call.cfg_coef != 1.0:
            for j in range(b.n):
                rid = len(r_slot)
                r_slot.append(slot_base[(call.batch, True)] + j)
                r_cfg.append(float(call.cfg_coef))
                r_temp.append(1.0)
                r_sampled.append(False)
                r_uncond.append(True)
                r_twin.append(mains[j])
                r_call.append(ci)
                r_out.append(-1)
                r_twin[mains[j]] = rid
        call_main.append(mains)
    n_rows = len(r_slot)
    # rows ordered by slot (the slot attention takes each slot's rows as consecutive queries)
    order = sorted(range(n_rows), key=lambda r: (r_slot[r], r))

    # sampled calls: a noise block of n rows each, drawn by the host before every step
    noise_off: dict[int, int] = {}
    off = 0
    for ci, call in enumerate(calls):
        if call.temp > 0:
            noise_off[ci] = off
            off += len(call_main[ci])
    noise = torch.ones(max(off, 1), card, device=device)
    noise_row = {
        r: noise_off[r_call[r]] + call_main[r_call[r]].index(r)
        for r in range(n_rows)
        if r_sampled[r]
    }
    blocks = {ci: noise[o : o + len(call_main[ci])] for ci, o in noise_off.items()}
    gens = {
        ci: (
            calls[ci].generator
            if calls[ci].generator is not None
            else _default_generator(device)
        )
        for ci in noise_off
    }

    out = torch.full((n_out + 1, max_gen_len), EOS, dtype=torch.long, device=device)
    first_eos = torch.full((n_out + 1,), max_gen_len, dtype=torch.long, device=device)
    plen_host = slot_plen.tolist()
    st = _RowState(bucket(n_rows), n_slots, card, device)
    st.lse_index = torch.zeros(st.cls_in.shape[0], H, dtype=torch.long, device=device)
    st.max_q = max(Counter(r_slot).values())
    weights = _Weights(lm, kv_dtype)
    cpu = torch.device("cpu")

    def upload(order):
        """Host attributes of the positions in `order` (+ inactive padding up to the bucket) to the device."""
        n = len(order)
        rb = bucket(max(n, 1))
        pos_of = {r: p for p, r in enumerate(order)}
        pad = rb - n

        def put(buf, vals, dt, fill):
            host = torch.tensor(vals + [fill] * pad, dtype=dt, device=cpu)
            buf[:rb].copy_(host, non_blocking=False)

        put(st.slot, [r_slot[r] for r in order], torch.long, 0)
        put(st.plen, [plen_host[r_slot[r]] for r in order], torch.long, 1)
        put(st.cfg, [r_cfg[r] for r in order], torch.float32, 1.0)
        put(st.temp, [r_temp[r] for r in order], torch.float32, 1.0)
        put(st.sampled, [r_sampled[r] for r in order], torch.bool, False)
        put(st.uncond, [r_uncond[r] for r in order], torch.bool, False)
        put(
            st.has_cfg,
            [r_cfg[r] != 1.0 and not r_uncond[r] for r in order],
            torch.bool,
            False,
        )
        st.twin[:rb].copy_(
            torch.tensor(
                [pos_of.get(r_twin[r], p) for p, r in enumerate(order)]
                + list(range(n, rb)),
                dtype=torch.long,
            )
        )
        put(
            st.gid,
            [r_out[r] if r_out[r] >= 0 else n_out for r in order],
            torch.long,
            n_out,
        )
        put(st.noise_src, [noise_row.get(r, 0) for r in order], torch.long, 0)
        counts = [0] * n_slots
        within = []
        for r in order:
            within.append(counts[r_slot[r]])
            counts[r_slot[r]] += 1
        cu = [0]
        for c in counts:
            cu.append(cu[-1] + c)
        st.cu_q.copy_(torch.tensor(cu, dtype=torch.int32))
        flat = torch.tensor(
            [r_slot[r] * H * st.max_q + w for r, w in zip(order, within)] + [0] * pad,
            dtype=torch.long,
        )
        st.lse_index[:rb].copy_(flat[:, None] + torch.arange(H)[None] * st.max_q)
        st.live[:rb].copy_(torch.tensor([1] * n + [0] * pad, dtype=torch.long))
        st.rb = rb

    upload(order)
    R = len(order)
    st.done[: st.rb] = False
    st.done[R : st.rb] = True
    cu_k = (torch.arange(n_slots + 1, dtype=torch.int32) * p_max).to(device)
    seqused_slot = slot_plen.to(device)
    t_alloc = CHUNK
    gen_k = [
        torch.empty(st.rb, t_alloc, H, Dh, device=device, dtype=kv_dtype)
        for _ in layers
    ]
    gen_v = [
        torch.empty(st.rb, t_alloc, H, Dh, device=device, dtype=kv_dtype)
        for _ in layers
    ]
    finished = [False] * len(calls)
    base: dict[int, tuple[int, torch.Tensor]] = {}
    scale = 1.0 / math.sqrt(Dh)
    graphs: dict[tuple[int, int], tuple] = {}
    pool = None
    use_graphs = cuda_graphs and device.type == "cuda" and backend == "flash"

    def draw(t):
        for ci, block in blocks.items():
            if finished[ci]:
                continue
            if ci not in base:
                base[ci] = (t, gens[ci].get_state())
            block.exponential_(1, generator=gens[ci])

    def choose(logits: torch.Tensor, t: int):
        """float logits [rb, card] of step t -> CFG, next classes, outputs and flags (positions 0..rb-1)."""
        rb = st.rb
        twin, done = st.twin[:rb], st.done[:rb]
        unc = logits.index_select(0, twin)
        logits = torch.where(
            st.has_cfg[:rb, None], unc + (logits - unc) * st.cfg[:rb, None], logits
        )
        if card > VOCAB_SIZE:
            logits[:, VOCAB_SIZE:] = -math.inf  # ids the tokenizer never produces
        probs = torch.softmax(logits / st.temp[:rb, None], dim=-1)
        q = noise.index_select(0, st.noise_src[:rb])
        next_class = torch.where(
            st.sampled[:rb], torch.div(probs, q).argmax(dim=-1), logits.argmax(dim=-1)
        )
        next_token = lm.token_of_class[next_class]
        gid = st.gid[:rb]
        out.view(-1).index_copy_(
            0, gid * max_gen_len + t, torch.where(done, EOS, next_token)
        )
        is_eos = next_token == EOS
        first_eos.index_copy_(
            0, gid, torch.where(~done & is_eos, t, first_eos.index_select(0, gid))
        )
        new_done = done | is_eos
        uncond = st.uncond[:rb]
        done.copy_(torch.where(uncond, new_done.index_select(0, twin), new_done))
        st.cls_in[:rb].copy_(
            torch.where(uncond, next_class.index_select(0, twin), next_class)
        )

    def forward(t_host: int = 0):
        return _step(
            weights,
            st,
            gen_k,
            gen_v,
            slot_k,
            slot_v,
            cu_k,
            seqused_slot,
            p_max,
            backend,
            scale,
            t_host,
        )

    eager_steps: dict[tuple[int, int], int] = {}

    def run_step(t: int):
        """The step, eager, or replayed as a CUDA graph for small buckets (where launching ~700 kernels costs more
        than running them) once the bucket and cache size have been stable for GRAPH_AFTER steps."""
        nonlocal pool
        key = (st.rb, t_alloc)
        if not use_graphs or st.rb > GRAPH_MAX_ROWS:
            return forward(t)
        if key not in graphs:
            eager_steps[key] = eager_steps.get(key, 0) + 1
            if eager_steps[key] <= GRAPH_AFTER:
                return forward(t)
            tc = time.perf_counter()
            side = torch.cuda.Stream(device)
            side.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(
                side
            ):  # warm-up (writes the step's keys / values, as the real run will)
                forward()
            torch.cuda.current_stream(device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                logits_out = forward()
            if pool is None:
                pool = graph.pool()
            graphs[key] = (graph, logits_out)
            stats["captures"] = stats.get("captures", 0) + 1
            stats["capture_s"] = stats.get("capture_s", 0.0) + time.perf_counter() - tc
        graph, logits_out = graphs[key]
        graph.replay()
        return logits_out

    # ---- step 0 on the prefill logits ----
    t0 = time.perf_counter()
    draw(0)
    logits0 = torch.zeros(st.rb, card, device=device)
    logits0[:R] = slot_logits.index_select(0, st.slot[:R])
    choose(logits0, 0)
    del slot_logits, logits0
    stats["steps"] += 1
    stats["row_steps"] += R

    removed = [False] * n_rows

    def check(t) -> bool:
        """Host check after step t: finish calls, rewind their generators, freeze the finished rows, compact the rows
        when the bucket shrinks. True: every call done."""
        done_list = st.done[:R].tolist()
        fe = first_eos.tolist()
        done_row = {r: d for r, d in zip(order, done_list)}
        for ci in range(len(calls)):
            if finished[ci]:
                continue
            if all(removed[r] or done_row[r] for r in call_main[ci]):
                end = max(fe[r_out[r]] for r in call_main[ci]) + 1
                finished[ci] = True
                if (
                    ci in blocks and end <= t
                ):  # generator back to its state after step end - 1's draw
                    t_base, state = base[ci]
                    gens[ci].set_state(state)
                    for _ in range(end - t_base):
                        blocks[ci].exponential_(1, generator=gens[ci])
            base.pop(ci, None)
        keep_pos = [p for p, r in enumerate(order) if not done_row[r]]
        if not keep_pos:
            return True
        st.live[:R].copy_(
            (~st.done[:R]).long()
        )  # done rows stop reading their generated keys
        if len(keep_pos) < R and (
            bucket(len(keep_pos)) < st.rb or len(keep_pos) <= 0.75 * R
        ):
            compact(keep_pos, done_row)
        return all(finished)

    def compact(keep_pos, done_row):
        """Keep the positions keep_pos (in order): row state and generated keys / values moved down, bucket shrunk."""
        nonlocal order, R
        for r in order:
            if done_row[r]:
                removed[r] = True
        nonlocal pool
        n = len(keep_pos)
        idx = torch.tensor(keep_pos, device=device)
        rb_new = bucket(max(n, 1))
        graphs.clear()  # they read the caches replaced below
        pool = None
        for lst in (
            gen_k,
            gen_v,
        ):  # new caches of the new bucket size: the memory of the finished rows is freed
            for i in range(len(lst)):
                old = lst[i]
                new = torch.empty(
                    rb_new, old.shape[1], H, Dh, device=device, dtype=old.dtype
                )
                new[:n] = old.index_select(0, idx)
                lst[i] = new
                del old
        st.cls_in[:n] = st.cls_in.index_select(0, idx)
        order = [order[p] for p in keep_pos]
        R = n
        upload(order)
        st.done[: st.rb] = False
        st.done[R : st.rb] = True
        stats["compactions"] += 1
        if device.type == "cuda":
            torch.cuda.empty_cache()

    evicted: list[int] = []

    def grow(t_new: int):
        """Every layer's cache to t_new positions (layer by layer: an out-of-memory error leaves the grown layers
        grown, and the retry grows the others)."""
        nonlocal pool
        graphs.clear()
        pool = None  # the graphs that shared it are gone
        for lst in (gen_k, gen_v):
            for i in range(len(lst)):
                old = lst[i]
                if old.shape[1] >= t_new:
                    continue
                new = torch.empty(
                    old.shape[0], t_new, H, Dh, device=device, dtype=old.dtype
                )
                new[:, : old.shape[1]] = old
                lst[i] = new
                del old, new
        if device.type == "cuda":
            torch.cuda.empty_cache()  # the old, smaller caches: give them back, or the next size fragments memory

    def evict() -> bool:
        """Out of memory mid-wave: the last unfinished calls leave the wave (their rows are dropped; generate_calls
        decodes them again in a later wave), until at most 60 % of the decoding rows remain. False: nothing to evict."""
        nonlocal pool
        graphs.clear()
        pool = None
        import gc

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        unfinished = [ci for ci in range(len(calls)) if not finished[ci]]
        if len(unfinished) <= 1:
            return False
        done_list = st.done[:R].tolist()
        live_of = Counter(r_call[r] for r, d in zip(order, done_list) if not d)
        live = sum(live_of.values())
        out_now = []
        for ci in reversed(unfinished[1:]):
            if live <= 0.6 * sum(live_of.values()):
                break
            out_now.append(ci)
            live -= live_of.get(ci, 0)
        for ci in out_now:
            finished[ci] = True
            base.pop(ci, None)
        evicted.extend(out_now)
        gone = set(out_now)
        done_row = {r: (d or r_call[r] in gone) for r, d in zip(order, done_list)}
        compact([p for p, r in enumerate(order) if not done_row[r]], done_row)
        stats["evicted"] = stats.get("evicted", 0) + len(out_now)
        if verbose:
            print(
                f"[muscriptor] out of GPU memory at step {t}: {len(out_now)} of {len(calls)} calls leave the wave "
                f"(decoded again later)",
                file=sys.stderr,
            )
        return True

    all_done = check(0) if (check_every == 1 or max_gen_len == 1) else False
    t = 1
    while not all_done and t < max_gen_len:
        oom = False
        try:
            if (
                t - 1 >= t_alloc
            ):  # grow the generated-key cache (graphs are captured again for the new size)
                done_list = st.done[:R].tolist()
                if any(done_list) and not all(
                    done_list
                ):  # drop the finished rows first: less to copy and to hold
                    done_row = {r: d for r, d in zip(order, done_list)}
                    compact(
                        [p for p, r in enumerate(order) if not done_row[r]], done_row
                    )
                t_new = min(t_alloc + min(t_alloc, GROW_MAX), max_gen_len)
                grow(t_new)
                t_alloc = t_new
            st.t.fill_(t)
            logits = run_step(t)
        except torch.cuda.OutOfMemoryError:
            oom = True
        if (
            oom
        ):  # outside the except block (its traceback holds the failed step's tensors)
            logits = None
            if not evict():
                raise torch.cuda.OutOfMemoryError(
                    "out of GPU memory with a single call left in the wave"
                )
            continue  # step t again (no noise was drawn for it yet)
        draw(
            t
        )  # after the forward pass: a step retried after an out-of-memory error draws once
        choose(logits, t)
        stats["steps"] += 1
        stats["row_steps"] += R
        if (t + 1) % check_every == 0 or t == max_gen_len - 1:
            all_done = check(t)
        t += 1
    _sync(device)
    stats["steps_s"] += time.perf_counter() - t0
    if verbose:
        print(
            f"[muscriptor] wave: {len(calls)} calls, {n_rows} rows, {n_slots} slots, {t} steps, "
            f"{time.perf_counter() - t0:.1f} s",
            file=sys.stderr,
        )
    out_cpu = out.cpu()
    first_cpu = first_eos.cpu()
    results = []
    for ci in range(len(calls)):
        if ci in evicted:
            results.append(None)
            continue
        rows = []
        for r in call_main[ci]:
            g = r_out[r]
            n = (
                int(first_cpu[g]) + 1
                if int(first_cpu[g]) < max_gen_len
                else max_gen_len
            )
            rows.append(out_cpu[g, :n].tolist())
        results.append(rows)
    graphs.clear()
    for lst in (slot_k, slot_v, gen_k, gen_v):  # free the caches before the next wave
        lst.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return results


def _default_generator(device):
    if device.type == "cuda":
        return torch.cuda.default_generators[
            device.index if device.index is not None else torch.cuda.current_device()
        ]
    return torch.default_generator


def _prefill(lm, conds: list[ConditioningAttributes]):
    """Prefix + BOS of each window, as generate's first step: per-layer keys / values [B, P + 1, H, Dh], the BOS
    logits [B, card] (float32) and P + 1."""
    with lm.autocast if lm.autocast_conditions else _null():
        prepared = lm.condition_provider.tokenize(conds)
        cond_tensors = lm.condition_provider(prepared)
    B = len(conds)
    plen = sum(c.shape[1] for c, _ in cond_tensors.values()) + 1
    state = init_states(lm, batch_size=B, sequence_length=plen)
    bos = torch.full(
        (B, lm.n_q, 1),
        lm.initial_token_id,
        device=lm.emb[0].weight.device,
        dtype=torch.long,
    )
    with lm.autocast:
        logits = lm(bos, cond_tensors, first_step=True, model_state=state)[
            :, 0, -1
        ].float()
    ks, vs = [], []
    for layer in lm.transformer.layers:
        cache = state[layer.self_attn._module_absolute_name][
            "cache"
        ]  # [2, B, plen, H, Dh]
        ks.append(cache[0, :, :plen])
        vs.append(cache[1, :, :plen])
    return ks, vs, logits, plen


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Weights:
    """The step's weights in the compute dtype (what autocast casts them to), cast once per wave."""

    def __init__(self, lm, dtype: torch.dtype):
        self.dtype = dtype
        self.dim = lm.dim
        self.max_period = lm.transformer.max_period

        def low(p):
            return p.detach().to(dtype)

        self.layers = [
            dict(
                n1=(layer.norm1.weight.detach(), layer.norm1.bias.detach()),
                win=low(layer.self_attn.in_proj_weight),
                wout=low(layer.self_attn.out_proj.weight),
                n2=(layer.norm2.weight.detach(), layer.norm2.bias.detach()),
                w1=low(layer.linear1.weight),
                w2=low(layer.linear2.weight),
                heads=layer.self_attn.num_heads,
            )
            for layer in lm.transformer.layers
        ]
        self.out_norm = (lm.out_norm.weight.detach(), lm.out_norm.bias.detach())
        self.wlin = low(lm.linears[0].weight)
        self.emb = lm.emb[0].weight.detach()


def _step(
    w: _Weights,
    st: "_RowState",
    gen_k,
    gen_v,
    slot_k,
    slot_v,
    cu_k,
    seqused_slot,
    p_max: int,
    backend: str,
    scale: float,
    t_host: int = 0,
) -> torch.Tensor:
    """One decoding step of positions 0..st.rb-1 (graph-safe: static buffers, the step index in st.t): input classes
    st.cls_in at positions plen - 1 + t -> float logits [rb, card]. Writes each row's key / value at generated
    position t - 1. The operations of LMModel's layers under autocast (float32 residual and norms, matmuls in the
    compute dtype)."""
    rb = st.rb
    t = st.t
    D, dt = w.dim, w.dtype
    x = F.embedding(st.cls_in[:rb], w.emb)  # [rb, D] float32
    positions = (st.plen[:rb] - 1 + t).view(-1, 1, 1)
    x = (
        x
        + create_sin_embedding(positions, D, max_period=w.max_period, dtype=x.dtype)[
            :, 0
        ]
    )
    write_at = t - 1
    if backend == "flash":
        st.seq_t[:rb].copy_(st.live[:rb] * t)
        T = gen_k[0].shape[1]
        gen_cu_q = torch.arange(rb + 1, dtype=torch.int32, device=x.device)
        gen_cu_k = gen_cu_q * T
    for i, L in enumerate(w.layers):
        H = L["heads"]
        h = F.layer_norm(x, (D,), *L["n1"], eps=1e-5).to(dt)
        q, k, v = F.linear(h, L["win"]).view(rb, 3, H, -1).unbind(1)  # [rb, H, Dh]
        gk, gv = gen_k[i][:rb], gen_v[i][:rb]
        gk.index_copy_(1, write_at, k[:, None].to(gk.dtype))
        gv.index_copy_(1, write_at, v[:, None].to(gv.dtype))
        if backend == "flash":
            o = _flash_attention(
                q,
                slot_k[i],
                slot_v[i],
                gk,
                gv,
                st,
                cu_k,
                seqused_slot,
                p_max,
                scale,
                gen_cu_q,
                gen_cu_k,
            )
        else:
            o = _torch_attention(q, slot_k[i], slot_v[i], gk, gv, st, t_host)
        x = x + F.linear(o.reshape(rb, D).to(dt), L["wout"])
        h = F.layer_norm(x, (D,), *L["n2"], eps=1e-5).to(dt)
        x = x + F.linear(F.gelu(F.linear(h, L["w1"])), L["w2"])
    x = F.layer_norm(x, (D,), *w.out_norm, eps=1e-5).to(dt)
    return F.linear(x, w.wlin).float()


def _flash_attention(
    q, sk, sv, gk, gv, st, cu_k, seqused_slot, p_max, scale, gen_cu_q, gen_cu_k
):
    """Slot part: one varlen flash call (the rows of a slot are one sequence's queries, its keys the slot's valid
    positions; padding positions past cu_q[-1] are not computed); generated part: each row over its own st.seq_t
    keys (0 for finished rows); merged by their log-sum-exps: o = o2 + sigmoid(lse1 - lse2) * (o1 - o2), float32."""
    rb, H, Dh = q.shape
    qc = q.contiguous()
    o1, lse1, *_ = torch.ops.aten._flash_attention_forward(
        qc,
        sk.view(-1, H, Dh),
        sv.view(-1, H, Dh),
        st.cu_q,
        cu_k,
        st.max_q,
        p_max,
        0.0,
        False,
        False,
        scale=scale,
        seqused_k=seqused_slot,
    )
    l1 = lse1.reshape(-1)[st.lse_index[:rb]]  # [rb, H]
    o2, lse2, *_ = torch.ops.aten._flash_attention_forward(
        qc,
        gk.reshape(-1, H, Dh),
        gv.reshape(-1, H, Dh),
        gen_cu_q,
        gen_cu_k,
        1,
        gk.shape[1],
        0.0,
        False,
        False,
        scale=scale,
        seqused_k=st.seq_t[:rb],
    )
    a = torch.sigmoid(l1 - lse2.reshape(rb, H))[..., None]
    o2 = o2.float()
    return torch.addcmul(o2, a, o1[:rb].float() - o2)


def _torch_attention(q, sk, sv, gk, gv, st, t: int):
    """The same attention with scaled_dot_product_attention over each row's gathered slot keys + generated keys."""
    rb = q.shape[0]
    ks = sk.index_select(0, st.slot[:rb])  # [rb, P, H, Dh]
    vs = sv.index_select(0, st.slot[:rb])
    P = ks.shape[1]
    k = (
        torch.cat([ks, gk[:, :t]], dim=1).transpose(1, 2).to(q.dtype)
    )  # [rb, H, P + t, Dh]
    v = torch.cat([vs, gv[:, :t]], dim=1).transpose(1, 2).to(q.dtype)
    valid = torch.cat(
        [
            torch.arange(P, device=q.device)[None] < st.plen[:rb, None],
            torch.ones(rb, t, dtype=torch.bool, device=q.device),
        ],
        dim=1,
    )
    o = F.scaled_dot_product_attention(
        q[:, :, None], k, v, attn_mask=valid[:, None, None, :]
    )  # [rb, H, 1, Dh]
    return o[:, :, 0]
