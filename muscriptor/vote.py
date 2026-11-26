"""Consensus vote over several decodes of one song (training repo: amt/vote.py, itself tools/vote.py on note lists).

The notes of each decode are turned into the string rows its CSV would hold (is_drum, program, onset, offset, pitch as
csv.writer writes them) and voted exactly as the training repository votes them, so the voted notes are the same:

  rows = vote([note_rows(notes_d) for each decode d], k=4, tol=0.05, median="onoff", dedup=0.05)
  notes = vote_notes(rows)

Clusters: notes of one (is_drum, pitch) with onsets within `tol` of the cluster's first note, at most one note per decode.
A cluster seen in `k` or more decodes is kept as its earliest decode's note (decode order = the list order, first =
primary), with the median onset (and offset, median "onoff"); dedup > 0 then keeps one of the kept notes of one
(is_drum, program, pitch) closer than `dedup` s (the most supported; ties: the earlier).
"""

from collections import defaultdict

from muscriptor.tokenizer.notes import Note

VOTE_COLUMNS = ["is_drum", "program", "onset", "offset", "pitch", "conf"]


def note_rows(
    notes: list[Note], confidences: list | None = None
) -> list[dict[str, str]]:
    """A decode's notes as the string rows of its CSV (training.evaluate.write_notes_csv, read back by csv.DictReader)."""
    rows = []
    for i, n in enumerate(notes):
        r = {
            "is_drum": str(n.is_drum),
            "program": str(n.program),
            "onset": str(n.onset),
            "offset": str(n.offset),
            "pitch": str(n.pitch),
        }
        if confidences:
            c, ca = confidences[i]
            r["conf"], r["conf_any"] = f"{c:.4g}", f"{ca:.4g}"
        rows.append(r)
    return rows


def vote(
    decs: list[list[dict[str, str]]],
    k: float,
    tol: float = 0.05,
    median: str = "",
    dedup: float = 0.0,
    weight: str = "",
) -> list[dict[str, str]]:
    """The training repository's vote on the decodes' rows (first = primary). Returns the output rows (VOTE_COLUMNS,
    strings as tools/vote.py writes them)."""
    by_key = defaultdict(list)  # (is_drum, pitch) -> [(onset, decode, row)]
    for i, rows in enumerate(decs):
        for r in rows:
            by_key[(r["is_drum"], r["pitch"])].append((float(r["onset"]), i, r))
    kept = []
    for key, items in by_key.items():
        items.sort(key=lambda x: (x[0], x[1]))
        used = [False] * len(items)
        for a in range(len(items)):
            if used[a]:
                continue
            used[a] = True
            cluster = {items[a][1]: items[a]}
            t0 = items[a][0]
            for b in range(a + 1, len(items)):
                if items[b][0] - t0 > tol:
                    break
                if not used[b] and items[b][1] not in cluster:
                    cluster[items[b][1]] = items[b]
                    used[b] = True
            if weight == "conf":

                def cf(x):
                    try:
                        v = float(x[2].get("conf") or "nan")
                    except ValueError:
                        v = float("nan")
                    return 1.0 if v != v else v

                support = sum(cf(x) for x in cluster.values())
            else:
                support = len(cluster)
            if support >= k - 1e-9:
                r = dict(cluster[min(cluster)][2])
                if median:
                    ons = sorted(float(x[2]["onset"]) for x in cluster.values())
                    offs = sorted(float(x[2]["offset"]) for x in cluster.values())

                    def med(v):
                        return (
                            v[len(v) // 2]
                            if len(v) % 2
                            else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2
                        )

                    r["onset"] = f"{med(ons):.4f}"
                    if median == "onoff":
                        r["offset"] = f"{max(med(offs), med(ons) + 0.011):.4f}"
                    elif float(r["offset"]) <= float(r["onset"]):
                        r["offset"] = f"{float(r['onset']) + 0.011:.4f}"
                r["_n"] = support
                kept.append(r)
    kept.sort(
        key=lambda r: (
            float(r["onset"]),
            r["is_drum"],
            int(r["program"]),
            int(r["pitch"]),
        )
    )
    if dedup > 0:
        last, out = {}, []
        for r in kept:
            key = (r["is_drum"], r["program"], r["pitch"])
            j = last.get(key)
            if (
                j is not None
                and float(r["onset"]) - float(out[j]["onset"]) < dedup
                and out[j] is not None
            ):
                if r["_n"] > out[j]["_n"]:
                    out[j] = r
                continue
            last[key] = len(out)
            out.append(r)
        kept = out
    return [{c: str(r.get(c, "")) for c in VOTE_COLUMNS} for r in kept]


def vote_notes(rows: list[dict[str, str]]) -> list[Note]:
    """Voted rows -> notes."""
    return [
        Note(
            r["is_drum"] == "True",
            int(r["program"]),
            float(r["onset"]),
            float(r["offset"]),
            int(r["pitch"]),
        )
        for r in rows
    ]
