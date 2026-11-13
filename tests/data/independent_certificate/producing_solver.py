#!/usr/bin/env python3
"""Independently executed producing implementation for the minimum biclique
cover task.

This is a second, structurally independent implementation: the rectangle
system is recovered by grouping columns into classes of equal row sets
(frozensets) and reducing the classes by inclusion, the packing witness is
derived with a queue-style owner scan, and both sides are re-checked against
the raw 1-entries with set arithmetic before the answer is written.  It also
emits `optimum.csv`, the independent minimum table the verifier compares
candidates against.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def read_input(path):
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def column_classes(cols, ones):
    """profile (frozenset of rows) -> columns with that row set."""
    by_column = [set() for _ in range(cols)]
    for u, v in ones:
        by_column[v].add(u)
    classes = {}
    for v, members in enumerate(by_column):
        if members:
            classes.setdefault(frozenset(members), []).append(v)
    return classes, by_column


def reduce_classes(classes):
    """Keep the classes that contain no smaller class."""
    kept = []
    for profile in sorted(classes, key=len):
        if any(smaller <= profile for smaller in kept):
            continue
        kept.append(profile)
    return kept


def build_rectangles(kept, by_column, cols):
    rectangles = []
    corners = []
    for profile in kept:
        row_list = sorted(profile)
        col_list = [v for v in range(cols) if profile <= frozenset(by_column[v])]
        rectangles.append((row_list, col_list))
        corner = None
        for v in range(cols):
            if frozenset(by_column[v]) == profile:
                corner = v
                break
        corners.append(corner)
    return rectangles, corners


def witness_by_owners(rectangles, row_count):
    owners = [set() for _ in range(row_count)]
    for index, (row_list, _cols) in enumerate(rectangles):
        for u in row_list:
            owners[u].add(index)
    remaining = set(range(len(rectangles)))
    used = [False] * row_count
    picks = {}
    progress = True
    while remaining and progress:
        progress = False
        for u in range(row_count):
            if used[u]:
                continue
            live = owners[u] & remaining
            if len(live) == 1:
                target = next(iter(live))
                picks[target] = u
                used[u] = True
                remaining.discard(target)
                progress = True
                break
    return picks if not remaining else None


def recheck(rows, cols, rectangles, entries, ones):
    ones_set = set(map(tuple, ones))
    covered = set()
    for row_list, col_list in rectangles:
        for u in row_list:
            for v in col_list:
                if (u, v) not in ones_set:
                    return False
                covered.add((u, v))
    if covered != ones_set:
        return False
    if len(entries) != len(rectangles):
        return False
    seen = set()
    for u, v in entries:
        if (u, v) in seen or (u, v) not in ones_set:
            return False
        seen.add((u, v))
    for i in range(len(entries)):
        u1, v1 = entries[i]
        for j in range(i + 1, len(entries)):
            u2, v2 = entries[j]
            if (u1, v2) in ones_set and (u2, v1) in ones_set:
                return False
    return True


def fallback_entries(rectangles, ones_set, attempts=4000, seed=99):
    rng = random.Random(seed)
    for _ in range(attempts):
        entries = [(rng.choice(row_list), rng.choice(col_list)) for row_list, col_list in rectangles]
        ok = True
        for i in range(len(entries)):
            u1, v1 = entries[i]
            for j in range(i + 1, len(entries)):
                u2, v2 = entries[j]
                if (u1, v2) in ones_set and (u2, v1) in ones_set:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return entries
    return None


def solve(instance):
    rows = int(instance["rows"])
    cols = int(instance["cols"])
    ones = [(int(u), int(v)) for u, v in instance["ones"]]
    classes, by_column = column_classes(cols, ones)
    kept = reduce_classes(classes)
    rectangles, corners = build_rectangles(kept, by_column, cols)
    picks = witness_by_owners(rectangles, rows)
    entries = None
    if picks is not None and all(corner is not None for corner in corners):
        entries = [(picks[index], corners[index]) for index in range(len(rectangles))]
    ones_set = set(map(tuple, ones))
    if entries is None:
        entries = fallback_entries(rectangles, ones_set)
    if entries is None or not recheck(rows, cols, rectangles, entries, ones):
        raise RuntimeError("independent producer could not certify an instance")
    answer = {
        "rectangles": [[list(row_list), list(col_list)] for row_list, col_list in rectangles],
        "packing": [[u, v] for u, v in entries],
    }
    return answer


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: producing_solver.py INPUT_JSON OUTPUT_DIRECTORY")
    document = read_input(sys.argv[1])
    answers = []
    sizes = []
    for number, instance in enumerate(document["instances"]):
        answer = solve(instance)
        answers.append(answer)
        sizes.append(len(answer["rectangles"]))
        print(
            "instance %d: rows=%d cols=%d minimum=%d"
            % (number, int(instance["rows"]), int(instance["cols"]), len(answer["rectangles"])),
            flush=True)
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    (output / "solution.json").write_text(
        json.dumps({"instances": answers}, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    lines = ["instance,minimum_size"]
    for number, size in enumerate(sizes):
        lines.append("%d,%d" % (number, size))
    (output / "optimum.csv").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
