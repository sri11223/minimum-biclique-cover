#!/usr/bin/env python3
"""Reference solver for the minimum biclique cover task.

The cover is recovered from the minimal column profiles of the 0/1 matrix: a
maximal all-ones rectangle is exactly a set of columns with the same row
profile, and the minimal distinct profiles are the structurally maximal
rectangles that every minimum cover can be built from.  The packing witness
comes from an owner-set peeling order: hand every rectangle a row whose
remaining owners are exhausted except for that rectangle itself, which leaves
no pair of chosen entries crossed.  Both sides are re-verified against the raw
1-entries before the answer is written.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path


def load_document(path):
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def split_bits(mask):
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def adjacency(rows, cols, ones):
    cols_of_row = [0] * rows
    rows_of_col = [0] * cols
    for u, v in ones:
        cols_of_row[u] |= 1 << v
        rows_of_col[v] |= 1 << u
    return cols_of_row, rows_of_col


def minimal_profiles(profiles):
    """Distinct non-empty profiles, keeping those that contain no other one."""
    distinct = sorted({p for p in profiles if p}, key=lambda value: value.bit_count())
    minimal = []
    for profile in distinct:
        if any(kept & profile == kept for kept in minimal):
            continue
        minimal.append(profile)
    return minimal


def recover_rectangles(cols, rows_of_col):
    minimal = minimal_profiles(rows_of_col)
    rectangles = []
    corner_columns = []
    for profile in minimal:
        row_list = split_bits(profile)
        col_list = [v for v in range(cols) if rows_of_col[v] & profile == profile]
        rectangles.append((row_list, col_list))
        corner = None
        for v in range(cols):
            if rows_of_col[v] == profile:
                corner = v
                break
        corner_columns.append(corner)
    return rectangles, corner_columns


def owner_sets(rectangles, row_count):
    owners = [set() for _ in range(row_count)]
    for index, (row_list, _cols) in enumerate(rectangles):
        for u in row_list:
            owners[u].add(index)
    return owners


def peel_packing(rectangles, owners):
    """Assign every rectangle a distinct owner row whose remaining owners are
    exactly that rectangle.  Every rectangle handled earlier owns none of the
    later chosen rows, so no two chosen entries are crossed."""
    count = [len(s) for s in owners]
    remaining = set(range(len(rectangles)))
    used = [False] * len(owners)
    picks = {}
    while remaining:
        choice = None
        for u, owners_of_u in enumerate(owners):
            if used[u] or count[u] != 1:
                continue
            target = next(iter(owners_of_u & remaining), None)
            if target is None:
                count[u] = 0
                continue
            choice = (u, target)
            break
        if choice is None:
            return None
        u, target = choice
        picks[target] = u
        used[u] = True
        remaining.discard(target)
        for w in rectangles[target][0]:
            if count[w]:
                count[w] -= 1
    return picks


def crossed(entry_one, entry_two, cols_of_row):
    u1, v1 = entry_one
    u2, v2 = entry_two
    return bool((cols_of_row[u1] >> v2) & 1) and bool((cols_of_row[u2] >> v1) & 1)


def packing_is_valid(entries, cols_of_row):
    seen = set()
    for u, v in entries:
        if (u, v) in seen or not (cols_of_row[u] >> v) & 1:
            return False
        seen.add((u, v))
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if crossed(entries[i], entries[j], cols_of_row):
                return False
    return True


def cover_is_valid(rectangles, rows, cols_of_row):
    covered = [0] * rows
    for row_list, col_list in rectangles:
        col_mask = 0
        for v in col_list:
            col_mask |= 1 << v
        for u in row_list:
            if cols_of_row[u] & col_mask != col_mask:
                return False
            covered[u] |= col_mask
    for u in range(rows):
        if cols_of_row[u] & ~covered[u]:
            return False
    return True


def random_packing(rectangles, cols_of_row, attempts=4000, seed=12345):
    rng = random.Random(seed)
    for _ in range(attempts):
        entries = []
        for row_list, col_list in rectangles:
            u = row_list[rng.randrange(len(row_list))]
            v = col_list[rng.randrange(len(col_list))]
            entries.append((u, v))
        if packing_is_valid(entries, cols_of_row):
            return entries
    return None


def solve_instance(instance):
    rows = int(instance["rows"])
    cols = int(instance["cols"])
    ones = [(int(u), int(v)) for u, v in instance["ones"]]
    cols_of_row, rows_of_col = adjacency(rows, cols, ones)
    rectangles, corner_columns = recover_rectangles(cols, rows_of_col)
    owners = owner_sets(rectangles, rows)
    picks = peel_packing(rectangles, owners)
    entries = None
    if picks is not None and all(corner is not None for corner in corner_columns):
        candidate = [(picks[index], corner_columns[index]) for index in range(len(rectangles))]
        if packing_is_valid(candidate, cols_of_row):
            entries = candidate
    if entries is None:
        entries = random_packing(rectangles, cols_of_row)
    if entries is None or not cover_is_valid(rectangles, rows, cols_of_row):
        raise RuntimeError("could not derive a valid two-sided answer for an instance")
    if len(rectangles) != len(entries):
        raise RuntimeError("cover and packing sizes disagree")
    answer = {
        "rectangles": [[list(row_list), list(col_list)] for row_list, col_list in rectangles],
        "packing": [[u, v] for u, v in entries],
    }
    return answer


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: biclique_cover_reference.py INPUT_JSON OUTPUT_DIRECTORY")
    document = load_document(sys.argv[1])
    answers = []
    for number, instance in enumerate(document["instances"]):
        answer = solve_instance(instance)
        answers.append(answer)
        print(
            "instance %d: rows=%d cols=%d minimum=%d"
            % (number, int(instance["rows"]), int(instance["cols"]), len(answer["rectangles"])),
            flush=True)
    output = Path(sys.argv[2])
    output.mkdir(parents=True, exist_ok=True)
    (output / "solution.json").write_text(
        json.dumps({"instances": answers}, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
