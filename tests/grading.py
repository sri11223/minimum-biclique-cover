"""Exact scientific validation for the minimum biclique cover task.

The verifier re-derives every requirement from the input matrix: a candidate
answer must provide a list of all-ones rectangles (every listed cell is a
1-entry, every 1-entry is covered by at least one rectangle) and a packing list
of 1-entries no two of which lie in a common all-ones rectangle - two entries
$(u_1, v_1)$, $(u_2, v_2)$ lie in a common all-ones rectangle exactly when
$(u_1, v_2)$ and $(u_2, v_1)$ are both 1-entries.  One all-ones rectangle can
contain at most one packing entry, so the packing size is a lower bound on every
rectangle cover and the two lists are required to have equal length: with equal
sizes the optimum is proved directly from the answer.  In addition the common
size must equal the independently certified minimum for that instance.  Read
paths are strict: JSON with duplicate keys or non-finite numbers, quoted CSV
fields and non-canonical integers are rejected with measured values in the
message.  Candidate-controlled files are read symlink-proof: the path must stay
inside the expected output directory, the opened descriptor is re-checked with
fstat, and the bytes are read from that descriptor (lstat plus O_NOFOLLOW where
the platform provides it), so a solution.json that is a link to a packaged
certificate is rejected instead of followed.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import stat
from pathlib import Path

SCHEMA = "minimum-biclique-cover-v1"
OUTPUT_KEYS = {"rectangles", "packing"}
INPUT_KEYS = {"rows", "cols", "ones"}
MAX_INSTANCES = 4
MAX_SIDE = 60000
MAX_ONES = 2000000


class ContractError(Exception):
    pass


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ContractError(f"non-finite JSON number: {value}")


# Symlink-proof reading of candidate-controlled paths.  O_NOFOLLOW is not
# available on every platform; the lstat check plus the fstat re-check on the
# opened descriptor still reject links, and the realpath containment check
# rejects links in any path component that would escape the expected output
# directory.
O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
MAX_READ_BYTES = 64_000_000


def read_regular_bytes(path: Path, expected_directory: Path | None = None, label: str = "file") -> bytes:
    """Read a regular file without ever following a link.

    When `expected_directory` is given (a candidate-controlled path), the
    realpath must stay inside that directory, the final component must not be a
    symlink (rejected with lstat and again by opening with O_NOFOLLOW), and the
    bytes are read from the opened descriptor so the object re-checked with
    fstat is exactly the object read.
    """
    if expected_directory is not None:
        directory = os.path.realpath(expected_directory)
        resolved = os.path.realpath(path)
        if resolved != directory and not resolved.startswith(directory + os.sep):
            raise ContractError(
                f"{label} resolves to {resolved}, outside the expected output directory {directory}")
    try:
        metadata = os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        raise ContractError(f"{label} is missing: {path}")
    if stat.S_ISLNK(metadata.st_mode):
        raise ContractError(f"{label} is a symbolic link; links are not accepted: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ContractError(f"{label} is not a regular file: {path}")
    if metadata.st_size > MAX_READ_BYTES:
        raise ContractError(f"{label} has size {metadata.st_size} bytes; required limit is {MAX_READ_BYTES} bytes")
    try:
        descriptor = os.open(path, os.O_RDONLY | O_NOFOLLOW)
    except OSError as exc:
        raise ContractError(f"{label} cannot be opened safely: {path}: {type(exc).__name__}: {exc}")
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ContractError(f"{label} is not a regular file: {path}")
        if opened.st_size > MAX_READ_BYTES:
            raise ContractError(f"{label} has size {opened.st_size} bytes; required limit is {MAX_READ_BYTES} bytes")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def read_json(path: Path, expected_directory: Path | None = None) -> dict:
    data = read_regular_bytes(path, expected_directory, label="JSON file")
    try:
        text = data.decode("utf-8")
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=_constant)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(f"invalid strict JSON in {path}: {type(exc).__name__}: {exc}")
    if not isinstance(value, dict):
        raise ContractError("top-level JSON value must be an object")
    return value


def parse_optimum(path: Path, expected_directory: Path | None = None) -> dict[int, int]:
    data = read_regular_bytes(path, expected_directory, label="independent optimum table")
    text = data.decode("utf-8")
    _lines = text.split("\n")
    required_header = "instance,minimum_size"
    if not _lines or _lines[0] != required_header:
        measured_header = _lines[0] if _lines else None
        raise ContractError(f"optimum.csv raw header is {measured_header!r}; required header is {required_header}")
    rows = list(csv.reader(io.StringIO(text, newline=""), strict=True))
    result = {}
    for line_number, (raw, row) in enumerate(zip(_lines, rows), 1):
        if raw.split(",") != row:
            raise ContractError(f"optimum.csv line {line_number} contains CSV-quoted or normalized fields")
        if line_number == 1:
            continue
        if len(row) != 2 or not re.fullmatch(r"(?:0|[1-9][0-9]*)", row[0]) or not re.fullmatch(r"(?:0|[1-9][0-9]*)", row[1]):
            raise ContractError(f"optimum.csv line {line_number} must contain an instance index and a canonical minimum size")
        index = int(row[0])
        if index in result:
            raise ContractError(f"optimum.csv repeats instance {index}")
        result[index] = int(row[1])
    if len(_lines) != len(rows) + 1 or _lines[-1] != "":
        raise ContractError(f"optimum.csv has {len(rows)} parsed rows and final spelling {_lines[-1]!r}; required layout is one terminal LF")
    if not result:
        raise ContractError("optimum.csv contains no data rows")
    return result


def _integer(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_input(document: dict) -> None:
    if set(document) != {"schema_version", "instances"} or document.get("schema_version") != SCHEMA:
        raise ContractError(f"input must have exactly schema_version/instances and schema {SCHEMA}")
    instances = document["instances"]
    if not isinstance(instances, list) or not instances:
        raise ContractError("input must contain a non-empty instance list")
    if len(instances) > MAX_INSTANCES:
        raise ContractError(
            f"input must contain at most {MAX_INSTANCES} instances; this one has {len(instances)}")
    for index, instance in enumerate(instances):
        if not isinstance(instance, dict) or set(instance) != INPUT_KEYS:
            raise ContractError(f"instance {index} keys must be exactly {sorted(INPUT_KEYS)}")
        rows = instance["rows"]
        if not _integer(rows) or not 1 <= rows <= MAX_SIDE:
            raise ContractError(f"instance {index}: rows={rows!r}; required range is 1 through {MAX_SIDE}")
        cols = instance["cols"]
        if not _integer(cols) or not 1 <= cols <= MAX_SIDE:
            raise ContractError(f"instance {index}: cols={cols!r}; required range is 1 through {MAX_SIDE}")
        ones = instance["ones"]
        if not isinstance(ones, list) or not ones:
            raise ContractError(f"instance {index}: ones must be a non-empty array of 1-entries")
        if len(ones) > MAX_ONES:
            raise ContractError(
                f"instance {index}: ones has {len(ones)} entries; required limit is {MAX_ONES}")
        seen = set()
        for position, entry in enumerate(ones):
            if not isinstance(entry, list) or len(entry) != 2:
                raise ContractError(f"instance {index} entry {position}: each 1-entry must be an array of [row, column]")
            u, v = entry
            if not _integer(u) or not 0 <= u < rows:
                raise ContractError(f"instance {index} entry {position}: row {u!r} is outside 0..{rows - 1}")
            if not _integer(v) or not 0 <= v < cols:
                raise ContractError(f"instance {index} entry {position}: column {v!r} is outside 0..{cols - 1}")
            key = (u, v)
            if key in seen:
                raise ContractError(f"instance {index} entry {position}: duplicate 1-entry {key}")
            seen.add(key)


def _validate_rectangles(index: int, rectangles, rows: int, cols: int, cols_of_row) -> list:
    if not isinstance(rectangles, list) or not rectangles:
        raise ContractError(f"answer {index}: rectangles must be a non-empty array")
    checked = []
    for position, rectangle in enumerate(rectangles):
        if (not isinstance(rectangle, list) or len(rectangle) != 2
                or not isinstance(rectangle[0], list) or not isinstance(rectangle[1], list)):
            raise ContractError(f"answer {index} rectangle {position}: each rectangle must be [rows, columns] arrays")
        row_list, col_list = rectangle
        if not row_list or not col_list:
            raise ContractError(f"answer {index} rectangle {position}: rows and columns must be non-empty arrays")
        seen_rows = set()
        for u in row_list:
            if not _integer(u) or not 0 <= u < rows:
                raise ContractError(
                    f"answer {index} rectangle {position}: row {u!r} is outside 0..{rows - 1}")
            if u in seen_rows:
                raise ContractError(f"answer {index} rectangle {position}: repeated row {u}")
            seen_rows.add(u)
        seen_cols = set()
        col_mask = 0
        for v in col_list:
            if not _integer(v) or not 0 <= v < cols:
                raise ContractError(
                    f"answer {index} rectangle {position}: column {v!r} is outside 0..{cols - 1}")
            if v in seen_cols:
                raise ContractError(f"answer {index} rectangle {position}: repeated column {v}")
            seen_cols.add(v)
            col_mask |= 1 << v
        for u in row_list:
            if cols_of_row[u] & col_mask != col_mask:
                raise ContractError(
                    f"answer {index} rectangle {position}: cell ({u}, v) is not a 1-entry of the matrix "
                    f"for some column v listed in the rectangle")
        checked.append((row_list, col_mask))
    return checked


def _validate_packing(index: int, packing, rows: int, cols: int, cols_of_row) -> list:
    if not isinstance(packing, list) or not packing:
        raise ContractError(f"answer {index}: packing must be a non-empty array")
    seen = set()
    entries = []
    for position, entry in enumerate(packing):
        if not isinstance(entry, list) or len(entry) != 2:
            raise ContractError(f"answer {index} packing entry {position}: each entry must be an array of [row, column]")
        u, v = entry
        if not _integer(u) or not 0 <= u < rows:
            raise ContractError(f"answer {index} packing entry {position}: row {u!r} is outside 0..{rows - 1}")
        if not _integer(v) or not 0 <= v < cols:
            raise ContractError(f"answer {index} packing entry {position}: column {v!r} is outside 0..{cols - 1}")
        if (u, v) in seen:
            raise ContractError(f"answer {index} packing entry {position}: repeated entry ({u}, {v})")
        seen.add((u, v))
        if (cols_of_row[u] >> v) & 1 == 0:
            raise ContractError(f"answer {index} packing entry {position}: ({u}, {v}) is not a 1-entry")
        entries.append((u, v))
    return entries


def _packing_pair_check(index: int, entries, cols_of_row) -> None:
    """No two packing entries may lie in a common all-ones rectangle."""
    for position in range(len(entries)):
        u1, v1 = entries[position]
        for other in range(position + 1, len(entries)):
            u2, v2 = entries[other]
            cross_one = (cols_of_row[u1] >> v2) & 1
            cross_two = (cols_of_row[u2] >> v1) & 1
            if cross_one and cross_two:
                raise ContractError(
                    f"answer {index}: packing entries {position} ({u1}, {v1}) and {other} ({u2}, {v2}) lie in a "
                    f"common all-ones rectangle, because ({u1}, {v2}) and ({u2}, {v1}) are both 1-entries")


def validate_solution(document: dict, solution: dict, optimum: dict[int, int]) -> None:
    if set(solution) != {"instances"}:
        raise ContractError("solution must have exactly the instances key")
    instances = document["instances"]
    answers = solution["instances"]
    if not isinstance(answers, list) or len(answers) != len(instances):
        raise ContractError(
            f"solution contains {len(answers) if isinstance(answers, list) else 'non-array'} entries; required count is {len(instances)}")
    if set(optimum) != set(range(len(instances))):
        raise ContractError("independent optimum roster does not match the input instance roster")
    for index, (instance, answer) in enumerate(zip(instances, answers)):
        if not isinstance(answer, dict) or set(answer) != OUTPUT_KEYS:
            raise ContractError(f"answer {index} must have exactly {sorted(OUTPUT_KEYS)}")
        rows = instance["rows"]
        cols = instance["cols"]
        cols_of_row = [0] * rows
        for u, v in instance["ones"]:
            cols_of_row[u] |= 1 << v
        rectangles_raw = answer["rectangles"]
        packing_raw = answer["packing"]
        # Size bounds come first so a candidate cannot force unbounded work: a
        # cover or packing larger than the certified minimum is rejected before
        # any list is walked.
        if isinstance(rectangles_raw, list) and len(rectangles_raw) > optimum[index]:
            raise ContractError(
                f"answer {index}: measured cover size {len(rectangles_raw)}; independently required minimum is {optimum[index]}")
        if isinstance(packing_raw, list) and len(packing_raw) > optimum[index]:
            raise ContractError(
                f"answer {index}: measured packing size {len(packing_raw)}; independently required minimum is {optimum[index]}")
        rectangles = _validate_rectangles(index, rectangles_raw, rows, cols, cols_of_row)
        packing = _validate_packing(index, packing_raw, rows, cols, cols_of_row)
        covered = [0] * rows
        for row_list, col_mask in rectangles:
            for u in row_list:
                covered[u] |= col_mask
        for u in range(rows):
            missing = cols_of_row[u] & ~covered[u]
            if missing:
                v = (missing & -missing).bit_length() - 1
                raise ContractError(
                    f"answer {index}: 1-entry ({u}, {v}) is not covered by any listed rectangle")
        if len(rectangles) != len(packing):
            raise ContractError(
                f"answer {index}: cover has {len(rectangles)} rectangles but the packing lists {len(packing)} entries; "
                f"equal lengths are required so the optimum is proved in-check")
        _packing_pair_check(index, packing, cols_of_row)


def validate_directory(data_path: Path, output_directory: Path) -> None:
    document = read_json(data_path)
    validate_input(document)
    solution = read_json(output_directory / "solution.json", expected_directory=output_directory)
    optimum = parse_optimum(output_directory / "optimum.csv", expected_directory=output_directory)
    validate_solution(document, solution, optimum)
