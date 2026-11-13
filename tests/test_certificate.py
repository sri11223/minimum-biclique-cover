"""Verifier tests for the minimum biclique cover task.

Each test names the requirement it enforces: candidate staging, the execution
contract (network isolation, process cleanup, no timeout), the output-directory
link screen, the independent certificate, the candidate's own two-sided answer,
derivation of the answer from the input (an additional packaged instance set run
through the same sandboxed invocation path), derivation of the answer from
instance sets generated at verification time, enforced resource limits, packaging
of the independent producer, strict parsing regressions, and end-to-end disk
parsing of the packaged certificate pair.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import shutil
import stat
from pathlib import Path

import pytest

from grading import (SCHEMA, ContractError, parse_optimum, read_json, validate_directory,
                     validate_input, validate_solution)

TESTS = Path("/tests")
DATA = TESTS / "data"
CERTIFICATE = DATA / "independent_certificate"
SPLITS = ("public", "heldout")
DERIVATION = Path("/root/derivation")


def input_path(split: str) -> Path:
    return DATA / f"{split}_instances.json"


def output_path(split: str) -> Path:
    return Path("/root/biclique_cover_submission/runtime_outputs") / split


def solver_stderr_tail(split: str) -> str:
    path = Path("/root/logs") / f"{split}.stderr"
    if not path.is_file():
        return "no solver stderr log was produced"
    return path.read_text(encoding="utf-8", errors="replace")[-4000:]


def load_harness():
    """Load the packaged harness module so the derivation test reuses its invocation path."""
    spec = importlib.util.spec_from_file_location("harness_run", "/harness_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def remove_tree(path: Path) -> None:
    """Remove a derivation path without ever following a link at its root."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def record_invocation_time(name, seconds):
    """Record a verifier-run invocation's wall time for the aggregate budget check."""
    directory = Path("/root/status")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"verifier_{name}.json").write_text(
        json.dumps({"elapsed_seconds": float(seconds)}) + "\n", encoding="utf-8", newline="\n")


GENERATED_SIZES = (8, 12)


def generated_cases(solver_digest=None):
    """The generated cases: a deterministic function of the submitted solver's bytes.

    The export stages the candidate source and records its SHA-256 in
    `/root/export.json`, so each submission always faces the same two instances
    and the verifier's verdict is reproducible; different submissions face
    different instances, so no precomputed table or input fingerprint can answer
    them.  Both sizes are fixed (8 and 12 planted rectangles: a mid-size instance
    and a larger one) so the verifier cost and the recovery depth exercised are
    bounded and known.  Every case certifies itself by construction inside
    `build_generated_instance` before the candidate runs.
    """
    if solver_digest is None:
        try:
            recorded = read_json(Path("/root/export.json")).get("solver_sha256", "")
            solver_digest = bytes.fromhex(recorded) if len(recorded) == 64 else None
        except Exception:
            solver_digest = None
    if not solver_digest:
        solver_digest = hashlib.sha256(b"minimum-biclique-cover").digest()
    cases = []
    for index, count in enumerate(GENERATED_SIZES):
        seed = int.from_bytes(hashlib.sha256(solver_digest + bytes([index])).digest()[:8], "big")
        cases.append((seed, count))
    return cases


def verifier_invocation_areas():
    """Every area the verifier itself stages for a candidate invocation."""
    return [DERIVATION] + [Path(f"/root/generated-{index}") for index in range(len(GENERATED_SIZES))]


def clear_other_invocation_areas(keep=None):
    """Remove every verifier-created invocation area except `keep`.

    The contract promises that no scratch state survives between invocations;
    the harness wipes its own runtime tree, so the areas the verifier stages must
    be removed here as well before each further invocation (and again after the
    last one), so the candidate can never read an earlier invocation's scratch.
    """
    for area in verifier_invocation_areas():
        if keep is not None and area == keep:
            continue
        remove_tree(area)


def test_export_has_no_staging_error():
    """Candidate code and launcher files were staged without an export error."""
    export = read_json(Path("/root/export.json"))
    assert export.get("error") is None, f"candidate staging failed: {export.get('error')}"


def test_submission_is_self_contained():
    """The export enforces the contract's single-file self-containment requirement.

    The contract requires the solver to be one self-contained source file that
    does not depend on any sibling in its submission directory.  The export
    removes every pre-existing entry other than the managed outputs tree, leaves
    the submission directory traverse-only and the staged source root-readable
    only, so the unprivileged solver can neither list nor read a sibling helper;
    this test asserts that staged state from the shipped files.
    """
    export = read_json(Path("/root/export.json"))
    assert isinstance(export.get("siblings_removed"), list), (
        "the export did not record the submission-directory cleanup")
    submission = Path("/root/biclique_cover_submission")
    entries = sorted(entry.name for entry in submission.iterdir())
    assert entries == ["runtime_outputs", "solver.py"], (
        f"the staged submission directory carries unexpected entries: {entries}")
    directory_mode = stat.S_IMODE(submission.stat().st_mode)
    assert directory_mode & 0o044 == 0, (
        f"the submission directory is group/world readable: {oct(directory_mode)}")
    source_mode = stat.S_IMODE((submission / "solver.py").stat().st_mode)
    assert source_mode & 0o077 == 0, (
        f"the staged source is readable by the solver user: {oct(source_mode)}")


@pytest.mark.parametrize("split", SPLITS)
def test_candidate_execution_contract(split):
    """The invocation finishes normally with network isolation and process cleanup."""
    status = read_json(Path("/root/status") / f"{split}.json")
    assert status["network_blocked"] is True, "network prevention probe did not confirm the seccomp block"
    assert status["timed_out"] is False, f"{split} invocation exceeded the 3600-second solver limit"
    assert status["cleanup_ok"] is True, f"{split} invocation left an unprivileged process running"
    assert status["returncode"] == 0, (
        f"{split} solver returned {status['returncode']}; required status is 0. "
        f"Solver stderr tail:\n{solver_stderr_tail(split)}")


@pytest.mark.parametrize("split", SPLITS)
def test_output_directory_contains_no_links(split):
    """The harness screened the output directory: no links or special files were left.

    Links can redirect the privileged grader into verifier-owned data (for
    example a packaged certificate) that the unprivileged solver cannot read, so
    any symlink or special entry the candidate leaves in its writable output
    directory is a hard failure, recorded by the harness before grading; extra
    regular files and real subdirectories are permitted.
    """
    status = read_json(Path("/root/status") / f"{split}.json")
    assert "violations" in status, "case status lacks the harness output-directory screening record"
    assert status["violations"] == [], (
        f"{split} output directory contains rejected entries: {status['violations']}")


@pytest.mark.parametrize("split", SPLITS)
def test_independent_certificate(split):
    """The packaged independent answer covers and packs at the certified minimum."""
    document = read_json(input_path(split))
    validate_input(document)
    certificate = read_json(CERTIFICATE / split / "solution.json")
    optimum = parse_optimum(CERTIFICATE / split / "optimum.csv")
    validate_solution(document, certificate, optimum)
    assert len(certificate["instances"]) == len(document["instances"])


@pytest.mark.parametrize("split", SPLITS)
def test_candidate_two_sided_answer(split):
    """The candidate's cover and packing are valid and attain the certified minimum."""
    document = read_json(input_path(split))
    validate_input(document)
    candidate = read_json(output_path(split) / "solution.json", expected_directory=output_path(split))
    optimum = parse_optimum(CERTIFICATE / split / "optimum.csv")
    validate_solution(document, candidate, optimum)


def test_resource_limits_are_enforced():
    """The candidate launcher enforces aggregate CPU-time and 8192-MiB address-space caps."""
    harness = Path("/harness_run.py").read_text(encoding="utf-8")
    assert "/usr/bin/prlimit" in harness
    assert "--cpu=14400" in harness
    assert "--as=8589934592" in harness


def test_invocation_isolation_properties():
    """Fresh outputs, wiped scratch state, the network block, and process cleanup are enforced by the packaged harness."""
    harness = Path("/harness_run.py").read_text(encoding="utf-8")
    assert "runtime_outputs" in harness, "case outputs must live in the verifier-owned tree"
    assert "/usr/bin/setpriv" in harness, "candidates must run unprivileged"
    assert "pkill" in harness and "-KILL" in harness, "leftover candidate processes must be swept"
    assert "os.chmod(other_out, 0o000" in harness, "the other case's outputs must be sealed"
    assert "def clear_runtime_tree" in harness, "prior scratch state must be wiped between invocations"
    assert "clear_runtime_tree()" in harness, "each case must wipe the runtime tree before its invocation"
    suite = Path(__file__).read_text(encoding="utf-8")
    assert "clear_runtime_tree()" in suite, "the derivation invocation must wipe prior scratch state too"
    import harness_run
    assert harness_run.CASE_TIMEOUT_SECONDS == 3600
    assert harness_run.solver_network_blocked() is True, "the socket filter must be active"


def test_independent_producer_is_packaged():
    """The independent producer is present and uses its own structural recovery path."""
    producer = CERTIFICATE / "producing_solver.py"
    assert producer.is_file(), "the certificate-producing implementation must be packaged"
    text = producer.read_text(encoding="utf-8")
    assert "/solution" not in text and "reference_solver" not in text
    assert "def main" in text
    assert "column_classes" in text and "witness_by_owners" in text
    assert "minimal_profiles" not in text and "peel_packing" not in text
    assert "recover_rectangles" not in text


def test_strict_json_and_optimum_regressions(tmp_path):
    """Malformed JSON, duplicate keys, quoting, oversized, mismatched and invalid answers are rejected."""
    document = read_json(input_path("public"))
    certificate = read_json(CERTIFICATE / "public" / "solution.json")
    optimum = parse_optimum(CERTIFICATE / "public" / "optimum.csv")
    validate_solution(document, certificate, optimum)

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"instances":NaN}', encoding="utf-8")
    with pytest.raises(ContractError):
        read_json(malformed)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"instances":[],"instances":[]}', encoding="utf-8")
    with pytest.raises(ContractError):
        read_json(duplicate)

    quoted = tmp_path / "quoted.csv"
    quoted.write_text('"instance","minimum_size"\n0,3\n', encoding="utf-8")
    with pytest.raises(ContractError):
        parse_optimum(quoted)

    wrong_optimum = dict(optimum)
    wrong_optimum[0] = max(1, wrong_optimum[0] - 1)
    with pytest.raises(ContractError, match="independently required minimum"):
        validate_solution(document, certificate, wrong_optimum)

    oversized = json.loads(json.dumps(certificate))
    first = document["instances"][0]
    u, v = first["ones"][0]
    oversized["instances"][0]["rectangles"].append([[u], [v]])
    with pytest.raises(ContractError, match="independently required minimum"):
        validate_solution(document, oversized, optimum)

    uncovered = json.loads(json.dumps(certificate))
    uncovered["instances"][0]["rectangles"].pop()
    uncovered["instances"][0]["packing"].pop()
    with pytest.raises(ContractError, match="is not covered"):
        validate_solution(document, uncovered, optimum)

    mismatched = json.loads(json.dumps(certificate))
    mismatched["instances"][0]["packing"].pop()
    with pytest.raises(ContractError, match="equal lengths are required"):
        validate_solution(document, mismatched, optimum)

    non_packing = json.loads(json.dumps(certificate))
    packing = non_packing["instances"][0]["packing"]
    ones_set = {(x, y) for x, y in first["ones"]}
    anchor_row, anchor_col = packing[1]
    replacement = None
    for candidate_row, candidate_col in sorted(ones_set):
        if (candidate_row, candidate_col) in {tuple(e) for e in packing}:
            continue
        if candidate_row != anchor_row:
            continue
        replacement = [candidate_row, candidate_col]
        break
    assert replacement is not None, "no conflicting replacement entry exists for the regression"
    packing[0] = replacement
    with pytest.raises(ContractError, match="common all-ones rectangle"):
        validate_solution(document, non_packing, optimum)


def test_packaged_certificate_pair_parses_from_disk(tmp_path):
    """The independent answer and optimum table pass end-to-end disk parsing."""
    shutil.copyfile(CERTIFICATE / "public" / "solution.json", tmp_path / "solution.json")
    shutil.copyfile(CERTIFICATE / "public" / "optimum.csv", tmp_path / "optimum.csv")
    validate_directory(input_path("public"), tmp_path)


def test_candidate_derives_answers_from_input():
    """The candidate solves a packaged additional instance set from that input.

    The additional set is never shown at development time and is packaged with
    its own independently produced certificate, so a solver that replays
    memorised development answers, or any precomputed variant of them, cannot
    answer it. The candidate is invoked through the harness's own sandboxed path
    (unprivileged user, network block, CPU/address-space limits, fresh scratch,
    and the prior invocations' runtime tree wiped first) and must write a valid
    minimal certificate for the additional set.
    """
    document = read_json(DATA / "derivation_instances.json")
    validate_input(document)
    certificate = read_json(CERTIFICATE / "derivation" / "solution.json")
    optimum = parse_optimum(CERTIFICATE / "derivation" / "optimum.csv")
    # The packaged certificate is the expected derivation: it is checked here
    # directly against the additional input before the candidate is graded.
    validate_solution(document, certificate, optimum)

    harness = load_harness()
    assert harness.solver_network_blocked() is True, (
        "the network-prevention probe did not confirm the seccomp block for the derivation invocation")
    # No scratch state survives between invocations: wipe the runtime tree left by
    # the development and unseen runs before this third invocation, exactly as the
    # harness does before each of its own cases, and remove every other area the
    # verifier stages, so the candidate can never read an earlier invocation's
    # leftover scratch files.
    harness.clear_runtime_tree()
    clear_other_invocation_areas()
    DERIVATION.mkdir(parents=True)
    os.chmod(DERIVATION, 0o755)
    staged = DERIVATION / "derivation_instances.json"
    staged.write_bytes((json.dumps(document, sort_keys=True) + "\n").encode("utf-8"))
    os.chmod(staged, 0o444)
    staged_output = DERIVATION / "output"
    runtime = DERIVATION / "runtime"
    scratch = DERIVATION / "scratch"
    logs = DERIVATION / "logs"
    for directory in (staged_output, runtime, scratch):
        directory.mkdir()
        os.chown(directory, 65534, 65534, follow_symlinks=False)
        os.chmod(directory, 0o700, follow_symlinks=False)
    logs.mkdir()
    execution = harness.execute_solver(
        staged, staged_output, runtime, scratch, logs / "stdout.log", logs / "stderr.log")
    record_invocation_time("derivation", execution["elapsed_seconds"])
    stderr_tail = (logs / "stderr.log").read_text(encoding="utf-8", errors="replace")[-4000:]
    assert execution["timed_out"] is False, "the derivation invocation exceeded the 3600-second solver limit"
    assert execution["cleanup_ok"] is True, "the derivation invocation left an unprivileged process running"
    assert execution["returncode"] == 0, (
        f"the candidate returned {execution['returncode']} on the additional instance set; required status is 0. "
        f"Solver stderr tail:\n{stderr_tail}")
    violations = harness.scan_output_directory(staged_output)
    assert violations == [], f"the derivation output directory contains rejected entries: {violations}"
    candidate = read_json(staged_output / "solution.json", expected_directory=staged_output)
    validate_solution(document, candidate, optimum)
    # Leave no scratch behind for the further verifier invocations.
    clear_other_invocation_areas()


def _try_build_generated(count, seed):
    """One deterministic attempt at a small planted instance, or None.

    The matrix carries exactly `count` planted rectangles, one private corner
    column per rectangle, corner rows that join only rectangles of smaller
    construction index, and shared rows/columns, with the labels permuted.  The
    corner cells are pairwise not co-coverable, so the minimum is exactly
    `count` whenever the planted structure also survives as the set of minimal
    column profiles; the caller asserts that survival before accepting.
    """
    rng = random.Random(seed)
    row_members = [{i} for i in range(count)]
    col_members = [{i} for i in range(count)]
    for i in range(count):
        for j in range(i):
            if rng.random() < 0.6:
                row_members[j].add(i)
    n_shared = max(4, count)
    next_row = count
    for _ in range(n_shared):
        size = rng.randint(2, max(2, count // 2))
        for i in rng.sample(range(count), size):
            row_members[i].add(next_row)
        next_row += 1
    next_col = count
    for _ in range(n_shared):
        size = rng.randint(2, max(2, count // 2))
        for i in rng.sample(range(count), size):
            col_members[i].add(next_col)
        next_col += 1
    row_perm = list(range(next_row))
    col_perm = list(range(next_col))
    rng.shuffle(row_perm)
    rng.shuffle(col_perm)
    ones = set()
    for i in range(count):
        for u in row_members[i]:
            for v in col_members[i]:
                ones.add((row_perm[u], col_perm[v]))
    for i in range(count):
        if (row_perm[i], col_perm[i]) not in ones:
            return None
    for i in range(count):
        for j in range(i + 1, count):
            if ((row_perm[i], col_perm[j]) in ones) and ((row_perm[j], col_perm[i]) in ones):
                return None
    by_column = [0] * next_col
    for u, v in ones:
        by_column[v] |= 1 << u
    distinct = sorted({profile for profile in by_column if profile}, key=lambda value: value.bit_count())
    minimal = []
    for profile in distinct:
        if any(kept & profile == kept for kept in minimal):
            continue
        minimal.append(profile)
    if len(minimal) != count:
        return None
    document = {
        "schema_version": SCHEMA,
        "instances": [{"rows": next_row, "cols": next_col, "ones": [list(entry) for entry in sorted(ones)]}],
    }
    return document


def build_generated_instance(count, seed):
    """Build a certified generated instance, deterministically from the seed."""
    for attempt in range(400):
        document = _try_build_generated(count, seed + attempt)
        if document is not None:
            return document, count
    raise AssertionError("no generated instance satisfied the construction certificate")


def test_candidate_derives_answers_from_generated_input():
    """The candidate derives answers for instance sets generated at verification time.

    The instruction requires every answer to be derived from the input rather
    than replayed.  The packaged sets are fixed, so a solver carrying precomputed
    answers for them could pass the packaged cases; this test therefore builds two
    further small instances as a deterministic function of the submitted solver's
    own bytes (the same submission always faces the same two instances, so the
    verdict is reproducible; different submissions face different instances, so
    no precomputed table or input fingerprint can answer them), and runs the
    candidate through the same sandboxed invocation path (unprivileged user,
    network block, resource limits, fresh scratch, and the prior invocations'
    runtime tree wiped first), requiring a valid minimal two-sided answer for
    each.  Every generated optimum is certified by construction - the planted
    rectangles cover the matrix with exactly `count` rectangles and the corner
    cells are pairwise not co-coverable, both asserted before the candidate runs
    - so no table of packaged answers can satisfy these invocations.
    """
    harness = load_harness()
    assert harness.solver_network_blocked() is True, (
        "the network-prevention probe did not confirm the seccomp block for a generated invocation")
    for index, (seed, count) in enumerate(generated_cases()):
        document, optimum_size = build_generated_instance(count, seed)
        validate_input(document)
        area = Path(f"/root/generated-{index}")
        harness.clear_runtime_tree()
        clear_other_invocation_areas(keep=area)
        remove_tree(area)
        area.mkdir(parents=True)
        os.chmod(area, 0o755)
        staged = area / "generated_instances.json"
        staged.write_bytes((json.dumps(document, sort_keys=True) + "\n").encode("utf-8"))
        os.chmod(staged, 0o444)
        staged_output = area / "output"
        runtime = area / "runtime"
        scratch = area / "scratch"
        logs = area / "logs"
        for directory in (staged_output, runtime, scratch):
            directory.mkdir()
            os.chown(directory, 65534, 65534, follow_symlinks=False)
            os.chmod(directory, 0o700, follow_symlinks=False)
        logs.mkdir()
        execution = harness.execute_solver(
            staged, staged_output, runtime, scratch, logs / "stdout.log", logs / "stderr.log")
        record_invocation_time(f"generated-{index}", execution["elapsed_seconds"])
        stderr_tail = (logs / "stderr.log").read_text(encoding="utf-8", errors="replace")[-4000:]
        assert execution["timed_out"] is False, (
            "the generated-instance invocation exceeded the 3600-second solver limit")
        assert execution["cleanup_ok"] is True, (
            "the generated-instance invocation left an unprivileged process running")
        assert execution["returncode"] == 0, (
            f"the candidate returned {execution['returncode']} on a generated instance set; "
            f"required status is 0. Solver stderr tail:\n{stderr_tail}")
        violations = harness.scan_output_directory(staged_output)
        assert violations == [], (
            f"the generated output directory contains rejected entries: {violations}")
        candidate = read_json(staged_output / "solution.json", expected_directory=staged_output)
        validate_solution(document, candidate, {0: optimum_size})
    # Leave no scratch behind once every verifier invocation has finished.
    clear_other_invocation_areas()


def test_aggregate_wall_clock_budget_is_enforced():
    """The combined wall-clock limit covers every invocation the verifier runs.

    The contract requires the solver to complete every invocation - the public
    call, the unseen call, the additional packaged set and the two
    generated-input calls - inside a combined 14,400-second wall-clock limit.
    This test sums the measured wall time of all five invocations and fails the
    run when the total exceeds that limit, so individually admissible slow runs
    cannot pass together.
    """
    total = 0.0
    for split in SPLITS:
        status = read_json(Path("/root/status") / f"{split}.json")
        total += float(status["elapsed_seconds"])
    for name in ["derivation"] + [f"generated-{index}" for index in range(len(GENERATED_SIZES))]:
        path = Path("/root/status") / f"verifier_{name}.json"
        if not path.is_file():
            # The invocation failed its own assertion before recording its time;
            # the run is already failing and there is nothing to add here.
            continue
        total += float(read_json(path)["elapsed_seconds"])
    assert total <= 14400.0, (
        f"the verifier-run invocations took {total:.1f} s in total; the combined wall-clock limit is 14,400 s")
