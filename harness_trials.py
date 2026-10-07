"""Harness test: MiniMax Code fixes three small failing projects, driven headless against the model on port 8001.

Usage: python3 trials.py <label> [--passes 2]
For each task it makes a fresh copy of the project, runs `mcode-qwen exec` in it with full permissions, then runs the
project's tests itself. A trial passes only if the tests pass and the test files are unchanged.
Writes results/harness-<label>.json next to the other results.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RES = HERE.parent / "results"
MCODE = Path.home() / ".local/bin/mcode-qwen"
PROMPT = ("The tests in this project are failing. Find the cause and fix the code so that "
          "`python3 -m unittest discover -s tests` passes. Do not change anything in the tests folder. "
          "When the tests pass, stop and reply DONE.")

TASKS = {
    "intervals": {  # one file, two small bugs
        "intervals.py": '''"""Interval helpers."""


def merge_intervals(intervals):
    """Merge overlapping closed intervals. Touching intervals such as [1, 2] and [2, 3] also merge."""
    merged = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return merged
''',
        "tests/test_intervals.py": '''import unittest
from intervals import merge_intervals


class TestMerge(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(merge_intervals([[1, 3], [2, 6], [8, 10]]), [[1, 6], [8, 10]])

    def test_touching(self):
        self.assertEqual(merge_intervals([[1, 2], [2, 3]]), [[1, 3]])

    def test_unsorted(self):
        self.assertEqual(merge_intervals([[5, 7], [1, 10], [2, 3]]), [[1, 10]])

    def test_empty(self):
        self.assertEqual(merge_intervals([]), [])


if __name__ == "__main__":
    unittest.main()
''',
    },
    "durations": {  # a function to write from a written spec
        "README.md": '''# durations

`parse_duration(text)` turns a duration such as `1h30m15s` into a number of seconds.

- Units: `d` (days), `h` (hours), `m` (minutes), `s` (seconds). Any subset, in that order.
- Spaces between parts are allowed: `2h 5m`.
- A bare number with no unit is seconds: `90`.
- Empty text, unknown units, repeated units, or units out of order raise `ValueError`.
''',
        "durations.py": '''"""Duration parsing. See README.md for the rules."""


def parse_duration(text):
    raise NotImplementedError
''',
        "tests/test_durations.py": '''import unittest
from durations import parse_duration


class TestParse(unittest.TestCase):
    def test_full(self):
        self.assertEqual(parse_duration("1h30m15s"), 5415)

    def test_days_and_spaces(self):
        self.assertEqual(parse_duration("2d 3h"), 183600)
        self.assertEqual(parse_duration("2h 5m"), 7500)

    def test_bare_number(self):
        self.assertEqual(parse_duration("90"), 90)

    def test_single_units(self):
        self.assertEqual(parse_duration("45m"), 2700)
        self.assertEqual(parse_duration("7s"), 7)

    def test_errors(self):
        for bad in ("", "5x", "1h1h", "30m2h", "h", "1.5h"):
            with self.assertRaises(ValueError, msg=bad):
                parse_duration(bad)


if __name__ == "__main__":
    unittest.main()
''',
    },
    "inventory": {  # two bugs in two different files
        "inventory/__init__.py": "",
        "inventory/loader.py": '''"""Read stock rows from CSV text."""
import csv
import io


def load_rows(text):
    """Return a list of dicts with category (str), quantity (int) and price (float)."""
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        rows.append({"category": row["category"], "quantity": int(row["quantity"]), "price": float(row["price"])})
    return rows
''',
        "inventory/report.py": '''"""Summaries over stock rows."""
from collections import defaultdict


def total_by_category(rows):
    """Total stock value (quantity times price) per category, rounded to 2 places."""
    totals = defaultdict(float)
    for row in rows:
        totals[row["category"]] += row["price"]
    return {k: round(v, 2) for k, v in totals.items()}


def most_valuable(rows):
    """Name of the category with the highest total value, or None when there are no rows."""
    totals = total_by_category(rows)
    return max(totals, key=totals.get) if totals else None
''',
        "tests/test_inventory.py": '''import unittest
from inventory.loader import load_rows
from inventory.report import most_valuable, total_by_category

CSV = """category,quantity,price
tools,3,10.00
 tools ,2,5.50
paint,10,2.25
Paint,1,100.00
"""


class TestInventory(unittest.TestCase):
    def test_totals(self):
        self.assertEqual(total_by_category(load_rows(CSV)), {"tools": 41.0, "paint": 122.5})

    def test_most_valuable(self):
        self.assertEqual(most_valuable(load_rows(CSV)), "paint")

    def test_empty(self):
        self.assertIsNone(most_valuable(load_rows("category,quantity,price\\n")))


if __name__ == "__main__":
    unittest.main()
''',
    },
}


def make(task, dest):
    if dest.exists():
        shutil.rmtree(dest)
    for rel, body in TASKS[task].items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)


def tests_hash(repo):
    h = hashlib.sha256()
    for p in sorted((repo / "tests").rglob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def run_tests(repo):
    p = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=repo, capture_output=True, text=True, timeout=120)
    return p.returncode == 0, (p.stderr or p.stdout)[-300:]


def main():
    label = sys.argv[1]
    passes = int(sys.argv[sys.argv.index("--passes") + 1]) if "--passes" in sys.argv else 2
    out_path = RES / f"harness-{label}.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {"label": label, "trials": []}
    done = {(t["task"], t["pass_no"]) for t in res["trials"]}
    for n in range(1, passes + 1):
        for task in TASKS:
            if (task, n) in done:
                continue
            repo = HERE / "work" / f"{label}-{task}-{n}"
            make(task, repo)
            before_ok, _ = run_tests(repo)
            before = tests_hash(repo)
            log = HERE / "work" / f"{label}-{task}-{n}.log"
            t0 = time.time()
            try:
                p = subprocess.run([str(MCODE), "exec", "--cwd", str(repo), "--permission", "full", "--timeout", "8m",
                                    "--max-steps", "40", "--prompt-mode", "coding", PROMPT],
                                   capture_output=True, text=True, timeout=600)
                code, tail = p.returncode, (p.stdout + "\n" + p.stderr)
            except subprocess.TimeoutExpired:
                code, tail = -1, "timed out after 600 s"
            secs = time.time() - t0
            log.write_text(tail)
            ok, test_tail = run_tests(repo)
            untouched = tests_hash(repo) == before
            res["trials"].append({"task": task, "pass_no": n, "failing_before": not before_ok, "tests_pass": ok,
                                  "tests_untouched": untouched, "passed": ok and untouched, "seconds": round(secs, 1),
                                  "exit_code": code, "said_done": "DONE" in tail[-400:], "test_output": test_tail if not ok else ""})
            out_path.write_text(json.dumps(res, indent=2))
            print(f"{label} {task} #{n}: {'PASS' if ok and untouched else 'FAIL'} in {secs:.0f}s "
                  f"(tests pass {ok}, tests untouched {untouched}, exit {code})", flush=True)
    t = res["trials"]
    print(f"SUMMARY {label}: {sum(x['passed'] for x in t)} of {len(t)} trials passed, "
          f"total {sum(x['seconds'] for x in t) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
