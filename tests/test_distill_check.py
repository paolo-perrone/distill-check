"""Tests for distill_check.py. Standard library only: python3 -m unittest discover -s tests"""
import io
import json
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import distill_check as dc  # noqa: E402

FIXTURE = os.path.join(ROOT, "fixtures", "support_tickets.jsonl")


def run_cli(*args):
    out, err = io.StringIO(), io.StringIO()
    code = dc.run(list(args), stdout=out, stderr=err)
    return code, out.getvalue(), err.getvalue()


def write_jsonl(rows):
    fh = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in rows:
        fh.write(json.dumps(r) + "\n")
    fh.close()
    return fh.name


class Arithmetic(unittest.TestCase):
    def test_wilson_matches_hand_values(self):
        # 95%, z = 1.959964: for 0 of 10 the high end is z^2 / (n + z^2) = 3.8415 / 13.8415
        self.assertAlmostEqual(dc.wilson(0, 10)[1], 0.27753, places=4)
        # for n of n the low end is n / (n + z^2)
        self.assertAlmostEqual(dc.wilson(10, 10)[0], 10 / (10 + 3.841459), places=5)
        # 414 of 420, the sample log's payments slice
        lo, hi = dc.wilson(414, 420)
        self.assertAlmostEqual(lo, 0.9692, places=3)
        self.assertAlmostEqual(hi, 0.9934, places=3)

    def test_needed_to_prove(self):
        self.assertEqual(dc.needed_to_prove(0.90), 35)
        self.assertEqual(dc.needed_to_prove(0.95), 73)
        self.assertEqual(dc.needed_to_prove(0.80), 16)
        for floor in (0.8, 0.9, 0.95, 0.99):
            n = dc.needed_to_prove(floor)
            self.assertGreaterEqual(dc.wilson(n, n)[0], floor)
            self.assertLess(dc.wilson(n - 1, n - 1)[0], floor)

    def test_false_accept_rate_at_the_floor(self):
        """A student that truly agrees 90% of the time, judged against a 90% floor on
        200 calls, must be routed to the student rarely: the Wilson low end is a 95%
        two-sided bound, so about 2.5% of such slices clear it by luck."""
        rng = random.Random(7)
        trials, accepted = 4000, 0
        for _ in range(trials):
            k = sum(rng.random() < 0.90 for _ in range(200))
            accepted += dc.wilson(k, 200)[0] >= 0.90
        self.assertLess(accepted / trials, 0.05)

    def test_a_clearly_good_student_is_accepted(self):
        rng = random.Random(11)
        trials, accepted = 2000, 0
        for _ in range(trials):
            k = sum(rng.random() < 0.97 for _ in range(300))
            accepted += dc.wilson(k, 300)[0] >= 0.90
        self.assertGreater(accepted / trials, 0.99)


class Parsing(unittest.TestCase):
    def test_dotted_paths(self):
        obj = {"response": {"choices": [{"message": {"content": "billing"}}]}, "a.b": 1}
        self.assertEqual(dc.get_path(obj, "response.choices.0.message.content"), "billing")
        self.assertEqual(dc.get_path(obj, "a.b"), 1)
        self.assertIsNone(dc.get_path(obj, "response.choices.3.message"))
        self.assertIsNone(dc.get_path(obj, "missing"))

    def test_normalize(self):
        n = lambda v: dc.normalize(v, "normalized")
        self.assertEqual(n("  Billing. "), n("billing"))
        self.assertEqual(n('"refunds"'), n("refunds"))
        self.assertEqual(n('{"queue": "billing", "p": 1}'), n({"p": 1, "queue": "billing"}))
        self.assertNotEqual(n("billing"), n("billings"))
        self.assertNotEqual(dc.normalize("Billing", "exact"), dc.normalize("billing", "exact"))

    def test_json_array_input(self):
        fh = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        json.dump([{"s": "a", "answer": "x"}, {"s": "a", "answer": "y"}], fh)
        fh.close()
        rows, bad = dc.read_calls(fh.name)
        self.assertEqual(len(rows), 2)
        self.assertEqual(bad, [])
        os.unlink(fh.name)

    def test_bad_lines_are_skipped_and_counted(self):
        path = write_jsonl([{"s": "a", "answer": "x"}])
        with open(path, "a") as fh:
            fh.write("not json\n")
        rows, bad = dc.read_calls(path)
        self.assertEqual((len(rows), bad), (1, [2]))
        os.unlink(path)


class Verdicts(unittest.TestCase):
    def rows(self, n, agree, slice_name="s", truth_wrong=0, copied=0):
        out = []
        for i in range(n):
            r = {"slice": slice_name, "answer": "a", "student": "a" if i < agree else "b"}
            if i < truth_wrong:
                r["truth"] = "c"
                if i >= copied:
                    r["student"] = "c"
            out.append(r)
        return out

    def test_route_student_when_low_end_clears_floor(self):
        slices, _ = dc.tally([(0, r) for r in self.rows(200, 198)], "slice", "answer", "student")
        res, _, _ = dc.judge(slices, after=True)
        self.assertEqual(res[0]["route"], "STUDENT")

    def test_route_teacher_when_low_end_misses_floor(self):
        slices, _ = dc.tally([(0, r) for r in self.rows(200, 182)], "slice", "answer", "student")
        res, _, _ = dc.judge(slices, after=True)
        self.assertEqual(res[0]["route"], "TEACHER")
        self.assertIn("under 90%", res[0]["why"])

    def test_too_few_compared_calls_stay_on_the_teacher(self):
        slices, _ = dc.tally([(0, r) for r in self.rows(34, 34)], "slice", "answer", "student")
        res, _, _ = dc.judge(slices, after=True)
        self.assertEqual(res[0]["route"], "TEACHER")
        self.assertIn("needs 35", res[0]["why"])

    def test_copied_errors_are_counted(self):
        rows = self.rows(100, 100, truth_wrong=10, copied=7)
        slices, _ = dc.tally([(0, r) for r in rows], "slice", "answer", "student", "truth")
        st = slices["s"]
        self.assertEqual((st["labeled"], st["teacher_right"]), (10, 0))
        self.assertEqual((st["teacher_wrong_compared"], st["copied"]), (10, 7))

    def test_before_training_readiness(self):
        rows = [(0, {"slice": "big", "answer": "a"}) for _ in range(85)]
        rows += [(0, {"slice": "small", "answer": "a"}) for _ in range(84)]
        slices, _ = dc.tally(rows, "slice", "answer")
        res, need_eval, need_total = dc.judge(slices, after=False)
        self.assertEqual((need_eval, need_total), (35, 85))
        self.assertEqual({r["slice"]: r["route"] for r in res}, {"big": "ready", "small": "too few"})

    def test_prompt_that_clears_the_floor_routes_to_prompt(self):
        rows = [(0, dict(r, prompted="a")) for r in self.rows(200, 199)]
        slices, _ = dc.tally(rows, "slice", "answer", "student", prompted_f="prompted")
        res, _, _ = dc.judge(slices, after=True, prompted=True)
        self.assertEqual(res[0]["route"], "PROMPT")
        self.assertEqual(dc.gain_cell(res[0]), "-0.5")

    def test_prompt_below_the_floor_leaves_the_student_route(self):
        rows = [(0, dict(r, prompted="a" if i < 170 else "b"))
                for i, r in enumerate(self.rows(200, 198))]
        slices, _ = dc.tally(rows, "slice", "answer", "student", prompted_f="prompted")
        res, _, _ = dc.judge(slices, after=True, prompted=True)
        self.assertEqual(res[0]["route"], "STUDENT")
        self.assertEqual(dc.gain_cell(res[0]), "+14.0")

    def test_prompt_needs_as_many_compared_calls_as_the_student(self):
        rows = [(0, {"slice": "s", "answer": "a", "prompted": "a"}) for _ in range(34)]
        slices, _ = dc.tally(rows, "slice", "answer", prompted_f="prompted")
        res, _, _ = dc.judge(slices, after=False, prompted=True)
        self.assertEqual(res[0]["route"], "too few")

    def test_prompt_scored_before_training(self):
        rows = [(0, {"slice": "easy", "answer": "a", "prompted": "a"}) for _ in range(100)]
        rows += [(0, {"slice": "hard", "answer": "a", "prompted": "a" if i < 60 else "b"})
                 for i in range(100)]
        slices, _ = dc.tally(rows, "slice", "answer", prompted_f="prompted")
        res, _, _ = dc.judge(slices, after=False, prompted=True)
        self.assertEqual({r["slice"]: r["route"] for r in res}, {"easy": "PROMPT", "hard": "ready"})
        self.assertEqual(dc.verdict_line(res, False, True),
                         "a prompt alone takes 1 of 2 slices, 100 of 200 calls (50%); "
                         "1 more have enough calls to distill")

    def test_where_filter_keeps_only_matching_calls(self):
        rows = [(0, {"slice": "s", "answer": "a", "student": "a", "split": "holdout"}),
                (0, {"slice": "s", "answer": "a", "student": "b", "split": "train"})]
        slices, skipped = dc.tally(rows, "slice", "answer", "student", where=[("split", "holdout")])
        self.assertEqual((slices["s"]["compared"], slices["s"]["agree"]), (1, 1))
        self.assertEqual(skipped["a --where mismatch"], 1)


class SampleLog(unittest.TestCase):
    """The README states this verdict; these tests hold the tool to it."""

    def test_after_training(self):
        code, out, _ = run_cli(FIXTURE, "--slice", "product", "--student", "student", "--truth", "truth")
        self.assertEqual(code, 0)
        self.assertIn("verdict  student takes 3 of 6 slices, 980 of 1,240 calls (79%)", out)
        self.assertIn("returns     150    123      82.0%   75.1-87.3%", out)
        self.assertIn("shipping: the teacher was right on 72 of 90 labeled calls", out)
        self.assertIn("the student copied 16 of the 18 teacher errors", out)

    def test_after_training_with_the_prompt_scored_first(self):
        code, out, _ = run_cli(FIXTURE, "--slice", "product", "--student", "student",
                               "--truth", "truth", "--prompted", "prompted")
        self.assertEqual(code, 0)
        self.assertIn("verdict  student takes 2 of 6 slices and a prompt alone takes 1: "
                      "980 of 1,240 calls (79%) leave the teacher, 300 of them with no training", out)
        self.assertIn("payments    420    414      98.6%   96.9-99.3%     83.8%          +14.8", out)
        self.assertIn("accounts    300    297      99.0%   97.1-99.7%     97.0%           +2.0", out)
        self.assertIn("PROMPT", out)

    def test_prompt_scored_before_training(self):
        code, out, _ = run_cli(FIXTURE, "--slice", "product", "--prompted", "prompted")
        self.assertEqual(code, 0)
        self.assertIn("verdict  a prompt alone takes 1 of 6 slices, 300 of 1,240 calls (24%); "
                      "4 more have enough calls to distill", out)
        self.assertIn("accounts    300      no labels     97.0%  94.4-98.4%  PROMPT", out)

    def test_before_training(self):
        code, out, _ = run_cli(FIXTURE, "--slice", "product")
        self.assertEqual(code, 0)
        self.assertIn("verdict  5 of 6 slices have enough calls to distill, 1,220 of 1,240 calls (98%)", out)
        self.assertIn("no, 65 short of 85", out)

    def test_markdown_and_json(self):
        code, out, _ = run_cli(FIXTURE, "--slice", "product", "--student", "student", "--markdown")
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("| slice | calls | agree |"))
        code, out, _ = run_cli(FIXTURE, "--slice", "product", "--student", "student", "--json")
        doc = json.loads(out)
        self.assertEqual(doc["needed_to_judge"], 35)
        self.assertEqual([s["route"] for s in doc["slices"]].count("STUDENT"), 3)
        code, out, _ = run_cli(FIXTURE, "--slice", "product", "--student", "student",
                               "--prompted", "prompted", "--json")
        doc = json.loads(out)
        self.assertTrue(doc["prompted"])
        self.assertEqual([s["route"] for s in doc["slices"]].count("PROMPT"), 1)

    def test_fixture_is_reproducible(self):
        sys.path.insert(0, os.path.join(ROOT, "fixtures"))
        import make_fixture
        with open(FIXTURE, encoding="utf-8") as fh:
            shipped = [json.loads(l) for l in fh if l.strip()]
        self.assertEqual(shipped, list(make_fixture.rows()))

    def test_self_test_passes(self):
        out = io.StringIO()
        self.assertEqual(dc.self_test(out), 0)


class Errors(unittest.TestCase):
    def test_missing_file(self):
        code, _, err = run_cli("nope.jsonl", "--slice", "product")
        self.assertEqual(code, 2)
        self.assertIn("no such file", err)

    def test_wrong_field_names(self):
        code, _, err = run_cli(FIXTURE, "--slice", "endpoint")
        self.assertEqual(code, 2)
        self.assertIn("check the field names", err)

    def test_prompted_field_absent(self):
        code, _, err = run_cli(FIXTURE, "--slice", "product", "--prompted", "small_model")
        self.assertEqual(code, 2)
        self.assertIn("drop --prompted", err)

    def test_student_field_absent(self):
        code, _, err = run_cli(FIXTURE, "--slice", "product", "--student", "small_model")
        self.assertEqual(code, 2)
        self.assertIn("drop --student", err)


if __name__ == "__main__":
    unittest.main()
