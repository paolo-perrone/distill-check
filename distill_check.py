#!/usr/bin/env python3
"""distill_check: which slices of your logged model calls could a distilled student take over?

    python3 distill_check.py calls.jsonl --slice product
    python3 distill_check.py calls.jsonl --slice product --prompted prompted
    python3 distill_check.py calls.jsonl --slice product --student student --prompted prompted
    python3 distill_check.py calls.jsonl --slice product --student student --truth label --markdown
    python3 distill_check.py --self-test

It reads the model calls your logging tool already exports, one JSON object per
line (a JSON array works too), and groups them by a slice field you name.

Before training (no --student), it says per slice whether you logged enough calls
to train a student and still hold out enough of them to test it.

With --prompted FIELD (the small model's answer before any training, under the best
prompt you have), it scores the prompt first: a slice whose prompted agreement
clears the floor routes to PROMPT, because it needs no training run at all.

After training (--student FIELD holds the student's answer to the same prompt), it
scores how often the student gave the teacher's answer, with a Wilson confidence
range, and routes each slice: STUDENT when the range's lower bound clears the
floor, TEACHER otherwise. With --prompted as well, it shows how many points of
agreement training added over the prompt on each slice.

Where rows carry a known right answer (--truth FIELD), it also scores the teacher
and counts the teacher's mistakes the student copied, because agreement with a
wrong teacher is still a wrong answer.

Standard library only. No GPU, no API key, nothing leaves the machine.
Exit code 0 on a completed report, 2 on a usage or input error.
"""
import argparse
import json
import math
import os
import statistics
import sys

VERSION = "0.2.0"

# OpenAI's supervised fine-tuning guide: "We recommend starting with 50 well-crafted
# demonstrations and evaluating the results." Applied per slice, because the student
# has to learn each slice from that slice's own examples.
DEFAULT_MIN_TRAIN = 50
DEFAULT_FLOOR = 0.90
DEFAULT_CONFIDENCE = 0.95


# ---------------------------------------------------------------- statistics

def z_for(confidence):
    """Two-sided normal quantile: 1.96 for 95%."""
    return statistics.NormalDist().inv_cdf(0.5 + confidence / 2)


def wilson(k, n, confidence=DEFAULT_CONFIDENCE):
    """Wilson score interval for k successes in n trials, as (low, high)."""
    if n == 0:
        return (0.0, 1.0)
    z = z_for(confidence)
    z2 = z * z
    p = k / n
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def needed_to_prove(floor, confidence=DEFAULT_CONFIDENCE):
    """Fewest compared calls whose Wilson lower bound reaches `floor` even if every
    one agrees. With k = n the lower bound is n / (n + z^2), so n >= floor * z^2 / (1 - floor):
    35 calls for a 90% floor at 95% confidence."""
    z2 = z_for(confidence) ** 2
    n = max(1, math.ceil(floor * z2 / (1 - floor)) - 1)
    while wilson(n, n, confidence)[0] < floor:
        n += 1
    return n


# ---------------------------------------------------------------- reading the log

def get_path(obj, path):
    """Read a field by dotted path: `output`, `response.choices.0.message.content`."""
    if isinstance(obj, dict) and path in obj:
        return obj[path]
    cur = obj
    for part in path.split("."):
        if isinstance(cur, dict):
            if part not in cur:
                return None
            cur = cur[part]
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def normalize(value, mode):
    """The form two answers are compared in. `normalized`, the default, ignores case,
    whitespace runs and surrounding quotes or a trailing period, and compares JSON
    objects by content; `json` parses both sides as JSON; `exact` compares raw text."""
    if value is None:
        return None
    if not isinstance(value, str):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if mode == "exact":
        return value
    s = value.strip()
    if mode == "json" or s[:1] in "{[":
        try:
            return json.dumps(json.loads(s), sort_keys=True, separators=(",", ":"))
        except ValueError:
            pass
    s = " ".join(s.split()).casefold()
    return s.strip("\"'`").rstrip(".").strip()


def read_calls(path):
    """Rows as (line number, object) plus the line numbers that were not JSON."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    if text.lstrip().startswith("["):
        data = json.loads(text)
        return [(i, row) for i, row in enumerate(data, 1)], []
    rows, bad = [], []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append((i, json.loads(line)))
        except ValueError:
            bad.append(i)
    return rows, bad


def present(v):
    return v is not None and v != ""


# ---------------------------------------------------------------- the check

def tally(rows, slice_f, answer_f, student_f=None, truth_f=None, match="normalized", where=(),
          prompted_f=None):
    """Per-slice counts, in first-seen order, plus the rows skipped and why.
    `where` holds (field, value) pairs a row must match to count, such as
    ("split", "holdout"), so a student is only scored on calls it did not train on.
    `prompted_f` holds the small model's answer before training, under a prompt."""
    slices, skipped = {}, {"no slice field": 0, "no teacher answer": 0, "a --where mismatch": 0}
    for _ln, obj in rows:
        if any(str(get_path(obj, f)) != v for f, v in where):
            skipped["a --where mismatch"] += 1
            continue
        s, a = get_path(obj, slice_f), get_path(obj, answer_f)
        if not present(s):
            skipped["no slice field"] += 1
            continue
        if not present(a):
            skipped["no teacher answer"] += 1
            continue
        key = s if isinstance(s, str) else json.dumps(s, sort_keys=True)
        st = slices.setdefault(key, {"calls": 0, "compared": 0, "agree": 0,
                                     "labeled": 0, "teacher_right": 0,
                                     "teacher_wrong_compared": 0, "copied": 0,
                                     "prompted_compared": 0, "prompted_agree": 0})
        st["calls"] += 1
        na = normalize(a, match)
        if prompted_f:
            pa = get_path(obj, prompted_f)
            if present(pa):
                st["prompted_compared"] += 1
                st["prompted_agree"] += int(normalize(pa, match) == na)
        nt = None
        if truth_f:
            t = get_path(obj, truth_f)
            if present(t):
                nt = normalize(t, match)
                st["labeled"] += 1
                st["teacher_right"] += int(na == nt)
        if student_f:
            sa = get_path(obj, student_f)
            if present(sa):
                ns = normalize(sa, match)
                st["compared"] += 1
                st["agree"] += int(ns == na)
                if nt is not None and na != nt:
                    st["teacher_wrong_compared"] += 1
                    st["copied"] += int(ns == na)
    return slices, skipped


def judge(slices, after, floor=DEFAULT_FLOOR, confidence=DEFAULT_CONFIDENCE,
          min_train=DEFAULT_MIN_TRAIN, prompted=False):
    """Attach a route (after training) or a readiness verdict (before) to each slice.
    With `prompted`, a slice whose prompted answers clear the floor routes to PROMPT
    first: the small model already does the job there without a training run."""
    need_eval = needed_to_prove(floor, confidence)
    need_total = min_train + need_eval
    out = []
    for name, st in slices.items():
        r = dict(st, slice=name)
        if st["labeled"]:
            r["teacher_range"] = wilson(st["teacher_right"], st["labeled"], confidence)
            r["teacher_flag"] = r["teacher_range"][1] < floor
        else:
            r["teacher_range"], r["teacher_flag"] = None, False
        pn, pk = st["prompted_compared"], st["prompted_agree"]
        r["prompted_range"] = wilson(pk, pn, confidence) if prompted and pn else None
        n, k = st["compared"], st["agree"]
        if after:
            r["range"] = wilson(k, n, confidence) if n else None
        if r["prompted_range"] and pn >= need_eval and r["prompted_range"][0] >= floor:
            r["route"], r["why"] = "PROMPT", ""
        elif after:
            if n == 0:
                r["route"], r["why"] = "TEACHER", "no student answers"
            elif n < need_eval:
                r["route"], r["why"] = "TEACHER", f"too few to judge, needs {need_eval}"
            elif r["range"][0] >= floor:
                r["route"], r["why"] = "STUDENT", ""
            else:
                r["route"], r["why"] = "TEACHER", f"low end {r['range'][0]:.1%} under {floor:.0%}"
        else:
            short = need_total - st["calls"]
            r["route"] = "ready" if short <= 0 else "too few"
            r["why"] = "" if short <= 0 else f"{short} short of {need_total}"
        out.append(r)
    return out, need_eval, need_total


# ---------------------------------------------------------------- output

def pct(x):
    return f"{x * 100:.1f}%"


def span(rng):
    return f"{rng[0] * 100:.1f}-{rng[1] * 100:.1f}%"


def teacher_cell(r):
    if not r["labeled"]:
        return "no labels"
    cell = f"{r['teacher_right']}/{r['labeled']}"
    if r["teacher_flag"]:
        cell += " LOW"
    return cell


def copied_cell(r):
    if not r["labeled"] or not r["compared"]:
        return "-"
    return f"{r['copied']} of {r['teacher_wrong_compared']}"


def prompted_cell(r):
    n = r["prompted_compared"]
    return pct(r["prompted_agree"] / n) if n else "-"


def gain_cell(r):
    """Points of agreement training added over the prompt: what distillation bought."""
    n, pn = r["compared"], r["prompted_compared"]
    if not n or not pn:
        return "-"
    return f"{(r['agree'] / n - r['prompted_agree'] / pn) * 100:+.1f}"


def route_cell(r):
    route = r["route"] + (f" ({r['why']})" if r["why"] else "")
    if r["route"] in ("STUDENT", "PROMPT") and r["teacher_flag"]:
        route += " *"
    return route


def table_rows(results, after, prompted=False):
    if after:
        head = ["slice", "calls", "agree", "agreement", "95% range"]
        if prompted:
            head += ["prompted", "training adds"]
        head += ["teacher right", "copied errors", "route"]
        body = []
        for r in results:
            n = r["compared"]
            row = [r["slice"], f"{r['calls']:,}", f"{r['agree']:,}",
                   pct(r["agree"] / n) if n else "-",
                   span(r["range"]) if r["range"] else "-"]
            if prompted:
                row += [prompted_cell(r), gain_cell(r)]
            body.append(row + [teacher_cell(r), copied_cell(r), route_cell(r)])
    elif prompted:
        head = ["slice", "calls", "teacher right", "prompted", "95% range", "next step"]
        step = {"PROMPT": "PROMPT", "ready": "distill"}
        body = [[r["slice"], f"{r['calls']:,}", teacher_cell(r), prompted_cell(r),
                 span(r["prompted_range"]) if r["prompted_range"] else "-",
                 step.get(r["route"], f"too few to distill, {r['why']}")] for r in results]
    else:
        head = ["slice", "calls", "teacher right", "ready to distill"]
        body = [[r["slice"], f"{r['calls']:,}", teacher_cell(r),
                 "yes" if r["route"] == "ready" else f"no, {r['why']}"] for r in results]
    return head, body


def verdict_line(results, after, prompted=False):
    total = sum(r["calls"] for r in results)
    share = lambda rs: sum(r["calls"] for r in rs)
    if prompted:
        by_prompt = [r for r in results if r["route"] == "PROMPT"]
        if after:
            by_student = [r for r in results if r["route"] == "STUDENT"]
            calls = share(by_prompt) + share(by_student)
            return (f"student takes {len(by_student)} of {len(results)} slices and a prompt "
                    f"alone takes {len(by_prompt)}: {calls:,} of {total:,} calls "
                    f"({calls / total:.0%}) leave the teacher, "
                    f"{share(by_prompt):,} of them with no training")
        ready = [r for r in results if r["route"] == "ready"]
        return (f"a prompt alone takes {len(by_prompt)} of {len(results)} slices, "
                f"{share(by_prompt):,} of {total:,} calls ({share(by_prompt) / total:.0%}); "
                f"{len(ready)} more have enough calls to distill")
    if after:
        took = [r for r in results if r["route"] == "STUDENT"]
        calls = sum(r["calls"] for r in took)
        return (f"student takes {len(took)} of {len(results)} slices, "
                f"{calls:,} of {total:,} calls ({calls / total:.0%})")
    ready = [r for r in results if r["route"] == "ready"]
    calls = sum(r["calls"] for r in ready)
    return (f"{len(ready)} of {len(results)} slices have enough calls to distill, "
            f"{calls:,} of {total:,} calls ({calls / total:.0%})")


def notes_lines(results, after):
    lines = []
    for r in results:
        if r["teacher_flag"]:
            lo, hi = r["teacher_range"]
            line = (f"{r['slice']}: the teacher was right on {r['teacher_right']} of "
                    f"{r['labeled']} labeled calls ({span((lo, hi))} at 95%)")
            if after and r["teacher_wrong_compared"]:
                line += (f", and the student copied {r['copied']} of the "
                         f"{r['teacher_wrong_compared']} teacher errors it was compared on")
            lines.append(line + "; fix the teacher's answers on this slice before trusting either model.")
    return lines


def render_text(path, results, after, skipped, bad, slice_f, floor, confidence,
                need_eval, need_total, min_train, prompted=False):
    total = sum(r["calls"] for r in results)
    out = [f"distill_check {VERSION}  {path}"]
    out.append(f"read     {total:,} logged calls with a teacher answer, "
               f"{len(results)} slices by \"{slice_f}\"")
    if prompted:
        out.append(f"prompted answers on {sum(r['prompted_compared'] for r in results):,} calls "
                   f"from the small model before training")
    if after:
        compared = sum(r["compared"] for r in results)
        labeled = sum(r["labeled"] for r in results)
        out.append(f"student  answers on {compared:,} calls; known right answers on {labeled:,}")
        out.append(f"rule     student takes a slice when agreement is at least {floor:.0%} at "
                   f"{confidence:.0%} confidence, which needs {need_eval}+ compared calls")
    else:
        out.append(f"rule     a slice is ready with {need_total}+ calls: {min_train} to train "
                   f"and {need_eval} held out, the fewest that can show {floor:.0%} agreement "
                   f"at {confidence:.0%} confidence")
    if prompted:
        out.append(f"         a prompt alone takes a slice first when its answers clear the same "
                   f"bar: no training run")
    for why, n in skipped.items():
        if n:
            out.append(f"skipped  {n:,} calls with {why}")
    if bad:
        out.append(f"skipped  {len(bad):,} lines that are not JSON (first: line {bad[0]})")
    head, body = table_rows(results, after, prompted)
    widths = [max(len(str(x)) for x in col) for col in zip(head, *body)]
    fmt = lambda row: "  ".join(str(c).ljust(w) if i in (0, len(row) - 1) else str(c).rjust(w)
                                 for i, (c, w) in enumerate(zip(row, widths))).rstrip()
    out.append("")
    out.append(fmt(head))
    out.extend(fmt(row) for row in body)
    out.append("")
    out.append("verdict  " + verdict_line(results, after, prompted))
    notes = notes_lines(results, after)
    if notes:
        out.append("")
        if not after:
            out.append("the teacher itself is below the floor on labeled calls here:")
        elif prompted:
            out.append("* routed off the teacher, and the teacher itself is below the floor there:")
        else:
            out.append("* routed to the student, and the teacher itself is below the floor there:")
        out.extend("  " + n for n in notes)
    return "\n".join(out)


def render_markdown(results, after, prompted=False):
    head, body = table_rows(results, after, prompted)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in body]
    lines += ["", verdict_line(results, after, prompted) + "."]
    notes = notes_lines(results, after)
    if notes:
        lines += [""] + [f"- {n}" for n in notes]
    return "\n".join(lines)


# ---------------------------------------------------------------- CLI

def run(argv=None, stdout=sys.stdout, stderr=sys.stderr):
    p = argparse.ArgumentParser(prog="distill_check.py", description=__doc__.split("\n\n")[0])
    p.add_argument("log", nargs="?", help="JSONL (or JSON array) of logged model calls")
    p.add_argument("--slice", dest="slice_f", help="field that splits traffic into slices, e.g. product")
    p.add_argument("--answer", default="answer", help="field holding the teacher's answer (default: answer)")
    p.add_argument("--student", help="field holding the student's answer to the same prompt")
    p.add_argument("--truth", help="field holding a known right answer, where you have one")
    p.add_argument("--prompted", help="field holding the small model's answer before training, "
                                      "under the best prompt you have")
    p.add_argument("--where", action="append", default=[], metavar="FIELD=VALUE",
                   help="only count calls whose FIELD equals VALUE, e.g. split=holdout; repeatable")
    p.add_argument("--match", choices=["normalized", "json", "exact"], default="normalized",
                   help="how two answers are compared (default: normalized)")
    p.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                   help="agreement the student must reach, as a fraction (default: 0.90)")
    p.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE,
                   help="confidence for the range (default: 0.95)")
    p.add_argument("--min-train", type=int, default=DEFAULT_MIN_TRAIN,
                   help="calls a slice needs to train on (default: 50)")
    p.add_argument("--markdown", action="store_true", help="print a markdown table for a design doc")
    p.add_argument("--json", action="store_true", help="print the numbers as JSON")
    p.add_argument("--self-test", action="store_true", help="run the known-answer checks and exit")
    p.add_argument("--version", action="version", version=VERSION)
    a = p.parse_args(argv)

    if a.self_test:
        return self_test(stdout)
    if not a.log or not a.slice_f:
        p.print_usage(stderr)
        print("distill_check.py: a log file and --slice are required", file=stderr)
        return 2
    if not (0 < a.floor < 1 and 0 < a.confidence < 1):
        print("distill_check.py: --floor and --confidence must be between 0 and 1", file=stderr)
        return 2
    if not os.path.exists(a.log):
        print(f"distill_check.py: no such file: {a.log}", file=stderr)
        return 2
    try:
        rows, bad = read_calls(a.log)
    except ValueError as e:
        print(f"distill_check.py: {a.log} is not JSONL or a JSON array: {e}", file=stderr)
        return 2
    where = []
    for w in a.where:
        if "=" not in w:
            print(f"distill_check.py: --where needs FIELD=VALUE, got {w!r}", file=stderr)
            return 2
        f, v = w.split("=", 1)
        where.append((f, v))
    slices, skipped = tally(rows, a.slice_f, a.answer, a.student, a.truth, a.match, where,
                            a.prompted)
    if not slices:
        print(f"distill_check.py: no call in {a.log} has both \"{a.slice_f}\" and "
              f"\"{a.answer}\"; check the field names (dotted paths work: "
              f"response.choices.0.message.content)", file=stderr)
        return 2
    after = bool(a.student)
    if after and not any(s["compared"] for s in slices.values()):
        print(f"distill_check.py: no call has a \"{a.student}\" field; drop --student to run "
              f"the before-training check", file=stderr)
        return 2
    prompted = bool(a.prompted)
    if prompted and not any(s["prompted_compared"] for s in slices.values()):
        print(f"distill_check.py: no call has a \"{a.prompted}\" field; drop --prompted, or log "
              f"the small model's answers under your prompt first", file=stderr)
        return 2
    results, need_eval, need_total = judge(slices, after, a.floor, a.confidence, a.min_train,
                                           prompted)
    if a.json:
        doc = {"version": VERSION, "phase": "after" if after else "before",
               "prompted": prompted,
               "floor": a.floor, "confidence": a.confidence, "min_train": a.min_train,
               "needed_to_judge": need_eval, "needed_per_slice": need_total,
               "verdict": verdict_line(results, after, prompted), "skipped": skipped,
               "slices": results}
        print(json.dumps(doc, indent=2), file=stdout)
    elif a.markdown:
        print(render_markdown(results, after, prompted), file=stdout)
    else:
        print(render_text(a.log, results, after, skipped, bad, a.slice_f, a.floor,
                          a.confidence, need_eval, need_total, a.min_train, prompted), file=stdout)
    return 0


def self_test(stdout=sys.stdout):
    """Known answers for the arithmetic and for the shipped sample log."""
    checks = []

    def ck(label, got, want):
        checks.append((label, got == want, got, want))

    ck("needed_to_prove(0.90) is 35 compared calls", needed_to_prove(0.90), 35)
    ck("needed_to_prove(0.95) is 73 compared calls", needed_to_prove(0.95), 73)
    ck("wilson(35, 35) low end clears 0.90", wilson(35, 35)[0] >= 0.90, True)
    ck("wilson(34, 34) low end misses 0.90", wilson(34, 34)[0] < 0.90, True)
    ck("wilson(0, 10) high end is 0.2775", round(wilson(0, 10)[1], 4), 0.2775)
    ck("normalize ignores case, spaces and a trailing period",
       normalize("  Billing. ", "normalized"), normalize("billing", "normalized"))
    ck("normalize compares JSON by content",
       normalize('{"b": 1, "a": 2}', "normalized"), normalize('{"a":2,"b":1}', "normalized"))
    ck("get_path reads lists by index",
       get_path({"r": {"c": [{"m": "x"}]}}, "r.c.0.m"), "x")

    here = os.path.dirname(os.path.abspath(__file__))
    fixture = os.path.join(here, "fixtures", "support_tickets.jsonl")
    if os.path.exists(fixture):
        rows, _bad = read_calls(fixture)
        slices, _ = tally(rows, "product", "answer", "student", "truth")
        res, _, _ = judge(slices, after=True)
        routes = {r["slice"]: r["route"] for r in res}
        ck("sample log routes", routes, {"payments": "STUDENT", "accounts": "STUDENT",
                                         "shipping": "STUDENT", "returns": "TEACHER",
                                         "api": "TEACHER", "legal": "TEACHER"})
        ck("sample log verdict", verdict_line(res, True),
           "student takes 3 of 6 slices, 980 of 1,240 calls (79%)")
        ck("sample log flags the shipping teacher",
           [r["slice"] for r in res if r["teacher_flag"]], ["shipping"])
        before, _, _ = judge(tally(rows, "product", "answer", None, "truth")[0], after=False)
        ck("sample log before training", verdict_line(before, False),
           "5 of 6 slices have enough calls to distill, 1,220 of 1,240 calls (98%)")
        slices, _ = tally(rows, "product", "answer", "student", "truth", prompted_f="prompted")
        res, _, _ = judge(slices, after=True, prompted=True)
        ck("sample log routes with the prompt scored first",
           {r["slice"]: r["route"] for r in res},
           {"payments": "STUDENT", "accounts": "PROMPT", "shipping": "STUDENT",
            "returns": "TEACHER", "api": "TEACHER", "legal": "TEACHER"})
        ck("sample log verdict with the prompt scored first", verdict_line(res, True, True),
           "student takes 2 of 6 slices and a prompt alone takes 1: 980 of 1,240 calls (79%) "
           "leave the teacher, 300 of them with no training")
        ck("sample log: training adds 14.8 points over the prompt on payments",
           gain_cell(res[0]), "+14.8")
    failed = [c for c in checks if not c[1]]
    for label, ok, got, want in checks:
        print(f"{'ok  ' if ok else 'FAIL'} {label}" + ("" if ok else f": got {got!r}, want {want!r}"),
              file=stdout)
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed", file=stdout)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(run())
