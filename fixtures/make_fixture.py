#!/usr/bin/env python3
"""Writes support_tickets.jsonl, the sample log distill_check.py ships with.

A made-up support-ticket router: a frontier model (the teacher) picked a queue for
each ticket, the same small model answered each ticket twice, once before training
under a rewritten prompt (`prompted`) and once after training (`student`), and a
human-reviewed subset carries the right queue. The counts per slice are set here so
the correct verdict can be worked out by hand; the README states it.

    slice      calls  student disagrees  labeled  teacher wrong  student copied  prompted disagrees
    payments     420                  6      140              4               3                  68
    accounts     300                  3      100              2               2                   9
    shipping     260                  8       90             18              16                  39
    returns      150                 27       50              3               1                  45
    api           90                  7       30              1               1                  19
    legal         20                  0        0              0               0                   1

Deterministic: running it twice writes the same file.
"""
import json
import os

QUEUES = {
    "payments": ["billing", "refunds", "billing"],
    "accounts": ["account_security", "account_access"],
    "shipping": ["logistics", "carrier_claims"],
    "returns": ["refunds", "logistics", "returns_desk"],
    "api": ["developer_support", "incident_desk"],
    "legal": ["legal"],
}
TEMPLATES = {
    "payments": ["I was charged twice for order {n}", "My card was declined on invoice {n}",
                 "Please refund the duplicate charge on {n}"],
    "accounts": ["I cannot log in since the reset on ticket {n}", "Someone changed my email, case {n}"],
    "shipping": ["Parcel {n} shows delivered but never arrived", "The carrier lost package {n}"],
    "returns": ["I want to return order {n}, it arrived broken", "Return label for {n} does not work"],
    "api": ["Webhook for account {n} returns 500", "API key {n} gets rate limited at 10 rps"],
    "legal": ["Data deletion request under GDPR for account {n}"],
}
# calls, disagreements, labeled, teacher errors among labeled, copied among those
SPEC = {
    "payments": (420, 6, 140, 4, 3),
    "accounts": (300, 3, 100, 2, 2),
    "shipping": (260, 8, 90, 18, 16),
    "returns": (150, 27, 50, 3, 1),
    "api": (90, 7, 30, 1, 1),
    "legal": (20, 0, 0, 0, 0),
}
# calls where the prompted small model disagrees with the teacher, the last ones of each slice
PROMPTED_DISAGREE = {"payments": 68, "accounts": 9, "shipping": 39, "returns": 45, "api": 19,
                     "legal": 1}


def other(queue, pool):
    for q in pool + ["escalations"]:
        if q != queue:
            return q
    return "escalations"


def rows():
    n = 10000
    for product, (calls, disagree, labeled, errors, copied) in SPEC.items():
        pool = QUEUES[product]
        not_copied = errors - copied          # student disagrees with the teacher on these
        extra = disagree - not_copied         # further disagreements on unlabeled rows
        assert extra >= 0 and labeled <= calls and extra <= calls - labeled
        for i in range(calls):
            n += 1
            teacher = pool[i % len(pool)]
            row = {"id": f"t-{n}", "product": product,
                   "prompt": TEMPLATES[product][i % len(TEMPLATES[product])].format(n=n),
                   "answer": teacher}
            student = teacher
            if i < labeled:
                if i < errors:                          # the teacher got this ticket wrong
                    row["truth"] = other(teacher, pool)
                    if i >= copied:                     # the student did not copy the error
                        student = row["truth"]
                else:
                    row["truth"] = teacher
            elif i - labeled < extra:                   # the student disagrees, no label here
                student = other(teacher, pool)
            row["student"] = student
            prompted_off = i >= calls - PROMPTED_DISAGREE[product]
            row["prompted"] = other(teacher, pool) if prompted_off else teacher
            yield row


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "support_tickets.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows():
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {out}")
