# distill-check

For engineers paying a frontier model on every call of a narrow task, who want to
know which slices of that traffic a small distilled model could take over, before
and after training one.

Point it at the calls your logging tool already exports as JSONL, and name the
field that splits your traffic into slices, such as product area or endpoint:

```
python3 distill_check.py calls.jsonl --slice product
```

Before training it prints, per slice, how many calls you logged and whether that
is enough to train a student and still hold out enough to test it.

Then score a prompt before you train anything. Run the small model on the same
calls under the best prompt you can write for it, log its answers, and pass them
with `--prompted`: every slice where the prompt alone clears the bar routes to
PROMPT, and needs no training run. The small model's score on the big model's
prompt is not that number. In NVIDIA's NVInfo router (arXiv 2510.27051, Table III),
Llama 3.1 8B scored 14% on the prompt written for the 70B, 86% with prompt tuning
and 96% after fine-tuning, the same as the 70B. The paper does not say how the
prompt was tuned.

After training, add the student's answers with `--student` and it prints each
slice's agreement with the teacher, a 95% confidence range, how many points
training added over the prompt, and a route: prompt, student or teacher.

The real output on the sample log in `fixtures/`, a support-ticket router whose
teacher picked a queue for 1,240 tickets, and whose small model answered the same
tickets twice, once under a rewritten prompt and once after training, 2026-10-29:

```
python3 distill_check.py fixtures/support_tickets.jsonl --slice product --student student --truth truth --prompted prompted
```

```
distill_check 0.2.0  fixtures/support_tickets.jsonl
read     1,240 logged calls with a teacher answer, 6 slices by "product"
prompted answers on 1,240 calls from the small model before training
student  answers on 1,240 calls; known right answers on 410
rule     student takes a slice when agreement is at least 90% at 95% confidence, which needs 35+ compared calls
         a prompt alone takes a slice first when its answers clear the same bar: no training run

slice     calls  agree  agreement    95% range  prompted  training adds  teacher right  copied errors  route
payments    420    414      98.6%   96.9-99.3%     83.8%          +14.8        136/140         3 of 4  STUDENT
accounts    300    297      99.0%   97.1-99.7%     97.0%           +2.0         98/100         2 of 2  PROMPT
shipping    260    252      96.9%   94.0-98.4%     85.0%          +11.9      72/90 LOW       16 of 18  STUDENT *
returns     150    123      82.0%   75.1-87.3%     70.0%          +12.0          47/50         1 of 3  TEACHER (low end 75.1% under 90%)
api          90     83      92.2%   84.8-96.2%     78.9%          +13.3          29/30         1 of 1  TEACHER (low end 84.8% under 90%)
legal        20     20     100.0%  83.9-100.0%     95.0%           +5.0      no labels              -  TEACHER (too few to judge, needs 35)

verdict  student takes 2 of 6 slices and a prompt alone takes 1: 980 of 1,240 calls (79%) leave the teacher, 300 of them with no training

* routed off the teacher, and the teacher itself is below the floor there:
  shipping: the teacher was right on 72 of 90 labeled calls (70.6-87.0% at 95%), and the student copied 16 of the 18 teacher errors it was compared on; fix the teacher's answers on this slice before trusting either model.
```

Payments is where training paid: the prompt agreed with the teacher on 83.8% of
calls and the trained student on 98.6%, 14.8 points more. Accounts is where it did
not: the prompt already agreed on 97.0%, so that slice routes to PROMPT and the
2 points training added there are not worth a training run.

`--markdown` prints the same table ready to paste into a design doc, and `--json`
prints the numbers for a pipeline. Drop `--prompted` and the table loses its two
prompt columns and routes between student and teacher only.

## Check it before you trust it

The sample log is made up, and its answer is known before you run it:
`fixtures/make_fixture.py` sets how many calls each slice has, how many the
student and the prompted model disagree on, and how many of the teacher's labeled
answers are wrong. The correct verdict is **student takes 2 of 6 slices and a
prompt alone takes 1: 980 of 1,240 calls (79%) leave the teacher, 300 of them with
no training**, with shipping flagged; without `--prompted` it is **student takes 3
of 6 slices, 980 of 1,240 calls (79%)**. `python3 distill_check.py --self-test`
checks both along with the arithmetic below. Work one row by hand: payments has
414 agreements in 420 calls, 98.6%, and its Wilson range at 95% is 96.9% to 99.3%,
so the low end clears 90% and the slice goes to the student. The prompted model
agreed on 352 of the same 420, 83.8%, whose range of 80.0% to 87.0% misses 90%, so
the prompt alone does not take it.

## Get your log into shape

- **One call per line**, as JSON. A JSON array file works too.
- **Field names are yours.** `--answer` (default `answer`) holds the teacher's
  answer, `--student` the student's answer to the same prompt, `--prompted` the
  small model's answer before training under your prompt, `--truth` a known right
  answer where you have one, such as tickets an agent re-routed by hand.
  Dotted paths reach into nested exports:
  `--answer response.choices.0.message.content`.
- **Slice by something you know before the call**: product area, endpoint,
  language, customer tier. The routing table decides who answers before anyone
  has answered, so a slice built from the teacher's own answer cannot be used
  to route.
- **Score the student on calls it did not train on.** Agreement on training calls
  is inflated. Mark your held-out calls and pass `--where split=holdout`.
- **Answers are compared as values.** By default case, spacing, surrounding quotes
  and a trailing period are ignored, and JSON objects compare by content
  (`--match json` forces JSON, `--match exact` compares raw text). That suits
  labels, extracted fields and short answers. For free text, compare a field
  inside the answer, not the prose around it.

## The arithmetic, so you can argue with it

- **Agreement** is the share of compared calls where the student gave the
  teacher's answer.
- **The range** is the Wilson score interval at 95% (`--confidence`). For k
  agreements in n calls, with z = 1.96, its center is
  (k/n + z²/(2n)) / (1 + z²/n), and its half-width is
  z·√(p(1-p)/n + z²/(4n²)) / (1 + z²/n) with p = k/n.
- **The route** is PROMPT when the prompted model's range has a low end at the
  floor or above, on at least the 35 compared calls below: a prompt that already
  does the job needs no training run and nothing to retrain when the teacher
  changes. Otherwise it is STUDENT when the student's range's low end reaches the
  floor, 90% by default (`--floor`). A student that truly agrees exactly 90% of the time still
  clears it by luck on 0.5% to 2.5% of slices, by exact binomial arithmetic for 35
  to 400 compared calls; `tests/` checks the rate stays under 5%.
- **35 compared calls** is the fewest that can clear a 90% floor at 95%
  confidence even if every one agrees: with k = n the low end is n / (n + z²),
  which first reaches 0.90 at n = 35. Fewer, and the slice stays on the teacher.
- **85 calls per slice before training**: 50 to train on, the starting set
  OpenAI's supervised fine-tuning guide recommends ("We recommend starting with
  50 well-crafted demonstrations"), plus the 35 held out to test the result.
  Change the first with `--min-train`.
- **Teacher right** counts labeled calls where the teacher's answer matches the
  known right answer. **LOW** marks a slice where the high end of that range is
  under the floor: the teacher is wrong there too often to be worth copying.
- **Copied errors** counts labeled calls where the teacher was wrong and the
  student gave the same wrong answer. Agreement treats those as wins; this column
  does not.
- **Training adds** is the student's agreement minus the prompted model's, in
  percentage points, on the same slice. It is the part of the student's score
  that training bought; the prompt bought the rest.

## What it will not tell you

- Whether the student is right. Agreement measures copying; only `--truth`
  measures correctness, and only on the calls you labeled.
- Anything about traffic you never logged. A slice missing from the log is
  missing from the table, and a student meets it cold in production.
- Whether the slices will stay stable. Re-run it on each month's log.

## Options

```
--slice FIELD        field that splits traffic into slices (required)
--answer FIELD       teacher's answer (default: answer)
--student FIELD      student's answer; switches to the after-training check
--truth FIELD        known right answer, where you have one
--prompted FIELD     small model's answer before training, under your prompt
--where FIELD=VALUE  only count matching calls, e.g. split=holdout; repeatable
--match MODE         normalized (default), json or exact
--floor F            agreement the student must reach (default: 0.90)
--confidence C       confidence for the range (default: 0.95)
--min-train N        calls a slice needs to train on (default: 50)
--markdown / --json  output formats
--self-test          known-answer checks
```

Exit code 0 on a completed report, 2 on a usage or input error.

## Support

Standard library only: no dependencies, no GPU, no API key, and nothing leaves
your machine. Tested 2026-10-29 on Python 3.9 and 3.14 with
`python3 -m unittest discover -s tests`. Field layouts differ between logging
tools; when an export does not read, open an issue with one redacted line.

## Licence

MIT.
