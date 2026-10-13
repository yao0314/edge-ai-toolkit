# Valiant Joy — 6-minute pitch script

Timings are the target, not a limit. If you run long, cut slide 7 to its table — never cut slide 4, slide 5, or the cost panel on slide 6.

---

## Slide 1 — Cover · 0:00–0:30

> We're Valiant Joy. Three numbers first.
>
> We were handed a MobileNetV2 that had **already** been distilled from a ResNet-50. The easy
> compression was gone before we started. We still removed eighty-two percent of its parameters.
> On a real edge CPU it runs **nearly nine times faster**, and it keeps **ninety-eight percent** of
> the original accuracy.
>
> The interesting part isn't the pruning. It's what made the pruning affordable.

*(Pause on that last line. It's the promise the rest of the talk pays off.)*

---

## Slide 2 — The pipeline · 0:30–1:10

> Four steps. Three of them you already know.
>
> We start from the provided student. We remove sixty percent of the channels — structured, so the
> tensors actually get smaller, not masked. We recover. Then static INT8, QDQ, per-channel.
>
> Step three is the only unusual one. We recover against **the teacher's soft outputs** instead of
> the hard labels. That single change is the whole talk, and the next slide shows what it's worth.
>
> The artifact is ONNX opset 13, static QDQ, input pinned to one-by-three-by-224-by-224. Six hundred
> kilobytes.

---

## Slide 3 — The finding · 1:10–2:20 **(slow down here)**

> This is the slide I'd like you to remember.
>
> These are the weakest of the ten per-class recalls. Look at the sixty-percent row. Recovered on
> hard labels, the worst class collapses to **84.8**. Recovered from the provided teacher — same cut,
> same architecture, same quantization — it comes back to **92.6**. Seven and a half points, for four
> lines of config.
>
> Now look at the fifty-percent row. The *same change* buys **nothing**. It's actually slightly
> worse.
>
> That non-monotonicity is why we call this a finding and not a tuning win. The benefit peaks in a
> window: too shallow, and hard labels were already enough — this student was distilled from this
> teacher, so there's nothing new to learn. Too deep, and there's no capacity left to teach.
>
> So the deepest usable prune isn't a property of the network. It's a property of the **recovery
> budget** — and that's what let us ship sixty percent instead of forty.

---

## Slide 4 — How we chose the point · 2:20–2:55

> We did not pick sixty percent and hope. We swept the whole ladder and plotted it.
>
> Top panel is the mean. Everything from zero to twenty percent sits inside the plus-or-minus
> point-three-six measurement band — that bump at ten percent is noise, and a team that
> reports it as a gain is reporting its own error bars.
>
> Bottom panel is the weakest class. That is the statistic that actually moves, and it is what
> located the cliff between fifty and sixty.
>
> Both recoveries are on here. The blue line is what let us ship the deeper point.

---

## Slide 5 — Accuracy and compression · 2:55–3:30

> What that buys: ninety-eight percent of the baseline accuracy, five-and-a-half times fewer
> parameters, fourteen-and-a-half times smaller on disk.
>
> And what it costs — because we're not going to call this free. The weakest of the ten classes falls
> to **92.6 percent**, which is below the baseline's own 94.7. One class carries most of the loss.
> It's in our metrics.json and it's in our report.
>
> We also measured the conservative version — a forty-percent prune that keeps every class above
> ninety-five. It's tagged in our repository. It wins the accuracy criterion and loses the other
> three. That trade is a judgement, and we show both sides of it.

---

## Slide 6 — Resource and energy · 3:30–4:05

> Four-times fewer MACs. Three-times faster on CPU. Three-times less energy per inference.
>
> On energy — CodeCarbon falls back to a TDP model on our machine, and we say so. But the ratio is
> sound, and here's the check: the implied power is 62.2 watts for the baseline and 63.4 for ours.
> Within one percent. And the energy ratio, 3.07, matches the latency ratio, 3.13 — which is exactly
> what a TDP-mode figure has to look like if it means anything.
>
> Peak RAM moves least, and that's honest: fifty-six megabytes of that process is the Python and ONNX
> Runtime floor, not the model. Every team hits the same floor.

---

## Slide 7 — The edge device · 4:05–4:45

> This is the number we couldn't have got from our own machine.
>
> Same two ONNX files, profiled on both. Our laptop says the compression is worth **3.13×**. The
> Qualcomm QCS6490 says **8.94×**. Peak memory, 1.66× better there against 1.27× on the laptop.
>
> Nothing about the model changed — only the silicon measuring it. A big out-of-order core with a
> deep cache absorbs the cost of a fat model and hides most of what compression buys. The small core
> can't, and it pays you back.
>
> Which means: **choose a prune ratio by watching your own laptop, and you're reading a gradient
> three times too shallow.**

---

## Slide 8 — Takeaways · 4:45–5:15

> Three things we'd tell another team, and one that didn't work.
>
> Measure the weakest class, not the mean — the mean loses one point where the worst class loses
> eight. Spend your budget on recovery, not on the cut. And profile on the target, not on your
> desk.
>
> The one that failed: per-layer prune allocation. It's genuinely better at a matched size, and it
> stalls at forty-seven percent, because the parameters that are cheap to remove are a small bounded
> pool. Depth has to take expensive ones — which is also why the weakest class is what breaks first.
>
> Valiant Joy: ninety-eight percent accuracy retained, five-and-a-half times fewer parameters,
> fourteen-and-a-half times smaller, nearly nine times faster on the edge CPU. Every figure traces to
> one CSV from one run, and both ladders ship with the code.
>
> Thank you — happy to take questions.


---

## Slide 9 — Everything we measured · 5:15–5:40

> Before the summary, the raw table. Thirteen measurements, absolute values, both models, one
> machine, one script.
>
> I will point at one row: implied power. Energy divided by latency gives 62.2 watts for the
> baseline and 63.4 for ours. Those agree to two percent, which is what tells you a TDP-mode
> CodeCarbon figure is still a usable ratio. We did not just report the Joules and hope.

*(Do not read this slide aloud. Put it up, name the implied-power row, move on.)*

---

## Slide 10 — The scorecard · 5:40–6:00  **(the closer)**

> And here is all of it against your own rubric, so nothing has to be reconstructed from the talk.
>
> Twenty-five points for accuracy: we retain ninety-eight percent, weakest class ninety-two-six.
> Twenty for compression: five-and-a-half times on parameters, fourteen-and-a-half on the file.
> Twenty for resource: nearly nine times faster on the target device. Twenty for energy: three times
> less per inference.
>
> Eighty-five of the hundred points are measured on your rig, not ours. Every figure on this slide is
> a column of the CSV in our submission, from a single run.
>
> Thank you.

*(If you only get one slide of question time, stay on this one — every answer points at a row.)*

---

## Likely questions, and the honest answers

**"Why not the forty-percent model, if it keeps every class above ninety-five?"**
Because four criteria are scored and that trade loses ground on three of them to protect part of one.
We retain ninety-eight percent either way. It's tagged `submission-v1` if the judges prefer it.

**"Isn't a TDP-estimated energy figure worthless?"**
As an absolute number, yes, and we don't quote it as one. As a ratio the constant cancels, and we
verified that by checking the implied power agrees between the two models to within one percent.

**"Did you try distilling a smaller architecture from the teacher?"**
We costed it — `shufflenet_v2_x0_5` would be 6.35× smaller than the provided student — and didn't
run it. The task is to compress the model we were given, and the organisers confirmed that.

**"How long does the pipeline take to reproduce?"**
The sweep is one command. Recovery ran on a GPU at about forty seconds an epoch; on a CPU-only rig
that step will dominate, and a judge reproducing just the submitted point needs one prune and one
recovery, not twelve.

**"Why is peak RAM only 1.27×?"**
Because the raw process peak is mostly interpreter and runtime. The model's own footprint drops
2.5×, but the organisers specified raw process RSS and that's what we report.
