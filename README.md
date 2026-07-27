# V5 Data Mixture & Curriculum Specification

**Run:** 30B params × 2.00T sequence tokens · **Author:** Shekar · **Status:** hypothesis, unvalidated at scale

Every number below is reproducible by running [`mixture_ledger.py`](./mixture_ledger.py). If a number in this
document disagrees with the script, the script is right and this document is stale.

---

## 0. The six commitments

| # | Commitment | Number |
|---|---|---|
| 1 | Total budget | **2.00T sequence tokens**, of which **100B (5%) is held back for the anneal** |
| 2 | Headline mixture | web 31.7 / code 23.6 / Indic 16.9 / STEM 11.6 / reasoning 6.9 / long-ctx 5.9 / **agentic 3.45** |
| 3 | Indic tier split | verified-native **40%** · unverified crawl **18%** · translated **22%** · synthetic **20%** |
| 4 | Protected floor | **Indic ≥ 11% and agentic ≥ 3% of *every batch***, injected before OPUS, invisible to it |
| 5 | Honest deficit | **66.5B of the 69B agentic lane does not exist and must be synthesised** (real supply: 0.627B) |
| 6 | Validation | **7,875 GPU-h proxy ladder at 1B/3B = 3.0% of the 261,750 GPU-h run it protects.** No share is trusted until E1–E5 report. |

The single most important line in this spec is #5, and the second is #4.

---

## 1. Accounting conventions

Most mixture plans are wrong because they equivocate between three different things called "tokens".
This spec fixes the definitions first, because every downstream number depends on which one you mean.

| Term | Definition | Why it matters |
|---|---|---|
| **Sequence token** | A token that occupies a position in a training sequence and costs FLOPs. | This is the unit the budget is denominated in. All shares below are shares of sequence tokens. |
| **Supervised token** | A sequence token that contributes to the loss (green in the loss map). | In an agentic trajectory the issue text, repo files, and tool returns are **masked**. The model is never trained to imitate the environment. Only its own reasoning and actions are supervised. |
| **Candidate-pool token** | A token that must exist in the pool for OPUS to consider it. | At keep-fraction *k*, a lane needs `trained / k` tokens in the pool. **The composer's "covered" badges compare demand to supply and ignore this factor entirely.** At 40% keep that understates the requirement by 2.5×. |
| **Effective token** | Trained token × OPUS utility multiplier (V4 measured 6.0× at 40% keep). | Marketing unit. Never used to claim supply coverage. |

**Consequence, stated plainly:** the agentic lane holds 3.45% of sequence tokens but only **0.80% of supervised
tokens**. A reviewer who reads 3.45% as "3.45% of the learning signal goes to agentic" is reading it wrong,
and so is a plan that budgets it that way.

| Lane | Sequence share | Supervised fraction (assumed) | Supervised share |
|---|---:|---:|---:|
| General web | 31.65% | 1.00 | 33.38% |
| Code | 23.61% | 1.00 | 24.90% |
| Indic | 16.88% | 1.00 | 17.80% |
| STEM / math | 11.62% | 0.90 (problem statement masked) | 11.03% |
| Reasoning traces | 6.88% | 0.85 (prompt masked) | 6.17% |
| Long-context | 5.91% | 0.95 | 5.92% |
| **Agentic / tool-use** | **3.45%** | **0.22** (observations + tool returns masked) | **0.80%** |
| | | **Total supervised: 1,896B of 2,000B (94.8%)** | |

The 0.22 figure is the number I am least sure of and the one I would measure first (see **E0b**). It is derived
from the SWE-bench Verified loss map: of a ~12K-token trajectory, only the assistant reasoning and the patch
are green. If the true value is 0.15, the agentic lane is under-budgeted by ~45% and Phase 3 share must rise.

---

## 2. Budget and curriculum

**Weights are a schedule, not a constant.** V4 proved this: web faded 72→18, code ramped 13→35, STEM ramped
7→39, and only the Always-On lane stayed pinned. V5 keeps that structure and makes the floor explicit.

### 2.1 Stage weights (% of that stage's tokens)

| Lane | S1 · 0–40%<br>760B | S2 · 40–80%<br>760B | S3 · 80–100%<br>380B | Anneal<br>100B | Main avg | Total tokens |
|---|---:|---:|---:|---:|---:|---:|
| General web | 54 | 24 | 9 | 6 | 33.0% | 633.0B |
| Code | 14 | 29 | 33 | 20 | 23.8% | 472.2B |
| Indic | 16 | 16 | 18 | 26 | 16.4% | 337.6B |
| STEM / math | 6 | 14 | 18 | 12 | 11.6% | 232.4B |
| Reasoning traces | 2 | 8 | 12 | 16 | 6.4% | 137.6B |
| Long-context | 5 | 6 | 7 | 8 | 5.8% | 118.2B |
| Agentic / tool-use | 3 | 3 | 3 | 12 | 3.0% | 69.0B |
| **Sum** | **100** | **100** | **100** | **100** | **100.0** | **2,000B** |

Sequence length: **S1 8K → S2 32K → S3 128K → anneal 128K** (5% of anneal at 256K). RoPE base rescaled at each
boundary; long-context share ramps 5→7% to give the extension something to learn on rather than just longer
padding.

### 2.2 Why each lane is the size it is

**General web 33% → falling to 9%.** Web is the substrate, not the product. It is the only lane where supply
(4.8T) is not the binding constraint, which makes it the lane that gets cut whenever anything else needs room.
It holds MMLU and general world knowledge. Its S1 weight of 54% is high because language and world model form
early; its S3 weight of 9% is low because by then every additional web token displaces a scarce token that
would move a benchmark. The naive-web-heavy preset is what happens when you never make this cut.

**Code 23.8%.** The largest funded specialist lane, and the only specialist lane where real supply comfortably
covers demand (1.1T against a 944B pool need). Code is doing double duty: it wins LiveCodeBench / Aider
Polyglot / Codeforces directly, and it is the substrate the agentic lane is mined from — SWE-smith-style task
synthesis works by mining PRs and tests out of the same repos. Under-funding code starves agentic twice.

**Indic 16.9%.** The differentiator, and the reason the project exists, so it is funded above what pure
benchmark-return would justify. It is held roughly flat across stages (16/16/18) rather than ramped, because
Indic is a *language* capability that has to be present while the embedding and tokenizer geometry is forming.
Ramping it late, the way you ramp code, would mean the model learns English structure first and then bolts
Indic onto it — which is exactly the failure mode that produces fluent translationese and poor IndicGenBench
generation scores.

**STEM / math 11.6%.** Feeds AIME and GPQA jointly with the reasoning lane. Supply (250B) is close to demand,
so it carries a 1.69-epoch repetition factor — acceptable, and cheaper than synthesis.

**Reasoning traces 6.9%.** Sized by supply, not by desire: 85.1B real. A larger share is reachable only through
repetition beyond 3 epochs or through self-generated CoT, and I would rather spend the marginal token on
verified STEM than on unverified reasoning.

**Long-context 5.9%.** See §5 — most of this lane is *constructed*, not collected.

**Agentic 3.45%.** The hardest number in the plan, and the one I have deliberately kept small. See §4.

---

## 3. Supply reconciliation

Every lane, against the real inventory. **Pool need = total ÷ keep-fraction.** Epochs = pool need ÷ real supply.

| Lane | Total | Keep | Pool need | Real supply | Epochs | Verdict |
|---|---:|---:|---:|---:|---:|---|
| General web | 633.0B | 0.35 | 1,808.6B | 4,800B | 0.38 | covered |
| Code | 472.2B | 0.50 | 944.4B | 1,058B* | 0.89 | covered |
| Indic | 337.6B | 1.00 | 337.6B | 276B | 1.22 | repetition |
| STEM / math | 232.4B | 0.55 | 422.5B | 250B | 1.69 | repetition |
| Reasoning traces | 137.6B | 0.75 | 183.5B | 85.1B | 2.16 | repetition |
| Long-context | 118.2B | 0.80 | 147.8B | 100B | 1.48 | repetition + construction |
| **Agentic / tool-use** | **69.0B** | **1.00** | **69.0B** | **0.627B** | **110.1** | **synthesis — 96.4% of the lane does not exist** |

\* 1,100B Stack v2 + D3 + CommitPack, minus 42B reassigned to the long-context lane as repo-packed sequences
(§5). Double-counting a repo as both a code token and a long-context token is the exact wishful accounting this
session exists to prevent.

**Why scarce lanes get keep = 1.00.** OPUS earns its 4.7% overhead by discarding redundant tokens from
*abundant* lanes. On a lane where demand already exceeds supply, selection is pure loss: you throw away a token
and then repeat another one to replace it. So Indic and agentic bypass selection entirely rather than only
their floor portion.

**What that costs.** 20.3% of trained tokens no longer pass through OPUS, so the blended effective-token
multiplier falls from V4's 6.0× to:

```
0.797 × 6.0  +  0.203 × 1.0  =  4.98×      →  2.0T trained ≈ 9.97T effective
full-selection counterfactual                →  2.0T trained ≈ 12.0T effective
```

**Protecting the scarce lanes costs ~2.0T effective tokens, about 17%.** That is the price of the
differentiator and I am paying it on purpose. It is also a number I want measured, not assumed: the 6.0×
was calibrated on V4's English-heavy proxy and will not survive a rebalanced proxy unchanged (**E3**).

Repetition is safe here because every lane sits below ~2.2 epochs, comfortably inside the regime where repeated
tokens are worth close to fresh ones. The plan does not rely on repetition beyond 4 epochs anywhere.

---

## 4. The Indic lane

### 4.1 A supply contradiction that must be resolved first

Two inventory figures do not reconcile, and any plan that ignores this is building on sand:

- Session-3 provenance reports the Indic slice as **A 40% / B 25% / C 20% / D 15%**. Applied to the 276B slot
  supply, that implies **179B non-synthetic**.
- The inventory design note reports **114B trusted non-synthetic against 162B synthetic**.

179B ≠ 114B. **I resolve this by treating 40/25/20/15 as the composition of the slice as loaded in Session 3
(a mixture decision), and 114B/162B as the hard supply ceiling (a fact about the world).** The supply ceiling
wins. Working decomposition, flagged for re-audit before the run:

| Tier | Definition | Supply |
|---|---|---:|
| A — verified native | human-authored, provenance-checked, license-clear | 62B |
| B — unverified crawl | native but unaudited | 52B |
| C — translated | machine- or human-translated into Indic | 96B |
| D — synthetic | LLM-generated Indic | 66B |

A+B = 114B non-synthetic ✓ · C+D = 162B synthetic-origin ✓ · total 276B ✓

### 4.2 The split I am committing to

| Tier | Share of Indic lane | Tokens | Anneal portion | Supply | Epochs |
|---|---:|---:|---:|---:|---:|
| A verified native | **40%** | 135.0B | 18.2B | 62B | 2.18 |
| B unverified crawl | **18%** | 60.8B | 1.3B | 52B | 1.17 |
| C translated | **22%** | 74.3B | 3.9B | 96B | 0.77 |
| D synthetic | **20%** | 67.5B | 2.6B | 66B | 1.02 |

**Defence of each number:**

- **A at 40% and 2.18 epochs** — I am deliberately repeating the scarcest tier rather than backfilling with
  the abundant one. Verified native text is the only tier that carries real Indic register, idiom, and
  code-mixing; C and D inherit English syntax through the generator. Repeating A twice is a better trade than
  seeing C once, and 2.18 epochs is far inside the safe regime.
- **C capped at 22% despite 96B available** — translated text is where translationese enters. It is present
  because it carries factual and instructional coverage that native crawl lacks, not because it is plentiful.
  0.77 epochs means I am leaving 22B of translated supply on the floor on purpose.
- **D hard-capped at 20%** — synthetic Indic is a generator's model of Indic, and training on it at scale
  makes the model fluent in its own priors. 20% is my prior, not a measurement, and **E4 exists specifically
  to falsify it** by testing 20% against 40%.
- **A dominates the anneal at 70%** — the cooldown is where distributional character is set. Nothing but the
  highest-provenance tier belongs there.

### 4.3 Per-language floors — a second protected floor

A 16.9% Indic share that is 80% Hindi is not an Indic model. Inside the lane:

- Hindi ≤ **35%** of the lane
- top-6 (hi, bn, ta, te, mr, gu) ≤ **72%** combined
- the remaining 16 scheduled languages ≥ **18%** combined, each ≥ **0.8%** (≈2.7B tokens)

The 0.8% per-language floor is set by IndicGenBench, which scores across the long tail. Without it, MILU
improves and IndicGenBench does not move, and the mixture looks successful while the differentiator does not
exist.

### 4.4 Token share is not content share

At 16.9% token share, a language with fertility 2.2 tok/word next to English at 1.3 receives about **9.5% of
the content share** the number implies. This is measured before anything else runs (**E0a**) and the Indic
share is corrected upward if fertility exceeds 1.6×. Reporting an uncorrected token share is the Indic-lane
version of counting supply without the pool multiplier.

**Datasets:** Sangraha (native + verified subsets), the V4 Indic lineage, licensed news and government
long-form archives for tier A/B; existing translation corpora for C; verifier-filtered generation for D.

---

## 5. The agentic lane — where the plan is most exposed

Real supply is **627M tokens across 9 sets** (ToolBench 80M · Glaive FC-v2 50M · ToolACE 60M · xLAM/APIGen 25M
· NexusRaven 30M · SWE-smith 120M · Hermes FC 22M · OpenHands rollouts 90M · +1). The lane needs **69B**.
That is a **110× gap**. This is the number that must not be hidden behind a slider.

### 5.1 Provenance gate first

ToolBench (80M) and Glaive (50M) are **tier D** — 21% of all real agentic supply is the least-trusted tier.
Rule: **tier D capped at 1 epoch and excluded from the anneal entirely.** Tier A sets (NexusRaven, SWE-smith,
OpenHands) carry to 4 epochs.

Real contribution: 0.627B × ~4 epochs ≈ **2.5B**. **Synthesis must produce 66.5B.**

### 5.2 Synthesis plan and its cost

| Source | Verification predicate | Target |
|---|---|---:|
| Repo-mined SWE tasks (PR + hidden tests, mined from Stack v2 / CommitPack) | hidden test suite goes green | 30B |
| Sandboxed tool-use trajectories (synthetic API schemas + goals) | terminal state-checker matches goal state | 18B |
| Self-hosted web/OS agent rollouts (WebArena/OSWorld-style envs) | environment success predicate | 12B |
| Long-horizon multi-tool sessions (feeds long-context lane too) | composite predicate | 6.5B |

**Every trajectory is execution-verified. Nothing enters on a model's opinion that it succeeded.**

**Failure-and-recovery injection:** 25% of trajectories have a tool return deliberately corrupted (timeout,
malformed JSON, wrong schema, stale result) so the model must detect, recover, and continue. The brief asks
for a model that recovers when a call fails; that behaviour has to be in the data, and it is nearly absent
from naturally occurring trajectories, which are survivorship-filtered toward clean runs.

**Cost, at ~12K tokens per trajectory and a 35% verified-pass rate:**

```
verified trajectories needed   5.54 M
rollouts required             15.83 M
generated tokens                 190 B
sandbox execution              88 k core-hours   (~44 h on 2,000 cores)
generation                     18 k GPU-hours    (~2.9 days on 256 GPUs)
supervised yield                15.2 B green tokens
```

**18k GPU-hours is 6.9% of the 262k GPU-hour training run, spent before training starts, to fill one 3.45%
lane.** That cost is the reason the lane is 3.45% and not 6%. Doubling the share roughly doubles the
pre-training bill and the pass-rate assumption is the fragile part: if verified-pass comes in at 15% rather
than 35%, the generation bill goes to ~42k GPU-hours and the lane must be cut to 2% or the schedule slips.
**That threshold is measured in E0c before any synthesis is commissioned at volume.**

### 5.3 Why 3.45% and not more

The lane is small in pretraining **on purpose** and concentrated in the anneal at **12%** — 4× its main-run
weight. Scarce, expensive, high-signal capability is worth more per token at the end of training than
distributed thinly through it. The pretraining job for agentic is to make the format familiar; the anneal's
job is to make the capability sharp.

---

## 6. The long-context lane — construction, not collection

Long context is a *format*, not a *source*. Treating it as a corpus you must find is how this lane ends up
padded with concatenated junk. The 118.2B breaks down as:

| Component | Tokens | Construction |
|---|---:|---|
| Real long documents | 45.0B | books, long-form web, Indic legal/government archives — 0.45 epochs of the 100B slot |
| Repo-level packed code | 42.0B | whole repos in dependency order, deducted from the code lane pool |
| Packed agentic trajectories | 14.0B | naturally 12–60K tokens; no packing needed |
| Multi-document Indic packing | 12.0B | news archives and Sangraha long-form, same-topic clustered |
| Synthetic long-horizon retrieval/QA | 5.2B | needle and multi-hop QA over the packed contexts above |

Rule: packed sequences are counted **once**, in the long-context lane, and deducted from the source lane's
candidate pool. Cross-document attention masking is **off** for clustered packs (that is the point) and **on**
for filler packs.

**Targets:** long-eval, plus retrieval accuracy measured at 8K / 32K / 128K to catch the failure where average
score holds while the middle of the context degrades.

---

## 7. Difficulty and reasoning-length bands

The brief asks for a model whose reasoning depth is controllable. That requires two independent labels: how
hard the item is, and how long the model should think. Conflating them is what produces a model that thinks
for 3,000 tokens about unit conversion.

### 7.1 Difficulty bands

| Band | Definition | Code | Math / STEM | Agentic | Indic |
|---|---|---|---|---|---|
| **D1** routine | reference model ≥95% pass@1 | reverse a string in Python | convert 45 °C to °F | single call: `get_weather(city="Pune")` | translate "good morning" to Hindi |
| **D2** competent | 60–95% | MBPP-style function with edge cases | GSM8K: Janet's duck-egg problem | two parallel BFCL calls with schema selection | MILU-style factual QA in Telugu |
| **D3** hard | 20–60% | LiveCodeBench medium, unfamiliar API | AIME 2024 mid-paper geometry | tau2-bench airline rebooking under a policy constraint | IndicGenBench cross-lingual summarisation, Bengali→English |
| **D4** frontier | <20% but >0% | multi-file Django `TypeError` fix from SWE-bench Verified | GPQA-Diamond organic chemistry, HLE physics | OSWorld multi-app task with a corrupted tool return midway | multi-clause Kannada legal reasoning |

**Band mix by stage:**

| Stage | D1 | D2 | D3 | D4 |
|---|---:|---:|---:|---:|
| S1 | 45 | 35 | 15 | 5 |
| S2 | 25 | 35 | 28 | 12 |
| S3 | 12 | 28 | 35 | 25 |
| Anneal | 5 | 15 | 40 | 40 |

Items with pass@1 = 0 across k=8 are **quarantined, not discarded** — an item nothing can solve is usually a
broken item, and a mixture that silently trains on broken items teaches confident nonsense.

### 7.2 Reasoning-length bands

Control surface: a `<budget:Ln>` tag in system position at train and inference time.

| Band | Think-token budget | Share of reasoning lane | Concrete example |
|---|---:|---:|---|
| **L0** | 0 — answer directly | 12% | "Convert 45 °C to °F" → `113 °F` |
| **L1** | 1–256 | 28% | GSM8K word problem, 3 arithmetic steps |
| **L2** | 257–2,048 | 35% | MATH level-5 algebra; early AIME |
| **L3** | 2,049–16,384 | 20% | GPQA-Diamond; hard AIME; SWE-bench Verified fault localisation |
| **L4** | 16,385–65,536 | 5% | multi-hour agentic session with replanning |

**The band label is assigned by measured difficulty, not by the length the generator happened to emit.** This
is the mechanism that makes the control real rather than decorative:

```
sample k=8 from the V4 checkpoint → p = pass rate
p ≥ 0.95 → L0     0.60 ≤ p < 0.95 → L1
0.20 ≤ p < 0.60 → L2     0 < p < 0.20 → L3     p = 0 → quarantine
```

If instead you label by emitted length, the model learns "long problems produce long output" — a correlation,
not a control — and the tag does nothing at inference.

Two additions: **5% budget-exceeded examples**, where the budget binds and the model must stop cleanly and give
its best current answer rather than degenerating; and **L3/L4 traces must be answer-verified**, since an
unverified 8,000-token trace is 8,000 tokens teaching a plausible-sounding wrong path. L3 is where synthesis
concentrates — long verified CoT barely exists in the 85.1B real supply.

---

## 8. Protected floor and the OPUS proxy

### 8.1 The floor

**Indic ≥ 11% and agentic ≥ 3% of every batch**, injected before selection and invisible to OPUS.
Always-On lane total **16%** (11 Indic + 3 agentic + 2 verified reasoning), up from V4's 8%.

**Per-batch, not per-stage-average.** A stage-average floor is satisfiable by starving 400 consecutive batches
and compensating later, which is not a floor.

### 8.2 The proxy is the actual bug

V4's OPUS proxy had **cosine 0.876 with the English web band**. It was not neutral — it was a web-preference
detector, and it rejected almost every Indic and agentic batch on the merits *of that proxy*. With the floor
off, the trained-this-iteration meters for both lanes read **0.0%**. The floor did not fix the proxy; it routed
around it.

V5 fixes both:

| | V4 | V5 target |
|---|---|---|
| Proxy | English-heavy, single band | balanced multi-band |
| Cosine with any single band | 0.876 | ≤ **0.60** |
| Cosine with *every* band | — | ≥ **0.45** |
| Max−min spread across bands | — | ≤ **0.25** |
| Always-On floor | 8% | **16%** |
| Realised Indic tokens / nominal | ~0% without floor | ≥ **95%** |

The floor stays as defence-in-depth. A proxy that has been rebalanced once can drift again over a 2T run, and
a guarantee that depends on the selector behaving is not a guarantee.

---

## 9. The anneal reserve

**100B (5% of budget), LR cooled to zero, no OPUS selection at all** — every token in the anneal set is
hand-chosen, so selecting among them discards work already done.

The reserve is a **data** reserve, not just a compute reserve. This data is withheld from the main run
entirely, so the cooldown sees it fresh:

| Lane | Anneal tokens | What is reserved |
|---|---:|---|
| Indic | 26.0B | 18.2B tier-A verified native — **29% of all tier-A supply, never shown in the main run** |
| Code | 20.0B | CommitPackFT + highest-tier permissive repos |
| Reasoning | 16.0B | answer-verified L3 traces only |
| Agentic | 12.0B | execution-verified tier-A trajectories; **zero tier-D** |
| STEM | 12.0B | competition-grade, verified solutions |
| Long-context | 8.0B | 128K packs, 5% at 256K |
| General web | 6.0B | curated reference only |

Withholding 29% of tier-A Indic raises main-run tier-A repetition from 2.18 to 2.67 epochs. That is the cost
and it is worth it: the anneal is where the model's distributional character is set, and spending the best
Indic tokens in S1 to save 0.5 epochs of repetition is a bad trade.

**The anneal is 94% non-web.** That is the whole point of holding it back.

---

## 10. Benchmark map and falsifiable targets

Each lane is accountable to named benchmarks. Targets are stated as deltas against the V4 checkpoint because
inventing absolute baselines would be exactly the wishful accounting this plan is meant to avoid — fill from
the V4 run log at review.

| Lane | Share | Benchmarks | Target | Refuted if |
|---|---:|---|---|---|
| Code | 23.6% | LiveCodeBench, Aider Polyglot, Codeforces ELO | +8 pts LCB, +6 Aider | +8 pts LCB not reached despite a 2× code share vs V4 |
| Agentic | 3.45% | SWE-bench Verified & Live/Pro, Terminal-Bench, tau2-bench, BFCL v3, WebArena/WorkArena, GAIA, BrowseComp, OSWorld | +12 pts SWE-bench Verified, +10 BFCL v3 | SWE-bench Verified moves <5 pts after 66.5B synthesised tokens — the synthesis pipeline, not the share, is wrong |
| Reasoning + STEM | 18.5% | AIME, GPQA-Diamond, HLE | +10 AIME, +6 GPQA | AIME gain <4 pts |
| Long-context | 5.9% | long-eval; retrieval @8K/32K/128K | ≥90% retrieval at 128K | mid-context accuracy drops >15 pts vs 8K |
| Indic | 16.9% | MILU, IndicGenBench | +7 MILU, +9 IndicGenBench chrF++ | MILU moves but IndicGenBench does not → the per-language floor failed |
| General web | 31.7% | MMLU | within −0.8 pts of V4 | MMLU drops >1.5 pts → specialist lanes over-funded |

The MMLU guardrail is deliberate. Every specialist share in this plan is paid for out of web, and a plan that
does not name the point at which that trade has gone too far is not falsifiable.

---

## 11. Proxy experiment ladder

**Nothing above is trusted until this reports. A data decision is a hypothesis until a cheap experiment has
tested it.**

### 11.1 Zero-cost probes (no training, run first)

| ID | Probe | Metric | Gate |
|---|---|---|---|
| **E0a** | Tokenizer fertility per Indic language | tokens/word vs English | if any target language > 1.6× English, correct the Indic share upward before E1 |
| **E0b** | Green-token fraction on 10K real trajectories | supervised ÷ sequence tokens | if < 0.18, raise the agentic share; the 0.22 assumption in §1 is load-bearing |
| **E0c** | Verified-pass rate on 5K pilot synthesis rollouts | fraction passing the predicate | if < 0.20, the 18k GPU-h estimate is wrong by >2× — recost before commissioning |

### 11.2 Training ladder

| ID | Scale | Arms | Question | Primary metric | Decision rule |
|---|---|---:|---|---|---|
| **E1** | 1B × 30B | 8 | mixture shares: Indic {8,16,24}, agentic {1,3,6}, reasoning {3,6.5,10}, fractional factorial | maximin over lane-normalised scores | adopt the maximin share subject to MMLU regression ≤ 0.8 pts |
| **E2** | 1B × 30B | 3 | curriculum ordering: flat vs V5 3-stage vs reverse | end scores + forgetting (S1-lane eval at 40% vs at end) | keep 3-stage only if it beats flat by ≥2 pts with forgetting ≤1 pt |
| **E3** | 1B × 30B | 2 | floor on/off × English-heavy vs balanced proxy | realised Indic & agentic tokens ÷ nominal | balanced proxy must reach ≥95% realised without the floor; also re-measures the effective-token multiplier |
| **E4** | 3B × 100B | 3 | E1/E2 winner; anneal reserve on/off; synthetic-Indic cap 20% vs 40% | MILU + IndicGenBench + native-speaker preference (n=500, 3 raters) | raise the D cap to 40% only if MILU gains ≥1.5 pts **and** human preference does not regress |
| **E5** | 3B × 100B | 2 | reasoning-band control fidelity | Spearman ρ(requested band, emitted think-tokens) | ship only if ρ ≥ 0.8, median think-tokens ≤64 on the easy set, and MATH-500 accuracy is monotone in band |

**E4's human-preference arm is not optional.** Automatic Indic metrics reward translationese; a native reader
catches it and chrF++ does not. This is the check that decides whether the differentiator is real.

### 11.3 Cost

```
E1  8 arms × 125 GPU-h  =  1,000
E2  3 arms × 125 GPU-h  =    375
E3  2 arms × 125 GPU-h  =    250
E4  3 arms × 1,250 GPU-h = 3,750
E5  2 arms × 1,250 GPU-h = 2,500
                        ─────────
              total       7,875 GPU-h

full run: 6ND at 30B × 2.0T = 3.6e23 FLOPs
        ≈ 250,000 GPU-h at 400 TFLOP/s effective, +4.7% OPUS overhead = 261,750 GPU-h

proxy ladder = 3.0% of the run it protects
```

Three percent to find out whether the other ninety-seven are aimed correctly.

### 11.4 Results

**Not yet run.** Table pre-registered so the numbers cannot be chosen after seeing them:

| Exp | Arm | MMLU | MILU | IndicGenBench | HumanEval+ | BFCL v3 | GSM8K | Realised Indic % | Decision |
|---|---|---|---|---|---|---|---|---|---|
| E0a | — | | | | | | | | |
| E1 | indic-8 | | | | | | | | |
| E1 | indic-16 | | | | | | | | |
| E1 | indic-24 | | | | | | | | |
| E3 | floor-off / eng proxy | | | | | | | | |
| E3 | floor-off / balanced | | | | | | | | |

---

## 12. What would refute this plan

1. **E0b returns a green fraction below 0.15** → the agentic lane's supervised contribution is under 0.55% and
   3.45% is too small regardless of synthesis cost.
2. **E0c returns a verified-pass rate below 0.20** → 66.5B of synthesis is unaffordable; the lane drops to 2%
   and the anneal carries the whole capability.
3. **E1 shows Indic 16%→24% buys under 1.5 MILU points while costing over 1.0 MMLU point** → hold at 16%.
4. **E3 shows a balanced proxy costs more than 1.0× of the effective-token multiplier** → the 4.98× estimate
   in §3 is optimistic and the whole budget needs rescaling.
5. **E4 shows human preference regressing at a 20% synthetic-Indic cap** → the cap is already too high, not
   too low.
6. **E5 shows ρ < 0.5** → the difficulty-derived band labelling did not produce a control surface; length
   control moves to post-training and out of the mixture entirely.

---

## 13. Cleaning work order

The mixture names the starved slots, so cleaning effort follows it. Allocation of the next cleaning cycle:

| Priority | Slot | Share of effort | Why |
|---|---|---:|---|
| 1 | Indic tier-A verification | 45% | tier A runs at 2.67 main-run epochs; every 10B of B promoted to A drops that by ~0.3 |
| 2 | Agentic trajectory verification + environment build | 35% | the sandbox environments are the bottleneck on §5, not the generator |
| 3 | Reasoning-trace answer verification | 15% | unverified L3 traces are actively harmful |
| 4 | Long-doc Indic sourcing | 5% | feeds both lanes at once |

**Gate:** this plan is reviewed only after the cumulative cleaning target is met. A mixture is only as
trustworthy as the cleaned and documented tokens standing behind it.

---

## 14. Numbers I could not verify

Stated openly, because a plan that hides its assumptions cannot be reviewed:

1. The 114B/162B vs 40/25/20/15 contradiction (§4.1) — resolved by argument, needs a data-team ruling.
2. The A|B (62/52) and C|D (96/66) decomposition — consistent with both published figures, not independently
   confirmed.
3. Green-token fraction 0.22 — inferred from one loss map. **E0b.**
4. Verified-pass rate 0.35 and 12K tokens/trajectory — pipeline estimates. **E0c.**
5. Effective-token multiplier 6.0× — V4's number under V4's proxy; will change under a balanced proxy. **E3.**
6. Per-lane keep-fractions — set by scarcity argument, not measured. **E1/E3.**
7. V4 benchmark baselines — targets stated as deltas until the run log is available.

---

**Reproduce every table:** `python mixture_ledger.py`
