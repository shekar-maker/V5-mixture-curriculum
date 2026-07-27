#!/usr/bin/env python3
"""
V5 mixture ledger — recomputes every number in README.md.

If the README and this script disagree, the script is right.
Run:  python mixture_ledger.py
"""

# ----------------------------------------------------------------------------
# Inputs.  Everything below this block is derived.
# ----------------------------------------------------------------------------

TOTAL_B = 2000.0          # total sequence tokens, in billions
ANNEAL_B = 100.0          # held back for the cooldown (5%)
MAIN_B = TOTAL_B - ANNEAL_B

STAGE_FRAC = {"S1": 0.40, "S2": 0.40, "S3": 0.20}   # fraction of MAIN per stage

# stage weights, % of that stage's tokens
STAGE_W = {
    "General web":      {"S1": 54, "S2": 24, "S3":  9},
    "Code":             {"S1": 14, "S2": 29, "S3": 33},
    "Indic":            {"S1": 16, "S2": 16, "S3": 18},
    "STEM / math":      {"S1":  6, "S2": 14, "S3": 18},
    "Reasoning traces": {"S1":  2, "S2":  8, "S3": 12},
    "Long-context":     {"S1":  5, "S2":  6, "S3":  7},
    "Agentic / tool":   {"S1":  3, "S2":  3, "S3":  3},
}

ANNEAL_W = {
    "General web": 6, "Code": 20, "Indic": 26, "STEM / math": 12,
    "Reasoning traces": 16, "Long-context": 8, "Agentic / tool": 12,
}

# OPUS keep-fraction per lane. 1.00 = bypasses selection entirely
# (scarce lanes: selection on a supply-constrained lane is pure loss).
KEEP = {
    "General web": 0.35, "Code": 0.50, "Indic": 1.00, "STEM / math": 0.55,
    "Reasoning traces": 0.75, "Long-context": 0.80, "Agentic / tool": 1.00,
}

# Real supply from the SOTA Dataset Inventory, in billions of tokens.
SUPPLY = {
    "General web": 4800.0, "Code": 1058.0,   # 1100 minus 42B reassigned to long-context
    "Indic": 276.0, "STEM / math": 250.0,
    "Reasoning traces": 85.1, "Long-context": 100.0, "Agentic / tool": 0.627,
}

# Fraction of sequence tokens that carry loss (green in the loss map).
# The agentic figure is the load-bearing assumption -- probe E0b.
SUPERVISED_FRAC = {
    "General web": 1.00, "Code": 1.00, "Indic": 1.00, "STEM / math": 0.90,
    "Reasoning traces": 0.85, "Long-context": 0.95, "Agentic / tool": 0.22,
}

INDIC_SUPPLY = {"A verified native": 62.0, "B unverified crawl": 52.0,
                "C translated": 96.0, "D synthetic": 66.0}
INDIC_SPLIT = {"A verified native": 0.40, "B unverified crawl": 0.18,
               "C translated": 0.22, "D synthetic": 0.20}
INDIC_ANNEAL_SPLIT = {"A verified native": 0.70, "B unverified crawl": 0.05,
                      "C translated": 0.15, "D synthetic": 0.10}

LANES = list(STAGE_W)


def rule(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ----------------------------------------------------------------------------
rule("1. BUDGET AND CURRICULUM")

for s in STAGE_FRAC:
    tot = sum(STAGE_W[l][s] for l in LANES)
    assert tot == 100, f"stage {s} sums to {tot}"
assert sum(ANNEAL_W.values()) == 100

main_share = {l: sum(STAGE_W[l][s] * STAGE_FRAC[s] for s in STAGE_FRAC) for l in LANES}
main_tok = {l: MAIN_B * main_share[l] / 100 for l in LANES}
anneal_tok = {l: ANNEAL_B * ANNEAL_W[l] / 100 for l in LANES}
tok = {l: main_tok[l] + anneal_tok[l] for l in LANES}
assert abs(sum(tok.values()) - TOTAL_B) < 1e-6

print(f"{'LANE':<18}{'S1':>5}{'S2':>5}{'S3':>5}{'ANN':>5}"
      f"{'main%':>8}{'main B':>9}{'ann B':>8}{'total B':>10}{'share%':>8}")
for l in LANES:
    print(f"{l:<18}{STAGE_W[l]['S1']:>5}{STAGE_W[l]['S2']:>5}{STAGE_W[l]['S3']:>5}"
          f"{ANNEAL_W[l]:>5}{main_share[l]:>8.1f}{main_tok[l]:>9.1f}"
          f"{anneal_tok[l]:>8.1f}{tok[l]:>10.1f}{100*tok[l]/TOTAL_B:>8.2f}")
print(f"{'TOTAL':<18}{'':>20}{sum(main_share.values()):>8.1f}"
      f"{sum(main_tok.values()):>9.1f}{sum(anneal_tok.values()):>8.1f}"
      f"{sum(tok.values()):>10.1f}{100:>8.2f}")

# ----------------------------------------------------------------------------
rule("2. SUPPLY RECONCILIATION  (pool need = total / keep)")

print(f"{'LANE':<18}{'total B':>9}{'keep':>7}{'pool B':>10}{'supply B':>10}"
      f"{'epochs':>9}  verdict")
for l in LANES:
    pool = tok[l] / KEEP[l]
    ep = pool / SUPPLY[l]
    verdict = ("covered" if ep <= 1.0 else
               "repetition" if ep <= 4.0 else
               f"SYNTHESIS -- {100*(1 - SUPPLY[l]*4/tok[l]):.1f}% of lane does not exist")
    print(f"{l:<18}{tok[l]:>9.1f}{KEEP[l]:>7.2f}{pool:>10.1f}{SUPPLY[l]:>10.1f}"
          f"{ep:>9.2f}  {verdict}")

# ----------------------------------------------------------------------------
rule("3. SUPERVISED TOKENS  (sequence share != learning-signal share)")

sup = {l: tok[l] * SUPERVISED_FRAC[l] for l in LANES}
sup_total = sum(sup.values())
print(f"{'LANE':<18}{'seq B':>9}{'sup frac':>10}{'sup B':>9}"
      f"{'seq share%':>12}{'sup share%':>12}")
for l in LANES:
    print(f"{l:<18}{tok[l]:>9.1f}{SUPERVISED_FRAC[l]:>10.2f}{sup[l]:>9.1f}"
          f"{100*tok[l]/TOTAL_B:>12.2f}{100*sup[l]/sup_total:>12.2f}")
print(f"\nsupervised total: {sup_total:.1f}B of {TOTAL_B:.0f}B "
      f"({100*sup_total/TOTAL_B:.1f}%)")

# ----------------------------------------------------------------------------
rule("4. EFFECTIVE-TOKEN MULTIPLIER  (cost of protecting the scarce lanes)")

selected = sum(tok[l] for l in LANES if KEEP[l] < 1.0) / TOTAL_B
print(f"through OPUS: {100*selected:.1f}%   bypassing: {100*(1-selected):.1f}%")
for m in (6.0, 5.5, 5.0):
    blended = selected * m + (1 - selected) * 1.0
    print(f"  selected multiplier {m:.1f}x -> blended {blended:.2f}x "
          f"-> {TOTAL_B*blended/1000:.2f}T effective")
print(f"  full-selection counterfactual at 6.0x: {TOTAL_B*6.0/1000:.1f}T")
print(f"  cost of the floor: ~{TOTAL_B*6.0/1000 - TOTAL_B*(selected*6+(1-selected))/1000:.1f}T "
      f"effective (~{100*(1 - (selected*6+(1-selected))/6.0):.0f}%)")

# ----------------------------------------------------------------------------
rule("5. INDIC TIER SPLIT")

assert abs(sum(INDIC_SPLIT.values()) - 1.0) < 1e-9
assert abs(sum(INDIC_ANNEAL_SPLIT.values()) - 1.0) < 1e-9
ind_tot, ind_ann = tok["Indic"], anneal_tok["Indic"]
print(f"Indic lane {ind_tot:.1f}B (main {main_tok['Indic']:.1f}, anneal {ind_ann:.1f}) "
      f"vs supply {sum(INDIC_SUPPLY.values()):.0f}B")
print(f"\n{'TIER':<22}{'share':>7}{'total B':>9}{'anneal B':>10}{'main B':>9}"
      f"{'supply B':>10}{'epochs':>8}{'main ep*':>10}")
for t in INDIC_SUPPLY:
    total_t = ind_tot * INDIC_SPLIT[t]
    ann_t = ind_ann * INDIC_ANNEAL_SPLIT[t]
    main_t = total_t - ann_t
    print(f"{t:<22}{100*INDIC_SPLIT[t]:>6.0f}%{total_t:>9.1f}{ann_t:>10.1f}"
          f"{main_t:>9.1f}{INDIC_SUPPLY[t]:>10.1f}"
          f"{total_t/INDIC_SUPPLY[t]:>8.2f}{main_t/(INDIC_SUPPLY[t]-ann_t):>10.2f}")
print("* main ep = main-run epochs on supply remaining after the anneal reserve is withheld")

# ----------------------------------------------------------------------------
rule("6. AGENTIC SYNTHESIS COST")

TRAJ_TOK_K, PASS_RATE, SANDBOX_S, GEN_TOK_S = 12.0, 0.35, 20, 3000
ag = tok["Agentic / tool"]
real_used = SUPPLY["Agentic / tool"] * 4          # tier-A at 4 epochs
synth = ag - real_used
n_traj = synth * 1e9 / (TRAJ_TOK_K * 1e3)
rollouts = n_traj / PASS_RATE
gen_tokens = rollouts * TRAJ_TOK_K * 1e3
core_h = rollouts * SANDBOX_S / 3600
gpu_h = gen_tokens / GEN_TOK_S / 3600

print(f"lane need                 {ag:>10.1f} B")
print(f"real supply x4 epochs     {real_used:>10.2f} B")
print(f"must synthesise           {synth:>10.2f} B   ({100*synth/ag:.1f}% of the lane)")
print(f"verified trajectories     {n_traj/1e6:>10.2f} M   at {TRAJ_TOK_K:.0f}K tokens each")
print(f"rollouts @ {PASS_RATE:.0%} pass      {rollouts/1e6:>10.2f} M")
print(f"generated tokens          {gen_tokens/1e9:>10.0f} B")
print(f"sandbox execution         {core_h/1e3:>10.0f} k core-h  "
      f"(~{core_h/2000:.0f} h on 2,000 cores)")
print(f"generation                {gpu_h/1e3:>10.0f} k GPU-h   "
      f"(~{gpu_h/256/24:.1f} days on 256 GPUs)")
print(f"supervised yield          {ag*SUPERVISED_FRAC['Agentic / tool']:>10.1f} B green tokens")

# ----------------------------------------------------------------------------
rule("7. PROXY LADDER vs FULL RUN")

GPU_FLOPS = 4e14          # 400 TFLOP/s effective (~40% MFU on H100)
OPUS_OVERHEAD = 0.047

def gpu_hours(params, tokens):
    return 6 * params * tokens / GPU_FLOPS / 3600

ladder = [("E1 mixture shares", 1e9, 3e10, 8), ("E2 curriculum order", 1e9, 3e10, 3),
          ("E3 floor x proxy", 1e9, 3e10, 2), ("E4 anneal + synth cap", 3e9, 1e11, 3),
          ("E5 band control", 3e9, 1e11, 2)]
total_proxy = 0
for name, n, d, arms in ladder:
    h = gpu_hours(n, d)
    total_proxy += h * arms
    print(f"{name:<24}{n/1e9:>4.0f}B x {d/1e9:>4.0f}B  {h:>7.0f} GPU-h/arm "
          f"x {arms}  = {h*arms:>7.0f}")
full = gpu_hours(3e10, 2e12) * (1 + OPUS_OVERHEAD)
print(f"\n{'proxy ladder total':<24}{total_proxy:>36.0f} GPU-h")
print(f"{'full run (30B x 2.0T)':<24}{full:>36.0f} GPU-h  (incl. 4.7% OPUS overhead)")
print(f"{'ladder as share of run':<24}{100*total_proxy/full:>35.2f}%")

print("\nAll assertions passed. Every README table is reproduced above.")
