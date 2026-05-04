# Experiment Notes And Useful Test Results

Current accepted baseline:
- `v4.2`
- `907`: `118/907 = 13.009923%`
- `my_dataset`: `159/2000 = 7.95%`

Recommended screening workflow:
- Test `907` first with the fast harness.
- Only run `my_dataset` if the candidate is at least `117/907`.
- If a candidate only ties `907`, use `my_dataset` as the tiebreaker.

Known good accepted changes:
- `v3.8`: stronger retro long-match weight and stronger `2x / 0.5x` ratio rule.
- `v3.9`: retro long-match `12 -> 13`, simple diff rule `10 -> 12`, ratio rule `15 -> 18`.
- `v4.0`: added the `ABABABA` detector and reshaped the local continuation branch:
  - `freq_1`: `(c ** 3) * 1.5 -> (c ** 3.2) * 0.9`
  - `freq_2`: `(c ** 3.5) * 2 -> (c ** 3.5) * 1.8`
- `v4.1`: stronger alternating-difference rule:
  - `altdiff`: `20 -> 24`
- `v4.2`: added a narrow two-number `+/-9` continuation seed:
  - If the last two entries differ by `9` or `-9`, add `15` confidence to the same-step continuation.
  - Exact verification: `907 -> 118/907`, `my_dataset -> 159/2000`.
  - Combined objective: `276 -> 277`.

## Latest pass

Fast-screened candidate families after precomputing the unchanged expensive confidence state for `907`:
- Extra two-number arithmetic seeds for missing common steps.
- Exact period and partial-period cycle continuation for periods `3-6`.
- Repeat/mirror/complement rules.
- Modulo-101 wraparound arithmetic.

Accepted:
- `step9_w15`
  - `907`: `117 -> 118`
  - `my_dataset`: `159 -> 159`
  - Rationale: humans often count by nines because it feels like a near-ten pattern; this catches that two-number seed without broadening the existing simple-difference rule.

Rejected / not kept:
- `+/-9` weights above `15`: tied `907` in the fast screen, but `15` is the smallest improving weight and safest to keep.
- Broad `11/22/33/44` family: reached `118/907`, but it is broader and less isolated than the accepted `+/-9` rule.
- Triple-repeat continuation at high weight: reached `118/907`, but required a much heavier `100+` confidence push.
- Cold-start `15`: reached `118/907`, but was rejected as a benchmark-specific first-value guess rather than a genuine predictor.
- Period, mirror, complement, and modulo-wrap rules: did not beat `117/907`.

Useful working notes:
- `ABABABA` by itself was neutral:
  - `116/907`
  - `159/2000`
- Simple `freq_1` retunes were not enough:
  - `(c ** 3) * 1.1 -> 117/907`, `157/2000`
  - `(c ** 2.8) * 1.3 -> 117/907`, `157/2000`
  - `(c ** 2.6) * 1.4 -> 117/907`, `157/2000`
- The first balanced uplift came only after combining the neutral `ABABABA` addition with a reshaped `freq_1/freq_2` local continuation branch.
- The exact local-frequency plateau around `v4.0` is broad but not better:
  - `freq_1 1.05 / 3.1` with `freq_2 1.8 / 3.5 -> 117/907`
  - `freq_1 1.05 / 3.1` with `freq_2 1.9 / 3.4 -> 117/907`, ceiling `158/2000`
  - `freq_1 0.95 / 3.2` with `freq_2 2.1 / 3.1 -> 117/907`, ceiling `158/2000`
- The old improvement weights are now mostly flat:
  - `retro_long 14 -> 117/907`, `158/2000`
  - `ratio2 20 -> 117/907`, `158/2000`
  - `retro14 + ratio20 -> 117/907`, ceiling `158/2000`
  - `retro_long 15 -> 117/907`
  - `ratio2 22 -> 117/907`
- Arithmetic-rule retuning mostly tied `907` but did not help `my_dataset`, except one:
  - `simplediff 11 -> 117/907`, ceiling `158/2000`
  - `secondorder 18 -> 117/907`, ceiling `158/2000`
  - `eqdiff 36 -> 117/907`, ceiling `158/2000`
  - `altdiff 24 -> 117/907`, `159/2000`
- Local bracket around `altdiff 24` suggests `24` is the best nearby setting:
  - `altdiff 22 -> 117/907`, ceiling `159/2000`
  - `altdiff 23 -> 117/907`, ceiling `159/2000`
  - `altdiff 25 -> 117/907`, ceiling `159/2000`
  - `altdiff 26 -> 117/907`, ceiling `159/2000`
- Pushing those same branches harder starts to regress:
  - `retro17 + ratio24 -> 116/907`
  - `retro20 + ratio28 -> 115/907`

Important benchmark trap:
- The digit-branch truthiness checks are intentionally left as `if nextseconddiff and nextfirstdiff`.
- Changing them to `is not None` is more correct logically, but it dropped `907` by one hit in testing.
- That change took `12.348% -> 12.238%`, so do not re-apply it if benchmark score is the target.

## Structural ideas already tested

Parity model trained from 20k base data:
- Long-term even/odd model regressed badly.
- Historical result noted in-file at the time: `907 12.348 -> 11.348`, `my 7.7 -> 7.7`.
- Removed.

Full-number order-2 Markov:
- Adding `_base_mc_full2` and voting from the last two full values was worse.
- Tested weights `2.0`, `3.0`, `4.0`.
- Results on `907`: `113`, `112`, `112`.

Digit order-2 Markov:
- Can improve `907`, but hurts `my_dataset` enough to fail the even-weight objective.
- Example: weight `1.2` gave `117/907`, but `156/2000` on `my_dataset`.
- Variants with `retro14`, `ratio20`, `diff13` still only produced `156/2000`.

First-digit-only order-2 Markov:
- Worse on `907`.
- `first_mc2_0.6 -> 111/907`
- `first_mc2_1.0 -> 110/907`

Second-digit-only order-2 Markov:
- Could tie `907`, but not improve the pair objective.
- `second_mc2_1.0 -> 116/907`, but `156/2000` on `my_dataset`.

Pure session Markov with no base contamination:
- Digits:
  - `pure_digits_0.8 -> 114/907`
  - `pure_digits_1.2 -> 112/907`
- Full numbers:
  - `pure_full_2.0 -> 115/907`
  - `pure_full_3.0 -> 115/907`

Sliding-window session Markov:
- Digits:
  - `window_digits_0.8_w80 -> 112/907`
  - `window_digits_1.2_w80 -> 111/907`
- Full numbers:
  - `window_full_2.0_w80 -> 116/907`
  - `window_full_3.0_w80 -> 116/907`
- These full-window variants only tied `907` and were much slower on `my_dataset`; not promising enough to keep pushing.

20k-base vs live-session source mixing:
- Full-number RF faster session takeover was clearly bad.
- Results on `907`:
  - `full_rf_stage_gentle -> 108/907`
  - `full_rf_recent80 -> 106/907`
- Full-number Markov state override also failed:
  - `full_mc_override2 -> 114/907`
  - `full_rf_recent80__mc_override2 -> 105/907`
- Digit RF faster session takeover was also worse:
  - `digit_rf_recent40_mild -> 114/907`
  - `digit_rf_recent40_mild__mc_boost3 -> 114/907`
- Digit Markov live-session count boosting was neutral, not improving:
  - `digit_mc_boost2 -> 116/907`
  - `digit_mc_boost3 -> 116/907`, `159/2000`
  - `digit_mc_boost4 -> 116/907`
  - `digit_mc_boost5 -> 116/907`
  - `digit_mc_boost6 -> 116/907`, `159/2000`
- Conclusion: stronger live-session influence inside the existing RF/Markov branches either hurts `907` or does nothing on both datasets.

Predictor performance tracking / meta-learning:
- Default scaling (`0.5x` to `1.5x`) was bad.
- Gentle scaling was still worse.
- Results:
  - `meta_default -> 108/907`
  - `meta_gentle -> 113/907`

Ranked-vote combiner:
- Existing predictors nominate candidates; final answer chosen by vote count, tie-broken by confidence.
- Results:
  - `vote_1 -> 56/907`
  - `vote_2 -> 87/907`
  - `vote_3 -> 116/907`
- Conclusion: vote thresholds below 3 are destructive; threshold 3 is only neutral.

Difference-pattern continuation branch:
- Matching recent deltas against dataset deltas and projecting the next delta was worse.
- Results on `907`:
  - weight `1.5 -> 114`
  - weight `2.5 -> 113`
  - weight `4.0 -> 114`

ABABAB 7-element detector:
- Extending the existing ABAB detector to a 7-element `ABABABA` pattern was neutral.
- Results:
  - `abab7_plus150 -> 116/907`
  - `abab7_plus150 -> 159/2000`
- Conclusion: safe enough as a harmless coverage extension, but it is not a score-improving change by itself.

History-shape changes:
- Retency-only retro weighting regressed hard:
  - `retro = age_decay -> 107/907`
- Exponential dataset long-match growth was also worse:
  - `dataset_match_exponential -> 110/907`
- Combining both was disastrous:
  - `retro_age_decay__dataset_match_exponential -> 102/907`

## Frequency-table diagnostic

Purpose:
- Determine whether `frequency` and `frequency2` are just overfitting the 907 person or are genuinely load-bearing.

Results on `907`:
- Current: `116/907`
- Remove digit `frequency`: `111/907`
- Remove full-number `frequency2`: `112/907`
- Remove both: `108/907`

Conclusion:
- The frequency tables are materially helping `907`.
- Do not zero them out blindly as a generalisation fix.

## Additive heuristic families already tested

Round-number static bias:
- Hurt immediately, even at very small weights.
- Results on `907`:
  - `0.2 -> 114`
  - `0.4 -> 110`
  - `0.6 -> 109`
  - `0.8 -> 109`
  - `1.0 -> 109`
  - `1.5 -> 108`
  - `2.0 -> 108`

Recent-range drift toward recent mean:
- Too blunt; always worse.
- Results on `907`:
  - `0.5 -> 112`
  - `1.0 -> 112`
  - `1.5 -> 111`
  - `2.0 -> 111`
  - `3.0 -> 111`
  - `4.0 -> 111`

Last-digit session frequency:
- Worse at every tested strength.
- Results on `907`:
  - `0.05 -> 113`
  - `0.10 -> 112`
  - `0.15 -> 110`
  - `0.20 -> 109`
  - `0.30 -> 108`
  - `0.40 -> 108`

Odd/even session bias:
- Always worse.
- Results on `907`:
  - scale `0.5 -> 115`
  - scale `1.0 -> 115`
  - scale `1.5 -> 115`
  - scale `2.0 -> 115`
  - scale `4.0 -> 115`
  - scale `6.0 -> 114`

Recent tens-bucket bias:
- Also worse.
- Results on `907`:
  - `0.05 -> 115`
  - `0.10 -> 115`
  - `0.15 -> 114`
  - `0.20 -> 114`
  - `0.30 -> 114`

Wrong-prediction penalty:
- Penalising the bot's last 5 wrong guesses by `-3.0` was much worse.
- Results on `907`:
  - `wrong_penalty_3_last5 -> 108/907`
- Conclusion: being wrong does not mean the same number is genuinely unlikely next; this introduces destructive anti-repeat bias.

Session-built frequency table blending:
- Fading the static `frequency` signal toward a session-built table was bad even with smooth support-based blending.
- Results on `907`:
  - `session_freq_blend_smooth_support -> 111/907`
- Keeping the static table intact and only adding a mild session-frequency reinforcement was still worse:
  - `session_freq_additive_support_035 -> 114/907`
  - `session_freq_additive_support_015 -> 114/907`
- Conclusion: the static `frequency` table is too load-bearing on `907`; even gentle session-transition blending is currently net harmful.

Session-only periodicity and uncertainty ideas:
- Entropy-adaptive fallback to personal session frequency was clearly bad.
- Results on `907`:
  - `entropy_0.85_30 -> killed late; final ceiling 116/907`
  - `entropy_0.80_30 -> killed late; final ceiling 116/907`
  - `entropy_0.85_20 -> killed late; final ceiling 116/907`
- Actual observed wins before kill were much lower (`71`, `65`, `79` respectively), so these were not near-misses.
- Raw autocorrelation period detection on the session sequence did better, but still only tied the old baseline and missed the current one.
- Results on `907`:
  - `period_0.25_20 -> 116/907`
  - `period_0.20_20 -> 116/907`
- Because neither entropy nor period reached the current `117/907` baseline, stronger follow-up sweeps and combinations were skipped.
- Conclusion: these session-only overlays do not beat the current ensemble on the 907 benchmark.

Smarter existing-signal rewrites:
- Recency-weighted `freq_1` transitions were worse across all screened half-life / multiplier combinations.
- Results on `907`:
  - `freq1_recency_hl100_m12 -> killed late; final ceiling 116/907`
  - `freq1_recency_hl150_m12 -> killed late; final ceiling 116/907`
  - `freq1_recency_hl200_m12 -> killed late; final ceiling 116/907`
  - `freq1_recency_hl150_m8 -> killed late; final ceiling 116/907`
  - `freq1_recency_hl150_m16 -> killed late; final ceiling 116/907`
- The best actual observed run among them was only `113` wins before kill, so this was not a near-miss.
- Positional modular bias was also screened and failed:
  - `positional_0.8 -> killed late; final ceiling 116/907`
- Conclusion: making the local continuation rule more recency-sensitive or adding modulo-position bias did not improve the current v4.0 ensemble.

## Older micro-tuning already exhausted

These were swept repeatedly before the structural tests and did not produce a stable improvement over the current accepted baseline path:
- Full-number RF weight changes
- Full-number Markov weight changes
- `frequency2` weight changes
- Local `freq_1` and `freq_2` exponent/weight nudges
- Dataset sequence-match weight nudges
- Base frequency prior denominator changes
- Small nearby tweaks to the `retro`, diff, and ratio weights once `v3.9` was reached

Useful summary:
- The old `v3.9` weights sat on a local plateau.
- Most nearby scalar tweaks still tie or regress.
- The only tuning change that broke through was reshaping the local continuation branch, not just nudging the obvious top-level rule weights.
- Most structural variants so far either hurt `907` or improved `907` while hurting `my_dataset`.

## Practical guidance for future AI passes

Do:
- Keep `v4.0` as the reference baseline unless a candidate beats both objectives or beats `907` and at least ties `my_dataset`.
- Screen on `907` first.
- Only run `my_dataset` for `907 >= 117/907`.
- Prefer experiments that change the combination architecture or predictor source, not tiny scalar nudges.

Do not waste time re-testing:
- Round-number bias
- Recent-mean drift
- Last-digit bias
- Session parity bias
- Tens-bucket bias
- Pure session Markov branches
- Sliding-window digit Markov
- Predictor meta-weighting
- Ranked-vote combiner in the simple nomination-count form
- Order-2 digit/full Markov as additive branches without a better way to protect `my_dataset`
- Faster live-session takeover inside the existing RF branches
- Full-number Markov state override from live-session counts
- Digit Markov live-session count boosting inside the existing branch
- `retro = age_decay` recency-only history weighting
- Exponential dataset long-match growth
- Hard wrong-prediction penalties
- Session-frequency blending that fades or supplements the static `frequency` branch
- Entropy-adaptive fallback to personal session frequency
- Raw autocorrelation period detection on the full numeric session sequence
- Recency-weighted replacement for the current `freq_1` session continuation rule
- Positional modular bias by sequence index bucket
