## Test scores (seed 42)

| Run | Training data | chrF | BLEU-4 | BLEU-1 |
|---|---|---|---|---|
| Copy | source unchanged | 41.83 | 29.29 | 45.72 |
| A | PHINC only, word | 41.50 | 28.49 | 54.45 |
| B | PHINC only, BPE | 46.71 | 30.38 | 54.28 |
| C | clean synthetic, then PHINC, BPE | 51.72 | 34.42 | 59.76 |
| D | noisy synthetic, then PHINC, BPE | 51.64 | 34.68 | 60.40 |
| E | as D, with copy mechanism | 51.59 | 35.44 | 59.88 |
| F | as E, synthetic data code-mixed | 52.23 | 35.93 | 60.68 |

## Three seeds (42, 43, 44)

| Run | chrF mean | chrF lowest | chrF highest | BLEU-4 mean | BLEU-4 lowest | BLEU-4 highest |
|---|---|---|---|---|---|---|
| B | 45.99 | 45.29 | 46.71 | 30.56 | 29.96 | 31.33 |
| C | 51.66 | 51.25 | 52.00 | 34.40 | 34.33 | 34.46 |
| D | 51.64 | 51.56 | 51.73 | 34.80 | 34.68 | 34.90 |
| E | 51.46 | 51.34 | 51.59 | 35.35 | 35.22 | 35.44 |
| F | 52.47 | 52.23 | 52.61 | 35.96 | 35.88 | 36.07 |

## Paired bootstrap, seed 42 outputs, 1000 samples

| Comparison | Metric | Difference | 95% interval | p-value |
|---|---|---|---|---|
| run_d minus run_b | chrF | +4.93 | +4.39 to +5.53 | < 0.001 |
| run_d minus run_b | BLEU-4 | +4.29 | +3.64 to +4.94 | < 0.001 |
| run_d minus run_c | chrF | -0.07 | -0.54 to +0.40 | 0.597 |
| run_d minus run_c | BLEU-4 | +0.26 | -0.35 to +0.89 | 0.169 |
| run_c minus run_b | chrF | +5.01 | +4.46 to +5.55 | < 0.001 |
| run_c minus run_b | BLEU-4 | +4.03 | +3.30 to +4.68 | < 0.001 |
| run_b minus run_a | chrF | +5.21 | +4.43 to +6.08 | < 0.001 |
| run_b minus run_a | BLEU-4 | +1.89 | +1.11 to +2.89 | < 0.001 |
| run_e minus run_d | chrF | -0.05 | -0.57 to +0.44 | 0.563 |
| run_e minus run_d | BLEU-4 | +0.77 | +0.16 to +1.43 | 0.014 |
| run_f minus run_e | chrF | +0.64 | +0.20 to +1.09 | 0.003 |
| run_f minus run_e | BLEU-4 | +0.49 | -0.10 to +1.06 | 0.048 |
| run_f minus run_d | chrF | +0.59 | +0.06 to +1.09 | 0.011 |
| run_f minus run_d | BLEU-4 | +1.26 | +0.61 to +1.91 | < 0.001 |
| run_f minus run_b | chrF | +5.53 | +4.96 to +6.09 | < 0.001 |
| run_f minus run_b | BLEU-4 | +5.55 | +4.81 to +6.30 | < 0.001 |
