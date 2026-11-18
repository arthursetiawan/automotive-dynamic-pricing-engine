# EDA summary — synthetic parts quotes

_Generated 2026-11-18 03:16 UTC_

## Dataset

- **Rows:** 20,000
- **Seed:** 42
- **Overall win rate:** 47.72%
- **Source:** `models.train.generate_synthetic_quotes` (no external files)

## Leakage / sanity checks

- Labels are **synthetic** from the same generative rule as margins and discounts; the model must learn correlations, not read a hidden ID.
- `won` is **not** passed to the API as an input; training uses only `FEATURE_COLUMNS`.
- Requested discounts respect per-tier caps (max 0.40).

## Win rate by part category

Sparkline (relative win rates, high → low): `█▇▇▇▇▇▇▇`

### Counts

| part_category | quotes | wins | win_rate |
| --- | --- | --- | --- |
| hvac | 2443 | 1192 | 0.4879 |
| transmission | 2497 | 1212 | 0.4854 |
| electrical | 2528 | 1222 | 0.4834 |
| suspension | 2555 | 1222 | 0.4783 |
| engine | 2542 | 1208 | 0.4752 |
| brake | 2514 | 1186 | 0.4718 |
| body | 2488 | 1167 | 0.4691 |
| filter | 2433 | 1134 | 0.4661 |

## Win rate by dealer tier

Sparkline: `█▇▆▆`

### Counts

| dealer_tier | quotes | wins | win_rate |
| --- | --- | --- | --- |
| bronze | 4032 | 2187 | 0.5424 |
| silver | 7070 | 3519 | 0.4977 |
| gold | 5947 | 2622 | 0.4409 |
| platinum | 2951 | 1215 | 0.4117 |

### Win rate by margin band

| margin_band | quotes | wins | win_rate |
| --- | --- | --- | --- |
| <0% | 1720 | 346 | 0.2012 |
| 0–5% | 876 | 256 | 0.2922 |
| 5–10% | 1088 | 372 | 0.3419 |
| 10–15% | 1331 | 544 | 0.4087 |
| 15–20% | 1647 | 675 | 0.4098 |
| ≥20% | 13338 | 7350 | 0.5511 |

### Win rate by discount depth

| discount_band | quotes | wins | win_rate |
| --- | --- | --- | --- |
| (-0.001, 0.05] | 5151 | 3026 | 0.5875 |
| (0.05, 0.1] | 5154 | 2737 | 0.531 |
| (0.1, 0.15] | 3109 | 1479 | 0.4757 |
| (0.15, 0.2] | 3100 | 1281 | 0.4132 |
| (0.2, 0.3] | 2770 | 885 | 0.3195 |
| (0.3, 0.41] | 716 | 135 | 0.1885 |

## Numeric distributions (selected fields)

```
        base_cost  list_price  requested_discount  gross_margin_pct  tariff_rate
count  20000.0000  20000.0000          20000.0000        20000.0000   20000.0000
mean     245.9232    448.6158              0.1197            0.2571       0.1403
std      192.8975    365.2921              0.0890            0.1746       0.0857
min        9.8000     13.1629              0.0000           -0.6608       0.0009
25%      106.3120    186.1037              0.0484            0.1496       0.0734
50%      198.0685    355.6816              0.0970            0.2869       0.1240
75%      337.4243    612.6749              0.1755            0.3913       0.1934
max     2040.0000   2864.6126              0.3998            0.5686       0.3500
```

## Takeaways for modeling

- Win rate rises with **gross margin** and falls with **requested discount**—consistent with the generative rule.
- **Platinum/Gold** tiers show higher win rates than bronze at similar discounts.
- Category effects are milder than tier/margin; expect tree models to lean on margin and discount features.

Regenerate: `python3 scripts/eda_synthetic_quotes.py`
