# DPO Final Validation Evaluation

This evaluation does not execute functional assertions. It measures syntax, format,
and whether the first generated top-level interface matches the `chosen` answer.

| Model | Syntax | Has Interface | Name | Signature | Code Only | Avg Lines | Avg Chars | Line Delta vs Chosen | Char Delta vs Chosen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 150/150 (100.0%) | 150/150 (100.0%) | 150/150 (100.0%) | 148/150 (98.7%) | 1/150 (0.7%) | 5.08 | 173.9 | +1.31 | +35.4 |
| sft | 150/150 (100.0%) | 150/150 (100.0%) | 150/150 (100.0%) | 147/150 (98.0%) | 150/150 (100.0%) | 4.59 | 160.0 | +0.82 | +21.4 |
| dpo | 150/150 (100.0%) | 150/150 (100.0%) | 150/150 (100.0%) | 147/150 (98.0%) | 150/150 (100.0%) | 4.48 | 155.2 | +0.71 | +16.6 |

## By Pair Type

| Pair Type | base | sft | dpo |
|---|---:|---:|---:|
| correctness_protection | sig 100.0%, code 1.7% | sig 100.0%, code 100.0% | sig 100.0%, code 100.0% |
| interface_and_format_protection | sig 87.5%, code 0.0% | sig 81.2%, code 100.0% | sig 81.2%, code 100.0% |
| simplicity_preference | sig 100.0%, code 0.0% | sig 100.0%, code 100.0% | sig 100.0%, code 100.0% |

##  Functional accuracy
base = 99/150 = 66.0%, sft = 101/150 = 67.3%, dpo = 101/150 = 67.3%
