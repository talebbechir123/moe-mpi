| Model | Total params | Active params / token | Val loss | Perplexity | Tokens/s |
|---|---|---|---|---|---|
| Dense | 5.3M | 5.3M | 2.107 | 8.22 | 5,264 |
| MoE (8 experts, top-1) | 20.0M | 5.3M | 2.059 | 7.84 | 4,421 |

MoE perplexity is 4.7% lower; throughput is 16% lower (routing + all-to-all).
