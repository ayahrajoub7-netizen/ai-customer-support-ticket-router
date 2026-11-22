Test set: 5520 messages (1000 out-of-scope). Latency: mean over 200 single-message predictions on Intel64 Family 6 Model 141 Stepping 1, GenuineIntel, Windows, 1 CPU thread, PyTorch 2.15.0+cpu.

| Model | Accuracy | In-scope acc. | OOS recall | Routing (team) acc. | Size (MB) | Latency (ms) |
|---|---|---|---|---|---|---|
| tf-idf + LogReg (baseline) | 78.0% | 90.9% | 19.6% | 81.8% | 39.7 | 0.45 |
| word2vec + FFN | 78.8% | 89.6% | 29.8% | 83.4% | 9.4 | 0.12 |
| BERT-base (FP32) | 89.7% | 95.6% | 63.3% | 91.3% | 439.1 | 47.64 |
| BERT-mini distilled (FP32) | 85.9% | 93.7% | 50.6% | 88.5% | 45.6 | 3.07 |
| BERT-mini distilled (INT8) | 86.4% | 93.3% | 54.8% | 89.2% | 12.3 | 4.62 |
| BERT-base (INT8) | 88.1% | 95.3% | 55.6% | 89.9% | 111.1 | 24.78 |
