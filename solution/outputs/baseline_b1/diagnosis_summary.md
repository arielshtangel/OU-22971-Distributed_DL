# Diagnosis Summary

- Bottleneck category: `roughly balanced`
- Stage-0 max average step time: 1.2714s
- Stage-1 max average step time: 1.2391s
- Estimated throughput: 3.15 images/s
- Explanation: Stage-0 and stage-1 ranks had similar average step times.

Use the profiler traces to inspect compute spans, communication spans, `gather_embeddings`, `loss_calculation`, and waiting around blocking send/recv or collectives.
