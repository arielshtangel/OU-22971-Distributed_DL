# Manual Batch-Size Diagnosis

The sweep compares throughput, per-stage step time, and phase timing evidence.

- Baseline run: `baseline_b1` with local batch size 1.
- Follow-up run: `followup_b2` with local batch size 2.
- Best observed run: `followup_b2` with 4.16 images/s.
- Result: the follow-up improved throughput by 1.02 images/s.

Run details:
- baseline_b1: local_batch_size=1, images/s=3.15, stage0_step=1.2714s, stage1_step=1.2391s, diagnosis=communication_or_waiting_visible
- followup_b2: local_batch_size=2, images/s=4.16, stage0_step=1.9219s, stage1_step=1.7836s, diagnosis=communication_or_waiting_visible

Trace interpretation:

- `stage0_forward` and `stage0_backward` represent stage-0 local compute.
- `stage1_forward`, `loss_calculation`, and `loss_backward` represent stage-1 and loss-side compute.
- `send_boundary`, `recv_boundary`, `send_boundary_grad`, and `recv_boundary_grad` show point-to-point transfer and waiting.
- `gather_embeddings`, `grad_sync_stage0`, and `grad_sync_stage1` show collective communication.
