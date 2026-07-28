# EgoRecovery — Code & Visualization

## Response to Reviewer "2ftg" W1, Reviewer "tMkX" Q2 & W1

###  OOD human recovery under cluttered interference

<p align="center"><img src="assets/rebuttal_figures/table_tg_ood_generalization.png" alt="Out-of-domain human recovery under cluttered interference" width="70%"></p>

The robot data contains no out-of-domain (OOD) conditions. Human recovery is collected either in matched scenes or in new environments, with different objects, and under clutter; all models are evaluated under cluttered-object interference using robot observations only. OOD human recovery improves both tasks over robot-only recovery and same-scene human recovery. 
### Demonstrations under cluttered interference
<table><tr>
<td><strong>Other environment / human success</strong><br><img src="assets/demo_gif/ego data in other environment/human success/3.gif" alt="Human success collection in another environment" width="100%"></td>
<td><strong>Other environment / human recovery</strong><br><img src="assets/demo_gif/ego data in other environment/human recovery/2.gif" alt="Human recovery collection in another environment" width="100%"></td>
</tr><tr>
<td><strong>Different object / human success</strong><br><img src="assets/demo_gif/ego data with diferrent objects/human success/5.gif" alt="Human success collection with a different object" width="100%"></td>
<td><strong>Different object / human recovery</strong><br><img src="assets/demo_gif/ego data with diferrent objects/human recovery/6.gif" alt="Human recovery collection with a different object" width="100%"></td>
</tr><tr>
<td><strong>Clutter / human success</strong><br><img src="assets/demo_gif/ego data with interference/human success/4.gif" alt="Human success collection under clutter" width="100%"></td>
<td><strong>Clutter / human recovery</strong><br><img src="assets/demo_gif/ego data with interference/human recovery/2.gif" alt="Human recovery collection under clutter" width="100%"></td>
</tr><tr>
<td><strong>Cup-brush inference / 1</strong><br><img src="assets/demo_gif/inference with interference/task1/1.gif" alt="Cup-brush inference under clutter, example 1" width="100%"></td>
<td><strong>Cup-brush inference / 2</strong><br><img src="assets/demo_gif/inference with interference/task1/2.gif" alt="Cup-brush inference under clutter, example 2" width="100%"></td>
</tr><tr>
<td><strong>Round-disk inference / 1</strong><br><img src="assets/demo_gif/inference with interference/task2/1.gif" alt="Round-disk inference under clutter, example 1" width="100%"></td>
<td><strong>Round-disk inference / 2</strong><br><img src="assets/demo_gif/inference with interference/task2/2.gif" alt="Round-disk inference under clutter, example 2" width="100%"></td>
</tr></table>


## Response to Reviewer "2ftg" W2 & Q2, Reviewer "tMkX" Q4 & W2

###  Closed-loop mechanism ablation

<p align="center"><img src="assets/rebuttal_figures/table_2_mechanism_ablation.png" alt="Closed-loop mechanism ablation" width="78%"></p>

Removing intent modulation, or zeroing the predicted intent at test time, reduces Recovery SR to **65.0%** even when intent can still be predicted. The intent therefore needs a path to the executed robot action rather than serving only as an auxiliary target.

###  Corrective-intent target comparison

<p align="center"><img src="assets/rebuttal_figures/table_ta_intent_representation.png" alt="Corrective-intent target comparison" width="70%"></p>

Only the intent-supervision target changes in this control; the gate, FiLM modulation, data, and training procedure are fixed. DCT-4D performs best across Initial and Recovery SR on both tasks, supporting a compact low-frequency magnitude representation rather than raw trajectories or learned observation/trajectory latents.

## Response to Reviewer "2ftg" W3 & Q1, Reviewer "VtdJ" Q7

### Redrawn Figure 6: Cost-matched and low-robot-grounding budgets

![Redrawn Figure 6: recovery success under cost-matched and low-robot-grounding budgets](assets/fig5-6/exp_budget_combined.png)

Panel (a) fixes the robot-equivalent recovery-collection budget at `B_RobotEq = R_rec + H_rec / 10 = 80`; the 50 robot-success episodes are fixed outside the budget. The moderate mixture `R_rec=50, H_rec=300` reaches **85.0% Recovery SR**, compared with **60.0%** for robot-only `R_rec=80, H_rec=0`. Panel (b) instead fixes a low robot-grounding set at `R_rec=20` and adds human recovery.The reported **10×** is a collection-throughput ratio, not a claim that 10× fewer training samples are required.

<!-- >###  Fixed-cost comparison

<p align="center"><img src="assets/rebuttal_figures/table_td_cost_matched.png" alt="Cost-matched robot-only recovery and EgoRecovery comparison" width="70%"></p>

Both rows consume the same measured robot-equivalent recovery-collection cost. The mixed dataset improves Recovery SR by **25.0 percentage points**. -->

## Response to Reviewer "2ftg" W4, Reviewer "VtdJ" W & Q5

###  Recovery supervision versus success-only data

<p align="center"><img src="assets/rebuttal_figures/table_6_recovery_supervision.png" alt="Initial and recovery success rates across recovery-supervision conditions" width="78%"></p>

The first two rows show that robot success data alone and robot success plus 300 human-success demonstrations both obtain **0.0% Recovery SR** from staged failure starts. Human recovery without robot recovery grounding is also limited (**8.8%**); robot recovery raises Recovery SR to **52.5%**, and combining grounded robot recovery with the proposed pathway reaches **85.0%**.

###  Failure-state coverage

<table>
<tr>
<td align="center" width="50%"><img src="assets/demo/recovery_scope/recovery_scope_overview.png" alt="Qualitative failure-state coverage: human recovery demonstrations surround a compact robot recovery set" height="320"></td>
<td align="center" width="50%"><img src="assets/visualization.png" alt="Failure-state coverage in the learned intent-feature space" height="320"></td>
</tr>
</table>

Representative frames and learned intent-feature distributions show that adding 300 human recovery episodes to 50 robot recovery episodes expands convex-hull volume by **15.4×**. 


## Response to Reviewer "VtdJ" Q6, Reviewer "tMkX" Q1

###  Test-time recovery-gate intervention

<p align="center"><img src="assets/rebuttal_figures/table_tb_gate_causality.png" alt="Test-time recovery-gate intervention" width="68%"></p>

The same final checkpoint is evaluated without retraining while `p_t` is replaced by zero, a constant, one, or a shuffled sequence. Only the learned gate preserves both nominal execution and recovery.

### Closed-loop recovery-gate visualization from a failure start
<p align="center"><img src="assets/demo_gif/visualization/1.gif" alt="Closed-loop recovery gate from a failure start" width="50%"></p>

<p align="center"><img src="assets/demo_gif/visualization/2.gif" alt="Closed-loop recovery gate during nominal execution" width="50%"></p>


## Response to Reviewer "VtdJ" Q8

###  Recovery pathway with a Diffusion Policy head

![Recovery pathway with a Diffusion Policy head](assets/rebuttal_figures/table_tc_diffusion_policy.png)

The action decoder is replaced by a Diffusion Policy head and evaluated in closed loop from the same off-nominal starts. Success-only Diffusion Policy obtains **0% Recovery SR** on both tasks; robot recovery helps, and the complete EgoRecovery pathway yields the strongest performance in this comparison.
### Demonstrations with DP backbone
<table><tr>
<td><strong>Cup-brush / 1</strong><br><img src="assets/demo_gif/inference with DP backbone/task1/1.gif" alt="Cup-brush recovery with Diffusion Policy, example 1" width="100%"></td>
<td><strong>Cup-brush / 2</strong><br><img src="assets/demo_gif/inference with DP backbone/task1/2.gif" alt="Cup-brush recovery with Diffusion Policy, example 2" width="100%"></td>
</tr><tr>
<td><strong>Cup-brush / 3</strong><br><img src="assets/demo_gif/inference with DP backbone/task1/4.gif" alt="Cup-brush recovery with Diffusion Policy, example 3" width="100%"></td>
<td><strong>Cup-brush / 4</strong><br><img src="assets/demo_gif/inference with DP backbone/task1/5.gif" alt="Cup-brush recovery with Diffusion Policy, example 4" width="100%"></td>
</tr><tr>
<td><strong>Round-disk / 1</strong><br><img src="assets/demo_gif/inference with DP backbone/task2/1.gif" alt="Round-disk recovery with Diffusion Policy, example 1" width="100%"></td>
<td><strong>Round-disk / 2</strong><br><img src="assets/demo_gif/inference with DP backbone/task2/2.gif" alt="Round-disk recovery with Diffusion Policy, example 2" width="100%"></td>
</tr><tr>
<td><strong>Round-disk / 3</strong><br><img src="assets/demo_gif/inference with DP backbone/task2/3.gif" alt="Round-disk recovery with Diffusion Policy, example 3" width="100%"></td>
<td><strong>Round-disk / 4</strong><br><img src="assets/demo_gif/inference with DP backbone/task2/4.gif" alt="Round-disk recovery with Diffusion Policy, example 4" width="100%"></td>
</tr></table>


## Response to Reviewer "VtdJ" Q4

###  Recovery-boundary annotation robustness

<p align="center"><img src="assets/rebuttal_figures/table_te_annotation_robustness.png" alt="Recovery-boundary annotation robustness" width="60%"></p>

A motion-energy curve proposes `t_rec`, an annotator confirms or adjusts it, and the model is retrained after injecting boundary noise. Recovery SR is unchanged at ±3 frames and decreases by only 5 percentage points at ±5 frames.


## Response to Reviewer "tMkX" Q3 & W1

###  Multiple failure modes with one shared pathway

<p align="center"><img src="assets/rebuttal_figures/table_tf_multiple_failures.png" alt="Multiple failure modes with one shared pathway" width="80%"></p>

One model per task, with a shared gate and corrective-intent pathway, is trained across two distinct failure modes. EgoRecovery improves placement/insertion and retreat-and-re-approach recovery while preserving Initial SR.
### Multi-failure collection & inference
<table><tr>
<td><strong>Collection / Cup-brush / 1</strong><br><img src="assets/demo_gif/multi_failures/collection/task1/1.gif" alt="Cup-brush multi-failure collection, example 1" width="100%"></td>
<td><strong>Collection / Cup-brush / 2</strong><br><img src="assets/demo_gif/multi_failures/collection/task1/2.gif" alt="Cup-brush multi-failure collection, example 2" width="100%"></td>
</tr><tr>
<td><strong>Collection / Round-disk / 1</strong><br><img src="assets/demo_gif/multi_failures/collection/task2/1.gif" alt="Round-disk multi-failure collection, example 1" width="100%"></td>
<td><strong>Collection / Round-disk / 2</strong><br><img src="assets/demo_gif/multi_failures/collection/task2/2.gif" alt="Round-disk multi-failure collection, example 2" width="100%"></td>
</tr><tr>
<td><strong>Inference / Cup-brush / 1</strong><br><img src="assets/demo_gif/multi_failures/inference/task1/1.gif" alt="Cup-brush multi-failure inference, example 1" width="100%"></td>
<td><strong>Inference / Cup-brush / 2</strong><br><img src="assets/demo_gif/multi_failures/inference/task1/2.gif" alt="Cup-brush multi-failure inference, example 2" width="100%"></td>
</tr><tr>
<td><strong>Inference / Round-disk / 1</strong><br><img src="assets/demo_gif/multi_failures/inference/task2/1.gif" alt="Round-disk multi-failure inference, example 1" width="100%"></td>
<td><strong>Inference / Round-disk / 2</strong><br><img src="assets/demo_gif/multi_failures/inference/task2/2.gif" alt="Round-disk multi-failure inference, example 2" width="100%"></td>
</tr></table>


## Demonstrations

### Standard task demonstrations

#### Task 1 | Cup-brush insertion

<table><tr><th colspan="2">Collection</th></tr><tr>
<td><strong>Human success</strong><br><img src="assets/demo_gif/collection/t1_hand_success/episode_000001.gif" alt="Cup-brush human success collection" width="100%"></td>
<td><strong>Human recovery</strong><br><img src="assets/demo_gif/collection/t1_hand_recovery/episode_000029.gif" alt="Cup-brush human recovery collection" width="100%"></td>
</tr><tr><td><strong>Robot success</strong><br><img src="assets/demo_gif/collection/t1_robot_success/episode_000029.gif" alt="Cup-brush robot success collection" width="100%"></td>
<td><strong>Robot recovery</strong><br><img src="assets/demo_gif/collection/t1_robot_recovery/episode_000002.gif" alt="Cup-brush robot recovery collection" width="100%"></td>
</tr><tr><th colspan="2">Inference</th></tr><tr><td><strong>Recovery example 1</strong><br><img src="assets/demo_gif/inference/t1_robot_recovery/1.gif" alt="Cup-brush robot recovery inference, example 1" width="100%"></td>
<td><strong>Recovery example 2</strong><br><img src="assets/demo_gif/inference/t1_robot_recovery/3.gif" alt="Cup-brush robot recovery inference, example 2" width="100%"></td></tr></table>

#### Task 2 | Table sweep

<table><tr><th colspan="2">Collection</th></tr><tr>
<td><strong>Human success</strong><br><img src="assets/demo_gif/collection/t2_hand_success/episode_000001.gif" alt="Table-sweep human success collection" width="100%"></td>
<td><strong>Human recovery</strong><br><img src="assets/demo_gif/collection/t2_hand_recovery/episode_000002.gif" alt="Table-sweep human recovery collection" width="100%"></td>
</tr><tr><td><strong>Robot success</strong><br><img src="assets/demo_gif/collection/t2_robot_success/episode_000016.gif" alt="Table-sweep robot success collection" width="100%"></td>
<td><strong>Robot recovery</strong><br><img src="assets/demo_gif/collection/t2_robot_recovery/episode_000005.gif" alt="Table-sweep robot recovery collection" width="100%"></td>
</tr><tr><th colspan="2">Inference</th></tr><tr><td><strong>Recovery example 1</strong><br><img src="assets/demo_gif/inference/t2_robot_recovery/2.gif" alt="Table-sweep robot recovery inference, example 1" width="100%"></td>
<td><strong>Recovery example 2</strong><br><img src="assets/demo_gif/inference/t2_robot_recovery/3.gif" alt="Table-sweep robot recovery inference, example 2" width="100%"></td></tr></table>

#### Task 3 | Round-disk placement

<table><tr><th colspan="2">Collection</th></tr><tr>
<td><strong>Human success</strong><br><img src="assets/demo_gif/collection/t3_hand_success/episode_000024.gif" alt="Round-disk human success collection" width="100%"></td>
<td><strong>Human recovery</strong><br><img src="assets/demo_gif/collection/t3_hand_recovery/episode_000001.gif" alt="Round-disk human recovery collection" width="100%"></td>
</tr><tr><td><strong>Robot success</strong><br><img src="assets/demo_gif/collection/t3_robot_success/episode_000010.gif" alt="Round-disk robot success collection" width="100%"></td>
<td><strong>Robot recovery</strong><br><img src="assets/demo_gif/collection/t3_robot_recovery/episode_000001.gif" alt="Round-disk robot recovery collection" width="100%"></td>
</tr><tr><th colspan="2">Inference</th></tr><tr><td><strong>Recovery example 1</strong><br><img src="assets/demo_gif/inference/t3_robot_recovery/2.gif" alt="Round-disk robot recovery inference, example 1" width="100%"></td>
<td><strong>Recovery example 2</strong><br><img src="assets/demo_gif/inference/t3_robot_recovery/3.gif" alt="Round-disk robot recovery inference, example 2" width="100%"></td></tr></table>

#### Task 4 | Cube stacking

<table><tr><th colspan="2">Collection</th></tr><tr>
<td><strong>Human success</strong><br><img src="assets/demo_gif/collection/t4_hand_success/1.gif" alt="Cube-stacking human success collection" width="100%"></td>
<td><strong>Human recovery</strong><br><img src="assets/demo_gif/collection/t4_hand_recovery/1.gif" alt="Cube-stacking human recovery collection" width="100%"></td>
</tr><tr><td><strong>Robot success</strong><br><img src="assets/demo_gif/collection/t4_robot_success/1.gif" alt="Cube-stacking robot success collection" width="100%"></td>
<td><strong>Robot recovery</strong><br><img src="assets/demo_gif/collection/t4_robot_recovery/4.gif" alt="Cube-stacking robot recovery collection" width="100%"></td>
</tr><tr><th colspan="2">Inference</th></tr><tr><td><strong>Recovery example 1</strong><br><img src="assets/demo_gif/inference/t4_robot_recovery/1.gif" alt="Cube-stacking robot recovery inference, example 1" width="100%"></td>
<td><strong>Recovery example 2</strong><br><img src="assets/demo_gif/inference/t4_robot_recovery/3.gif" alt="Cube-stacking robot recovery inference, example 2" width="100%"></td></tr></table>
