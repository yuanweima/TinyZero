# Change: Add Intrinsic Motivation Rewards (IntrinsicZero)

## Why

Current TinyZero relies solely on outcome-based rewards (correct=1.0, wrong=0.0), which may lead to:
- Memorization over generalization
- Convergence to single solution strategies
- Lack of self-correction capabilities

By injecting biologically-inspired intrinsic motivation signals, we aim to explore whether LLMs can develop:
- Higher-level abstraction and compression (Occam's Razor)
- Diverse problem-solving approaches (Curiosity)
- Self-verification and error correction (Meta-cognition)

## What Changes

### Core Capabilities

1. **Intrinsic Reward Functions** - Three new reward strategies:
   - **Efficiency Reward**: Penalizes verbose reasoning, rewards compression
   - **Curiosity Reward**: Rewards diverse solution paths within GRPO groups
   - **Reflection Reward**: Rewards self-correction patterns in reasoning

2. **Experiment Framework** - Infrastructure for parallel experiments:
   - Configuration-driven experiment selection
   - Baseline (control) + 3 experimental conditions
   - Composable reward function architecture

3. **Metrics & Tracking** - Extended metrics for research:
   - Token efficiency metrics (response length distribution)
   - Diversity metrics (intra-group similarity)
   - Reflection pattern detection and tracking
   - Grokking detection (accuracy vs training steps)

### Affected Files

| File | Change Type |
|------|-------------|
| `verl/utils/reward_score/intrinsic/` | NEW - Intrinsic reward module |
| `verl/utils/reward_score/intrinsic/efficiency.py` | NEW - Occam's Razor reward |
| `verl/utils/reward_score/intrinsic/curiosity.py` | NEW - Curiosity reward |
| `verl/utils/reward_score/intrinsic/reflection.py` | NEW - Meta-cognition reward |
| `verl/utils/reward_score/intrinsic/composer.py` | NEW - Reward composition |
| `verl/trainer/main_ppo.py` | MODIFY - Integrate intrinsic rewards |
| `verl/trainer/ppo/ray_trainer.py` | MODIFY - Add diversity metrics |
| `verl/trainer/config/ppo_trainer.yaml` | MODIFY - Add intrinsic config |
| `examples/data_preprocess/countdown.py` | MINOR - Add experiment tags |
| `scripts/run_intrinsic_experiments.sh` | NEW - Experiment launcher |

## Impact

- **Affected specs**: None (new capability)
- **Breaking changes**: None - all changes are additive
- **Backward compatibility**: Default config preserves original behavior

## Research Hypotheses

| Phase | Hypothesis | Expected Observation |
|-------|------------|---------------------|
| Control | Baseline | 90% accuracy benchmark |
| Phase 1 | Efficiency → Compression | Initial accuracy drop, then Grokking emergence |
| Phase 2 | Curiosity → Diversity | Loss oscillation, improved generalization |
| Phase 3 | Reflection → Self-correction | Longer chains, reduced hallucination |

## Success Criteria

1. All experiments can run in parallel with independent configs
2. Metrics dashboards show clear differentiation between conditions
3. At least one experimental condition shows statistically significant improvement on held-out test set
