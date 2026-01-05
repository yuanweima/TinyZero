# Design: IntrinsicZero - Intrinsic Motivation Rewards

## Context

TinyZero successfully reproduces DeepSeek R1 Zero's self-verification emergence through GRPO training. However, the reward signal is purely outcome-based (binary correctness). This design introduces biologically-inspired intrinsic motivation signals to explore emergent reasoning capabilities.

**Stakeholders**: Researchers exploring LLM reasoning emergence
**Constraints**: Must work with existing VeRL/GRPO infrastructure, single GPU (RTX 3090/4090) feasible

## Goals / Non-Goals

### Goals
- Implement 3 intrinsic reward functions (efficiency, curiosity, reflection)
- Enable parallel experiment execution with independent configs
- Provide rich metrics for research analysis
- Maintain backward compatibility with existing training

### Non-Goals
- Hyperparameter optimization (manual exploration first)
- Multi-task learning (single task focus: Countdown)
- Production deployment considerations

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RewardManager                             │
│  ┌─────────────┐    ┌──────────────────────────────────────┐   │
│  │ Task Reward │    │       IntrinsicRewardComposer        │   │
│  │ (countdown) │    │  ┌──────────┬──────────┬──────────┐  │   │
│  │             │ +  │  │Efficiency│Curiosity │Reflection│  │   │
│  │ score=1/0   │    │  │  Reward  │  Reward  │  Reward  │  │   │
│  └─────────────┘    │  └──────────┴──────────┴──────────┘  │   │
│         │           │              │                        │   │
│         └───────────┴──────────────┘                        │   │
│                      │                                       │   │
│              final_reward = task_reward + intrinsic_bonus    │   │
└─────────────────────────────────────────────────────────────────┘
```

## Decisions

### Decision 1: Additive Reward Composition

**Choice**: `final_reward = task_reward + alpha * intrinsic_reward`

**Alternatives**:
- Multiplicative: `task_reward * (1 + intrinsic_reward)` - Risk of amplifying noise
- Gated: Only apply intrinsic when task correct - Limits exploration

**Rationale**: Additive allows independent tuning, maintains gradient flow even on incorrect responses (for reflection bonus).

### Decision 2: Intrinsic Reward Implementations

#### Efficiency Reward (Occam's Razor)
```python
def efficiency_reward(correctness: bool, token_count: int, config: EfficiencyConfig) -> float:
    if not correctness:
        return 0.0
    # Soft exponential penalty above threshold
    base = config.base_score  # 1.0
    penalty = config.penalty_coef * (token_count ** config.penalty_exp)
    return max(config.min_score, base - penalty)
```

**Config defaults**:
- `penalty_coef`: 0.005
- `penalty_exp`: 1.1
- `min_score`: 0.1
- `threshold_tokens`: 200 (soft reference)

#### Curiosity Reward (Diversity)
```python
def curiosity_reward(response: str, group_responses: List[str],
                     correctness: bool, config: CuriosityConfig) -> float:
    if not correctness:
        return 0.0
    # N-gram based uniqueness within GRPO group
    diversity = calculate_ngram_diversity(response, group_responses, n=config.ngram_n)
    return config.diversity_weight * diversity
```

**Config defaults**:
- `ngram_n`: 3
- `diversity_weight`: 0.5
- `min_group_size`: 2

#### Reflection Reward (Meta-cognition)
```python
def reflection_reward(text: str, correctness: bool, config: ReflectionConfig) -> float:
    patterns = [r"wait", r"hold on", r"let me (?:re)?check", r"no,? that'?s wrong",
                r"incorrect", r"re-?calculate", r"actually"]
    has_reflection = any(re.search(p, text, re.IGNORECASE) for p in patterns)

    intrinsic = config.reflection_bonus if has_reflection else 0.0
    outcome = config.correct_bonus if correctness else 0.0
    return outcome + intrinsic
```

**Config defaults**:
- `reflection_bonus`: 0.1
- `correct_bonus`: 1.0
- `patterns`: configurable list

### Decision 3: GRPO Group Access for Curiosity

**Challenge**: Curiosity reward needs access to all responses in a GRPO group.

**Solution**: Modify `RewardManager.__call__()` to:
1. Group responses by prompt UID (already available in `non_tensor_batch['uid']`)
2. Pass group context to curiosity reward function
3. Compute diversity after all responses decoded

```python
# In RewardManager.__call__()
uid_to_responses = defaultdict(list)
for i, data_item in enumerate(data):
    uid = data_item.non_tensor_batch['uid']
    uid_to_responses[uid].append((i, decoded_response))

for uid, group in uid_to_responses.items():
    for idx, response in group:
        other_responses = [r for i, r in group if i != idx]
        curiosity_score = curiosity_reward(response, other_responses, ...)
```

### Decision 4: Configuration Schema

```yaml
intrinsic_rewards:
  enable: false  # Master switch (default off for backward compat)
  mode: "efficiency"  # efficiency | curiosity | reflection | composite

  efficiency:
    enable: true
    penalty_coef: 0.005
    penalty_exp: 1.1
    min_score: 0.1

  curiosity:
    enable: false
    ngram_n: 3
    diversity_weight: 0.5

  reflection:
    enable: false
    reflection_bonus: 0.1
    patterns: ["wait", "hold on", "let me check", "incorrect"]

  composite:
    weights:
      efficiency: 0.3
      curiosity: 0.3
      reflection: 0.4
```

### Decision 5: Metrics Collection

| Metric | Computation | Purpose |
|--------|-------------|---------|
| `avg_response_length` | Mean token count | Efficiency tracking |
| `response_length_std` | Std of token count | Compression variance |
| `intra_group_diversity` | Mean n-gram diversity | Curiosity effectiveness |
| `reflection_rate` | % responses with patterns | Meta-cognition adoption |
| `grokking_ratio` | accuracy_delta / step_delta | Sudden learning detection |

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|------------|
| Intrinsic rewards dominate task signal | Model ignores correctness | Cap intrinsic contribution at 50% of task reward |
| Reflection gaming (add patterns without reasoning) | False positives | Also check reasoning coherence (future work) |
| Curiosity collapses diversity | Group converges anyway | Track diversity over time, early stop if flat |
| Increased compute for group metrics | Slower training | Batch diversity computation, cache n-grams |

## Migration Plan

1. **Phase 1**: Add intrinsic reward module (no behavior change, `enable: false`)
2. **Phase 2**: Add metrics tracking (visible in logs)
3. **Phase 3**: Enable experiment configs one at a time
4. **Rollback**: Set `intrinsic_rewards.enable: false`

## Experiment Execution Plan

```
┌──────────────────────────────────────────────────────────────────┐
│                    Parallel Experiment Execution                  │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Agent 1: Control (baseline)     ──┐                             │
│  Agent 2: Efficiency Experiment  ──┼──► Independent GPU/runs     │
│  Agent 3: Curiosity Experiment   ──┤                             │
│  Agent 4: Reflection Experiment  ──┘                             │
│                                                                   │
│  Each agent runs:                                                 │
│  1. Load experiment-specific config                               │
│  2. Initialize training with unique experiment_name               │
│  3. Train for N steps                                            │
│  4. Log metrics to separate wandb/tensorboard run                │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Open Questions

1. **Optimal intrinsic reward weights**: Start with small (0.1-0.5), tune based on initial results
2. **Interaction effects**: Should we test composite rewards combining all three?
3. **Task transfer**: Will intrinsic motivation benefits transfer to GSM8K?
