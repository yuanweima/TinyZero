# Tasks: IntrinsicZero Implementation

## Dependency Graph

```
Phase 1 (Foundation) ─────────────────────────────────────────────
   │
   ├─► [1.1] Config schema          ◄── Can start immediately
   ├─► [1.2] Efficiency reward      ◄── Can start immediately
   ├─► [1.3] Curiosity reward       ◄── Can start immediately  ├─► PARALLEL
   ├─► [1.4] Reflection reward      ◄── Can start immediately
   └─► [1.5] Unit tests (rewards)   ◄── Can start immediately
           │
Phase 2 (Integration) ────────────────────────────────────────────
           │
   ├─► [2.1] Reward composer        ◄── Depends on 1.2, 1.3, 1.4
   ├─► [2.2] RewardManager update   ◄── Depends on 2.1
   └─► [2.3] Config integration     ◄── Depends on 1.1
           │
Phase 3 (Metrics) ────────────────────────────────────────────────
           │
   ├─► [3.1] Metrics collectors     ◄── Can start after 2.2
   ├─► [3.2] Logging integration    ◄── Depends on 3.1        ├─► PARALLEL
   └─► [3.3] Metrics unit tests     ◄── Can start after 3.1
           │
Phase 4 (Experiments) ────────────────────────────────────────────
           │
   ├─► [4.1] Experiment configs     ◄── Depends on 2.3
   ├─► [4.2] Launch scripts         ◄── Depends on 4.1        ├─► PARALLEL
   └─► [4.3] Integration tests      ◄── Depends on 2.2, 3.2
           │
Phase 5 (Execution) ──────────────────────────────────────────────
           │
   ├─► [5.1] Run Control baseline   ◄──┐
   ├─► [5.2] Run Efficiency exp     ◄──┼── ALL PARALLEL (4 agents)
   ├─► [5.3] Run Curiosity exp      ◄──┤
   └─► [5.4] Run Reflection exp     ◄──┘
```

---

## Phase 1: Foundation (Parallelizable: 4 agents)

### 1.1 Configuration Schema
- [ ] Create `verl/trainer/config/intrinsic_rewards.yaml` with full schema
- [ ] Add type hints and validation for all config fields
- [ ] Document each parameter with expected ranges
- **Files**: `verl/trainer/config/intrinsic_rewards.yaml`
- **Validation**: Config loads without errors, schema validates

### 1.2 Efficiency Reward Function
- [ ] Create `verl/utils/reward_score/intrinsic/__init__.py`
- [ ] Create `verl/utils/reward_score/intrinsic/efficiency.py`
- [ ] Implement `EfficiencyConfig` dataclass
- [ ] Implement `compute_efficiency_reward(correctness, token_count, config)`
- [ ] Handle edge cases: empty response, very long response
- **Files**: `verl/utils/reward_score/intrinsic/efficiency.py`
- **Validation**: Unit tests pass for all edge cases

### 1.3 Curiosity Reward Function
- [ ] Create `verl/utils/reward_score/intrinsic/curiosity.py`
- [ ] Implement `CuriosityConfig` dataclass
- [ ] Implement `calculate_ngram_diversity(response, others, n)`
- [ ] Implement `compute_curiosity_reward(response, group, correctness, config)`
- [ ] Handle single-response groups gracefully
- **Files**: `verl/utils/reward_score/intrinsic/curiosity.py`
- **Validation**: Unit tests pass, diversity=0 for identical responses

### 1.4 Reflection Reward Function
- [ ] Create `verl/utils/reward_score/intrinsic/reflection.py`
- [ ] Implement `ReflectionConfig` dataclass
- [ ] Implement `detect_reflection_patterns(text, patterns)`
- [ ] Implement `compute_reflection_reward(text, correctness, config)`
- [ ] Support configurable pattern list
- **Files**: `verl/utils/reward_score/intrinsic/reflection.py`
- **Validation**: Unit tests pass for pattern detection

### 1.5 Unit Tests for Reward Functions
- [ ] Create `tests/unit/test_intrinsic_rewards.py`
- [ ] Test efficiency reward: correct/incorrect, short/long responses
- [ ] Test curiosity reward: diverse/identical groups, single item groups
- [ ] Test reflection reward: with/without patterns, various pattern matches
- **Files**: `tests/unit/test_intrinsic_rewards.py`
- **Validation**: `pytest tests/unit/test_intrinsic_rewards.py` passes

---

## Phase 2: Integration (Sequential, depends on Phase 1)

### 2.1 Reward Composer
- [ ] Create `verl/utils/reward_score/intrinsic/composer.py`
- [ ] Implement `IntrinsicRewardComposer` class
- [ ] Support mode selection: efficiency | curiosity | reflection | composite
- [ ] Implement weighted composition for composite mode
- [ ] Add enable/disable master switch
- **Files**: `verl/utils/reward_score/intrinsic/composer.py`
- **Validation**: Composer correctly routes to individual reward functions

### 2.2 RewardManager Integration
- [ ] Modify `verl/trainer/main_ppo.py` to import intrinsic rewards
- [ ] Add group-level processing for curiosity reward
- [ ] Extend `RewardManager.__call__()` to apply intrinsic rewards
- [ ] Ensure backward compatibility when `intrinsic_rewards.enable: false`
- **Files**: `verl/trainer/main_ppo.py`
- **Validation**: Training runs with and without intrinsic rewards enabled

### 2.3 Config Integration
- [ ] Modify `verl/trainer/config/ppo_trainer.yaml` to include intrinsic config
- [ ] Add defaults that preserve original behavior
- [ ] Validate config merging with Hydra
- **Files**: `verl/trainer/config/ppo_trainer.yaml`
- **Validation**: `python -m verl.trainer.main_ppo --help` shows new options

---

## Phase 3: Metrics (Partially parallelizable: 2 agents)

### 3.1 Metrics Collectors
- [ ] Create `verl/utils/metrics/intrinsic_metrics.py`
- [ ] Implement `ResponseLengthTracker` (mean, std, distribution)
- [ ] Implement `DiversityTracker` (intra-group n-gram diversity)
- [ ] Implement `ReflectionTracker` (pattern occurrence rate)
- [ ] Implement `GrokkingDetector` (accuracy delta / step delta)
- **Files**: `verl/utils/metrics/intrinsic_metrics.py`
- **Validation**: Metrics compute correctly on sample data

### 3.2 Logging Integration
- [ ] Modify `verl/trainer/ppo/ray_trainer.py` to log intrinsic metrics
- [ ] Add metrics to wandb/tensorboard logging
- [ ] Create metric groups for dashboard organization
- **Files**: `verl/trainer/ppo/ray_trainer.py`
- **Validation**: Metrics appear in training logs

### 3.3 Metrics Unit Tests
- [ ] Create `tests/unit/test_intrinsic_metrics.py`
- [ ] Test each tracker with synthetic data
- [ ] Test grokking detection edge cases
- **Files**: `tests/unit/test_intrinsic_metrics.py`
- **Validation**: `pytest tests/unit/test_intrinsic_metrics.py` passes

---

## Phase 4: Experiment Setup (Parallelizable: 2 agents)

### 4.1 Experiment Configs
- [ ] Create `configs/experiments/control.yaml` (baseline)
- [ ] Create `configs/experiments/efficiency.yaml`
- [ ] Create `configs/experiments/curiosity.yaml`
- [ ] Create `configs/experiments/reflection.yaml`
- [ ] Create `configs/experiments/composite.yaml` (all three)
- **Files**: `configs/experiments/*.yaml`
- **Validation**: Each config loads and validates

### 4.2 Launch Scripts
- [ ] Create `scripts/run_intrinsic_experiments.sh`
- [ ] Support parallel launch of multiple experiments
- [ ] Add experiment naming with timestamps
- [ ] Include GPU assignment for parallel runs
- **Files**: `scripts/run_intrinsic_experiments.sh`
- **Validation**: Script syntax check passes

### 4.3 Integration Tests
- [ ] Create `tests/e2e/test_intrinsic_training.py`
- [ ] Test short training run with each reward type
- [ ] Verify metrics are logged correctly
- [ ] Test config override mechanism
- **Files**: `tests/e2e/test_intrinsic_training.py`
- **Validation**: E2E tests pass on sample data

---

## Phase 5: Experiment Execution (Fully Parallel: 4 agents)

> **Note**: Each experiment can run on a separate GPU/agent simultaneously.

### 5.1 Control Baseline
- [ ] Run baseline training (no intrinsic rewards)
- [ ] Train until 90% accuracy or max steps
- [ ] Record: steps to 90%, final accuracy, avg response length
- **Config**: `configs/experiments/control.yaml`
- **Validation**: Training completes, metrics logged

### 5.2 Efficiency Experiment
- [ ] Run efficiency reward training
- [ ] Monitor for Grokking patterns
- [ ] Record: accuracy curve, response length over time
- **Config**: `configs/experiments/efficiency.yaml`
- **Validation**: Training completes, efficiency metrics logged

### 5.3 Curiosity Experiment
- [ ] Run curiosity reward training
- [ ] Monitor diversity metrics
- [ ] Record: diversity over time, test set generalization
- **Config**: `configs/experiments/curiosity.yaml`
- **Validation**: Training completes, diversity metrics logged

### 5.4 Reflection Experiment
- [ ] Run reflection reward training
- [ ] Monitor reflection rate
- [ ] Record: reflection rate, accuracy, response length
- **Config**: `configs/experiments/reflection.yaml`
- **Validation**: Training completes, reflection metrics logged

---

## Parallelization Summary

| Phase | Tasks | Max Parallel Agents |
|-------|-------|---------------------|
| 1 | 1.1, 1.2, 1.3, 1.4, 1.5 | **4** (reward functions independent) |
| 2 | 2.1, 2.2, 2.3 | 1 (sequential dependencies) |
| 3 | 3.1+3.3, 3.2 | **2** (metrics and tests parallel) |
| 4 | 4.1+4.2, 4.3 | **2** (configs and tests parallel) |
| 5 | 5.1, 5.2, 5.3, 5.4 | **4** (experiments fully parallel) |

**Total estimated parallel speedup**: ~3-4x with 4 agents
