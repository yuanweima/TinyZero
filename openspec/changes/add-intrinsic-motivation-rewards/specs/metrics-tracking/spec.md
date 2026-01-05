# Metrics Tracking Capability

## ADDED Requirements

### Requirement: Response Length Metrics
The system SHALL track response length statistics to measure efficiency.

#### Scenario: Average response length logging
- **WHEN** a training batch completes
- **THEN** the system SHALL log the average response length (in tokens) for that batch

#### Scenario: Response length distribution
- **WHEN** metrics are recorded
- **THEN** the system SHALL track mean, standard deviation, min, and max response lengths
- **AND** these metrics SHALL be visible in tensorboard/wandb

#### Scenario: Length over training time
- **WHEN** training progresses
- **THEN** the system SHALL track response length trends over training steps
- **AND** enable visualization of length compression or expansion

---

### Requirement: Diversity Metrics
The system SHALL track intra-group diversity for GRPO training.

#### Scenario: N-gram diversity computation
- **WHEN** a GRPO batch completes (multiple responses per prompt)
- **THEN** the system SHALL compute average n-gram diversity within each prompt group

#### Scenario: Diversity trend tracking
- **WHEN** training progresses
- **THEN** the system SHALL log diversity metrics over time
- **AND** detect if diversity collapses (converges to single strategy)

#### Scenario: Diversity by correctness
- **WHEN** diversity is computed
- **THEN** the system SHALL separately track diversity among correct and incorrect responses

---

### Requirement: Reflection Pattern Metrics
The system SHALL track the occurrence of self-reflection patterns.

#### Scenario: Reflection rate computation
- **WHEN** a training batch completes
- **THEN** the system SHALL compute the percentage of responses containing reflection patterns

#### Scenario: Pattern breakdown
- **WHEN** reflection metrics are logged
- **THEN** the system SHALL track which specific patterns are most common
- **AND** enable analysis of pattern adoption over training

#### Scenario: Reflection vs correctness correlation
- **WHEN** metrics are recorded
- **THEN** the system SHALL track correlation between reflection presence and answer correctness

---

### Requirement: Grokking Detection
The system SHALL detect sudden performance improvements (grokking).

#### Scenario: Accuracy spike detection
- **WHEN** accuracy increases significantly (>10%) over a short training period (<100 steps)
- **THEN** the system SHALL log a grokking event with timestamp and magnitude

#### Scenario: Grokking ratio tracking
- **WHEN** training progresses
- **THEN** the system SHALL compute grokking ratio (accuracy_delta / step_delta) over sliding windows

#### Scenario: Grokking visualization
- **WHEN** grokking events occur
- **THEN** they SHALL be marked on accuracy plots for easy identification

---

### Requirement: Metrics Dashboard Organization
The system SHALL organize metrics into logical groups for dashboard viewing.

#### Scenario: Metric grouping
- **WHEN** metrics are logged to wandb/tensorboard
- **THEN** they SHALL be organized into groups: `intrinsic/efficiency`, `intrinsic/curiosity`, `intrinsic/reflection`, `intrinsic/grokking`

#### Scenario: Cross-experiment comparison
- **WHEN** multiple experiments complete
- **THEN** metrics SHALL use consistent naming to enable comparison across experiments
