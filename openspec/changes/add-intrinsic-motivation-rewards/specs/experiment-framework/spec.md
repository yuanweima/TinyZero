# Experiment Framework Capability

## ADDED Requirements

### Requirement: Experiment Configuration Management
The system SHALL support independent experiment configurations for parallel execution.

#### Scenario: Load experiment-specific config
- **WHEN** an experiment config file is specified (e.g., `configs/experiments/efficiency.yaml`)
- **THEN** the system SHALL load and merge the experiment config with base config
- **AND** experiment-specific settings SHALL override base settings

#### Scenario: Unique experiment naming
- **WHEN** an experiment is launched
- **THEN** the system SHALL generate a unique experiment name combining condition name and timestamp
- **AND** logs and checkpoints SHALL be stored in experiment-specific directories

#### Scenario: Parallel experiment execution
- **WHEN** multiple experiments are launched simultaneously
- **THEN** each experiment SHALL operate independently without shared state
- **AND** each experiment SHALL log to its own metrics endpoint

---

### Requirement: Baseline Control Experiment
The system SHALL support a control experiment configuration that reproduces original TinyZero behavior.

#### Scenario: Control experiment execution
- **WHEN** the control experiment config is used
- **THEN** the training SHALL use only task-based rewards (no intrinsic motivation)
- **AND** metrics SHALL establish baseline performance benchmarks

#### Scenario: Control vs experimental comparison
- **WHEN** control and experimental results are available
- **THEN** the system SHALL log metrics in a format suitable for statistical comparison

---

### Requirement: Experiment Launch Automation
The system SHALL provide scripts for launching parallel experiments.

#### Scenario: Single experiment launch
- **WHEN** `scripts/run_intrinsic_experiments.sh efficiency` is executed
- **THEN** the efficiency experiment SHALL start with correct configuration

#### Scenario: All experiments launch
- **WHEN** `scripts/run_intrinsic_experiments.sh all` is executed
- **THEN** all four experiments (control, efficiency, curiosity, reflection) SHALL launch in parallel
- **AND** each experiment SHALL be assigned to available GPU resources

#### Scenario: GPU assignment
- **WHEN** multiple experiments run on a multi-GPU system
- **THEN** the script SHALL assign each experiment to a different GPU via CUDA_VISIBLE_DEVICES
