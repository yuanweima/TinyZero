# Intrinsic Rewards Capability

## ADDED Requirements

### Requirement: Efficiency Reward Function
The system SHALL provide an efficiency reward function that penalizes verbose reasoning while rewarding correct answers with concise explanations.

#### Scenario: Correct answer with short response
- **WHEN** the model produces a correct answer with token count below threshold
- **THEN** the efficiency reward SHALL be close to the base score (1.0)

#### Scenario: Correct answer with long response
- **WHEN** the model produces a correct answer with token count significantly above threshold
- **THEN** the efficiency reward SHALL be reduced according to the exponential penalty formula
- **AND** the reward SHALL NOT fall below the configured minimum score

#### Scenario: Incorrect answer
- **WHEN** the model produces an incorrect answer regardless of length
- **THEN** the efficiency reward SHALL be 0.0

---

### Requirement: Curiosity Reward Function
The system SHALL provide a curiosity reward function that rewards diverse solution paths within GRPO response groups.

#### Scenario: Unique correct solution in group
- **WHEN** the model produces a correct answer that differs significantly from other group responses
- **THEN** the curiosity reward SHALL include a diversity bonus proportional to the uniqueness score

#### Scenario: Duplicate correct solution in group
- **WHEN** the model produces a correct answer that is similar to other group responses
- **THEN** the curiosity reward SHALL have minimal or zero diversity bonus

#### Scenario: Incorrect answer with unique reasoning
- **WHEN** the model produces an incorrect answer regardless of uniqueness
- **THEN** the curiosity reward SHALL be 0.0

#### Scenario: Single response in group
- **WHEN** a prompt has only one response in the group
- **THEN** the system SHALL handle gracefully with zero diversity bonus

---

### Requirement: Reflection Reward Function
The system SHALL provide a reflection reward function that rewards self-correction patterns in reasoning.

#### Scenario: Correct answer with reflection patterns
- **WHEN** the model produces a correct answer AND the response contains reflection patterns (e.g., "wait", "let me check", "actually")
- **THEN** the reflection reward SHALL include both the correctness bonus AND the reflection bonus

#### Scenario: Correct answer without reflection
- **WHEN** the model produces a correct answer WITHOUT reflection patterns
- **THEN** the reflection reward SHALL include only the correctness bonus

#### Scenario: Incorrect answer with reflection
- **WHEN** the model produces an incorrect answer WITH reflection patterns
- **THEN** the reflection reward SHALL include only the reflection bonus (intrinsic motivation)

#### Scenario: Incorrect answer without reflection
- **WHEN** the model produces an incorrect answer WITHOUT reflection patterns
- **THEN** the reflection reward SHALL be 0.0

---

### Requirement: Reward Composer
The system SHALL provide a reward composer that combines task rewards with intrinsic motivation rewards.

#### Scenario: Single intrinsic reward mode
- **WHEN** the configuration specifies a single reward mode (efficiency, curiosity, or reflection)
- **THEN** the composer SHALL apply only that intrinsic reward function

#### Scenario: Composite reward mode
- **WHEN** the configuration specifies composite mode with weights
- **THEN** the composer SHALL apply weighted combination of all enabled intrinsic rewards

#### Scenario: Intrinsic rewards disabled
- **WHEN** the intrinsic rewards master switch is disabled
- **THEN** the system SHALL behave identically to the original implementation

---

### Requirement: Intrinsic Reward Configuration
The system SHALL provide comprehensive configuration options for intrinsic rewards.

#### Scenario: Default configuration
- **WHEN** no intrinsic reward configuration is provided
- **THEN** the system SHALL use defaults that disable intrinsic rewards (backward compatible)

#### Scenario: Override configuration via Hydra
- **WHEN** intrinsic reward parameters are provided via command line or config file
- **THEN** the system SHALL apply the specified parameters

#### Scenario: Invalid configuration
- **WHEN** invalid intrinsic reward configuration is provided
- **THEN** the system SHALL raise a clear validation error before training starts
