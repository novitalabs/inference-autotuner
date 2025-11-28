# Bayesian Optimization with Failed Experiments

## Question

**无穷大的得分是否能对贝叶斯优化产生有效的引导？**

Can infinite scores (inf/-inf) provide effective guidance to Bayesian optimization?

## Short Answer

**No.** Pure infinite scores provide only **weak negative guidance** (what to avoid) but **no positive gradient** (where to go). When all experiments fail, Bayesian optimization degrades to **random search**.

## Detailed Analysis

### How Our System Reports Failed Experiments

In `src/web/workers/autotuner_worker.py` (lines 581-589):

```python
# When experiment fails (timeout, crash, etc.)
objective_name = optimization_config.get("objective", "minimize_latency")
worst_score = float("inf") if "minimize" in objective_name else float("-inf")
strategy.tell_result(
    parameters=params,
    objective_score=worst_score,
    metrics={}
)
```

**For Task 8** (objective: `maximize_throughput`):
- All 50 experiments failed
- All received score = `-inf`
- Logged as: `[Bayesian] Trial complete: score=-inf`

### Theoretical Background: TPE Sampler

Optuna's TPE (Tree-structured Parzen Estimator) sampler works by:

1. **Building surrogate models** for parameter distributions
2. **Separating observations** into "good" (top γ%) and "bad" (rest)
3. **Modeling two distributions**:
   - `l(x)`: density of parameters in "good" trials
   - `g(x)`: density of parameters in "bad" trials
4. **Sampling** from regions where `l(x)/g(x)` is high

**Critical requirement**: Need varying scores to distinguish "good" vs "bad" regions.

### Experiment: TPE Behavior with Infinite Scores

#### Scenario A: All trials return `-inf` (like Task 8)

```python
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
for i in range(15):
    trial = study.ask()
    x = trial.suggest_float('x', 0, 10)
    y = trial.suggest_int('y', 0, 10)
    study.tell(trial, float('-inf'))
```

**Result**:
```
x values: [9.7, 6.0, 2.4, 7.5, 3.4, 7.1, 1.4, 1.3, 0.5, 3.4]
y values: [0, 6, 0, 3, 9, 2, 2, 1, 5, 3]
x variance: 9.71
y variance: 8.10
```

**Interpretation**: High variance indicates **random exploration** with no convergence.

#### Scenario B: Mixed results (some finite scores)

```python
for i in range(25):
    trial = study.ask()
    x = trial.suggest_float('x', 0, 10)
    y = trial.suggest_int('y', 0, 10)

    # Success in optimal region [7≤x≤9, 4≤y≤6] with 80% probability
    if 7 <= x <= 9 and 4 <= y <= 6 and random.random() < 0.8:
        score = 80 + (10 - abs(x-8)) * 5 + (6 - abs(y-5)) * 3
    else:
        score = float('-inf')

    study.tell(trial, score)
```

**Result** (with sufficient successes):
- Later trials converge to optimal region
- TPE learns parameter-score relationship
- Effective optimization

### Why Infinite Scores Don't Work

#### 1. No Discriminative Power

When all trials have the same score (`-inf`):
- All parameter combinations are **equally bad**
- Cannot distinguish "slightly worse" from "much worse"
- No information about which direction to search

#### 2. Cannot Build Surrogate Model

TPE needs to model `P(score | parameters)`:
- With finite scores: Can learn "x=8, y=5 → high score"
- With all `-inf`: All parameters → same score (no relationship)
- Result: Surrogate model is **flat** (uninformative)

#### 3. No Good/Bad Separation

TPE separates trials into "good" (top γ%) and "bad":
- With varying scores: Top 20% are genuinely better
- With all `-inf`: All trials are tied → arbitrary separation
- Result: `l(x)/g(x)` ratio is **meaningless**

### Real-World Evidence: Task 8

From `logs/task_8.log`:

```
[2025-11-27 19:58:16] [INFO] [Bayesian] Trial complete: score=-inf
[2025-11-27 19:58:16] [INFO] [Bayesian] Best so far: score=-inf, params={...}
[2025-11-27 20:08:16] [INFO] [Bayesian] Trial complete: score=-inf
[2025-11-27 20:08:16] [INFO] [Bayesian] Best so far: score=-inf, params={...}
...
[2025-11-27 22:13:30] [INFO] [Bayesian] Trial complete: score=-inf
[2025-11-27 22:13:30] [INFO] [Bayesian] Best so far: score=-inf, params={...}
```

**Observation**:
- All 50 experiments: score = `-inf`
- "Best" trial is arbitrary (first one)
- No convergence in parameter space
- Optimization degraded to **random search**

### Comparison: Grid Search vs Bayesian with All Failures

| Strategy | Behavior with All Failures |
|----------|---------------------------|
| **Grid Search** | Systematic coverage of parameter space |
| **Bayesian (all `-inf`)** | Random sampling (no learning) |
| **Bayesian (some finite)** | Smart exploration → exploitation |

**Key insight**: When all experiments fail, **Grid Search is more reliable** because it guarantees parameter space coverage, while Bayesian becomes random.

## What Information Do Infinite Scores Provide?

### Weak Negative Guidance

Infinite scores tell the optimizer:
- ✓ "This parameter combination failed"
- ✓ "Avoid exact duplicates of this configuration"

But they DON'T provide:
- ✗ "How bad was the failure?" (mildly bad vs catastrophically bad)
- ✗ "Which parameters caused the failure?" (was it `x` or `y` or interaction?)
- ✗ "Which direction should I adjust?" (increase x? decrease y?)

### Mathematical Perspective

Consider the objective function being optimized:

**With finite scores**:
```
f(x=7, y=5) = 85.3
f(x=8, y=5) = 92.1  ← Best so far
f(x=9, y=5) = 87.4
```
**Gradient information**: Increase x from 7→8, but not beyond 8.

**With infinite scores**:
```
f(x=7, y=5) = -inf
f(x=8, y=5) = -inf
f(x=9, y=5) = -inf
```
**No gradient**: All directions equally bad → random walk.

## Recommendations

### 1. Ensure Some Experiments Succeed

For Bayesian optimization to work effectively:
- **Minimum**: At least 3-5 successful experiments (finite scores)
- **Ideal**: 10-20% success rate or higher
- **Critical**: Success rate > 0% (otherwise use Grid Search)

### 2. Adjust Task Configuration

If all experiments fail (like Task 8):

**Root cause**: Traffic scenario `D(3300,150)` too extreme for hardware.

**Solutions**:
- **Reduce traffic load**: `D(512,128)` instead of `D(3300,150)`
- **Lower concurrency**: `[1, 2, 4]` instead of `[6, 8]`
- **Increase timeout**: 900s or 1200s instead of 600s
- **Relax SLO constraints**: Allow more headroom for success

### 3. Consider Alternative Strategies

When success rate is very low (<10%):
- **Grid Search**: More reliable for systematic coverage
- **Random Search**: Equivalent to Bayesian with all failures, but simpler
- **Staged approach**: Grid search first to find feasible region, then Bayesian refinement

### 4. Use Penalty Scores Instead of Infinity

**Current approach** (in our code):
```python
worst_score = float("-inf")  # All failures treated equally
```

**Potential improvement**:
```python
# Assign different penalties based on failure severity
if timeout:
    score = -1000  # Very bad but not infinite
elif crash_early:
    score = -500   # Moderately bad
elif benchmark_failed:
    score = -100   # Mildly bad
```

**Benefits**:
- Provides **gradient information** even among failures
- TPE can learn: "timeouts (x=2) worse than early crashes (x=5)"
- Enables **partial optimization** even without full success

**Trade-offs**:
- More complex scoring logic
- Need to carefully choose penalty magnitudes
- May still struggle if no experiments succeed

## Key Takeaways

1. **Infinite scores provide minimal guidance** to Bayesian optimization
   - Negative guidance: "avoid this"
   - No positive guidance: "try this instead"

2. **When all experiments fail** (all scores = inf/-inf):
   - TPE cannot distinguish parameter quality
   - All combinations treated as equally bad
   - Optimization degrades to **random search**

3. **Bayesian optimization requires** at least some finite scores:
   - Minimum: 3-5 successful experiments
   - Ideal: 10-20% success rate
   - Critical: success_rate > 0

4. **For Task 8's situation** (50/50 failures):
   - Root cause: Configuration too extreme (traffic, timeout, SLO)
   - Solution: Adjust parameters to enable some successes
   - Alternative: Switch to Grid Search for reliability

5. **Future improvement**: Use graded penalties instead of pure infinity
   - Enables learning from failure severity
   - Provides gradient information
   - Better than binary success/failure

## Related Documentation

- `docs/SLO_SCORING.md` - SLO-aware objective scoring
- `docs/TROUBLESHOOTING.md` - Common failure modes
- `CLAUDE.md` - Optimization strategies overview
