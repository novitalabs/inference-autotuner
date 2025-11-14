# GuideLLM Integration Analysis

## Overview

This document analyzes the [GuideLLM](https://github.com/vllm-project/guidellm) project located at `third_party/guidellm` and identifies key features that could benefit the inference-autotuner project.

GuideLLM is a comprehensive platform for evaluating and optimizing LLM deployments through realistic workload simulation. Unlike genai-bench (which we currently use), GuideLLM provides a **rich Python API** for programmatic integration, making it ideal for our autotuning workflow.

## Key Features Relevant to Inference-Autotuner

### 1. **Rich Python API (vs CLI-only)**

**Current State (genai-bench):**
- CLI-only interface
- Parse JSON output files
- Subprocess management overhead
- Brittle error handling

**GuideLLM Advantage:**
```python
from guidellm.benchmark import benchmark_generative_text, BenchmarkGenerativeTextArgs

# Fully typed, programmatic API
args = BenchmarkGenerativeTextArgs(
    target="http://localhost:8000",
    data=["prompt_tokens=256,output_tokens=128"],
    profile="sweep",
    max_seconds=30
)

# Direct Python integration
report, outputs = await benchmark_generative_text(args)

# Structured results (Pydantic models)
ttft_p50 = report.benchmarks[0].metrics.time_to_first_token.successful.p50
tpot_p99 = report.benchmarks[0].metrics.inter_token_latency.successful.p99
throughput = report.benchmarks[0].metrics.requests_per_second.successful.mean
```

**Benefits:**
- Type-safe configuration
- Structured metric access
- Native async/await support
- No subprocess overhead
- Better error handling

### 2. **Advanced Scheduling Strategies**

GuideLLM supports multiple request scheduling patterns:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `synchronous` | One request at a time | Minimum latency baseline |
| `concurrent` | Fixed parallel streams | Sustained concurrency testing |
| `throughput` | Maximum parallelism | Peak throughput discovery |
| `constant` | Fixed requests/second | SLO validation at specific rates |
| `poisson` | Realistic arrival pattern | Production-like workload simulation |
| `sweep` | Auto-range min→max rates | Comprehensive performance profiling |

**Current genai-bench limitations:**
- Only supports fixed concurrency levels (`num_concurrency`)
- No rate-based scheduling
- Manual rate discovery required

**GuideLLM sweep profile:**
```python
# Automatically finds min/max sustainable rates
# Then runs 10 evenly-spaced benchmarks between them
profile = "sweep"
rate = [10]  # Number of intermediate benchmarks
```

This replaces our manual experimentation with different `num_concurrency` values!

### 3. **Comprehensive Metric Distributions**

**Current genai-bench metrics:**
- Mean, min, max, p50, p75, p90, p95, p99
- Basic token counts
- Request latencies

**GuideLLM metric enhancements:**

```python
# Every metric includes full distribution
metrics = benchmark.metrics.time_to_first_token.successful

# Available statistics
metrics.mean        # Average
metrics.median      # Middle value
metrics.mode        # Most common
metrics.stdev       # Standard deviation
metrics.variance    # Variance
metrics.min / max   # Range
metrics.count       # Sample size
metrics.sum         # Total

# Comprehensive percentiles
metrics.p001  # 0.1th percentile
metrics.p01   # 1st percentile
metrics.p05   # 5th percentile
metrics.p10   # 10th percentile
metrics.p25   # 25th (Q1)
metrics.p50   # 50th (median)
metrics.p75   # 75th (Q3)
metrics.p90   # 90th
metrics.p95   # 95th
metrics.p99   # 99th
metrics.p999  # 99.9th percentile
```

**Additional metrics not in genai-bench:**
- `request_concurrency`: Actual concurrent request count over time
- `decode_time`: Separate decoding phase latency
- `e2e_request_latency`: Full end-to-end timing
- Per-request detailed statistics stored

### 4. **SLO-Aware Benchmarking with Constraints**

GuideLLM has built-in constraint system for intelligent stopping:

```python
# Stop after time limit
max_seconds=30

# Stop after request count
max_requests=1000

# Stop if error rate exceeds threshold
max_error_rate=0.1  # 10% errors (over window)

# Stop after total errors
max_errors=50

# Combine constraints (any triggers stop)
args = BenchmarkGenerativeTextArgs(
    target="http://localhost:8000",
    max_seconds=60,
    max_requests=500,
    max_error_rate=0.05,
    max_errors=10
)
```

**Integration with our SLO system:**
- Can validate SLO compliance during benchmark
- Early stopping if constraints violated
- Better resource efficiency (don't continue failed experiments)

### 5. **Multi-Benchmark Reports**

GuideLLM naturally supports running multiple benchmarks in sequence:

```python
# Sweep profile runs 10+ benchmarks automatically
profile = "sweep"
rate = [10]

# Or custom rate sequence
profile = "constant"
rate = [5.0, 10.0, 15.0, 20.0]

# Single report with all results
report = GenerativeBenchmarksReport(benchmarks=[...])

# Compare across rates/strategies
for benchmark in report.benchmarks:
    strategy = benchmark.scheduler.strategy
    print(f"{strategy.type_} @ rate {strategy.rate}:")
    print(f"  TTFT p99: {benchmark.metrics.time_to_first_token.successful.p99}")
    print(f"  Throughput: {benchmark.metrics.requests_per_second.successful.mean}")
```

**Current approach:**
- One genai-bench run per concurrency level
- Manual aggregation of results
- Complex file management

### 6. **Flexible Data Pipeline**

GuideLLM supports multiple data sources:

**Synthetic Data (like genai-bench):**
```python
data = ["prompt_tokens=256,output_tokens=128,samples=1000"]
```

**HuggingFace Datasets:**
```python
data = ["wikitext", "openai/humaneval"]
data_args = [{"split": "test", "prompt_column": "text"}]
```

**Local Files:**
```python
data = ["conversations.jsonl", "prompts.csv"]
```

**Custom Preprocessing:**
```python
from guidellm.data import DatasetPreprocessor

class CustomPreprocessor(DatasetPreprocessor):
    def preprocess(self, item):
        # Transform data item
        return GenerationRequest(
            type_="chat_completions",
            messages=[{"role": "user", "content": item["text"]}]
        )
```

**Our use case:**
- Currently: Fixed synthetic data format
- With GuideLLM: Support real conversation datasets, custom tokenization, varied distributions

### 7. **Real-Time Progress Tracking**

```python
from guidellm.benchmark import GenerativeConsoleBenchmarkerProgress

# Built-in progress bars and live updates
progress = GenerativeConsoleBenchmarkerProgress()

report, _ = await benchmark_generative_text(args, progress=progress)

# Custom progress handler for web UI integration
class WebUIProgress(BenchmarkerProgress):
    async def on_benchmark_update(self, estimated_state, scheduler_state):
        # Push updates to frontend via WebSocket
        await websocket.send_json({
            "processed": scheduler_state.processed_count,
            "total": scheduler_state.total_count,
            "throughput": estimated_state.get_metric("benchmark_metrics", "requests_per_second"),
            "errors": scheduler_state.error_count
        })
```

**Integration opportunity:**
- Real-time experiment progress in our React frontend
- Live metric streaming during benchmarks
- Better user experience than polling logs

### 8. **HTML Report Generation**

GuideLLM includes a companion UI for visualization:

```python
args = BenchmarkGenerativeTextArgs(
    target="http://localhost:8000",
    output_path="results.html",  # Generates interactive HTML report
    output_formats=["html", "json", "csv"]
)
```

**Features:**
- Interactive charts (request rate, latency distributions, throughput)
- Benchmark comparison views
- Filtering by strategy/time range
- Export to multiple formats

**Integration opportunity:**
- Embed HTML reports in our experiment results view
- Replace custom visualization code with battle-tested UI

### 9. **Extensible Backend System**

```python
@Backend.register("custom_backend")
class CustomBackend(Backend):
    async def resolve(self, request, info):
        # Custom request handling
        response = await self.custom_api_call(request)
        yield response, info
```

**Current limitations:**
- genai-bench: OpenAI-compatible only
- Hard to test with mock servers

**GuideLLM advantages:**
- Plugin architecture for new backends
- Built-in mock server for testing
- Easy to add custom authentication, retries, etc.

### 10. **Poisson Distribution for Realistic Load**

```python
# Realistic request arrival patterns
strategy = AsyncPoissonStrategy(
    rate=10.0,  # Mean requests per second
    random_seed=42  # Reproducible
)
```

**Why this matters:**
- Real-world traffic doesn't arrive at fixed intervals
- Poisson distribution models independent arrivals
- Better SLO validation (captures burst behavior)
- More realistic than fixed-concurrency testing

**Example scenario:**
- Fixed concurrency: Always 10 requests in-flight
- Poisson (rate=10): Sometimes 15 requests, sometimes 5, averages to 10
- Poisson reveals queueing behavior and latency spikes that fixed-concurrency misses

## Comparison: genai-bench vs GuideLLM

| Feature | genai-bench | GuideLLM | Impact |
|---------|-------------|----------|--------|
| **API Type** | CLI only | Python API + CLI | ⭐⭐⭐ Better integration |
| **Scheduling** | Fixed concurrency | 6 strategies (sync/concurrent/throughput/sweep/constant/poisson) | ⭐⭐⭐ More comprehensive testing |
| **Metrics** | Basic distributions | Full distributions + additional metrics | ⭐⭐ Richer analysis |
| **Constraints** | Max requests/time | Constraints + error rates | ⭐⭐ Smarter stopping |
| **Progress** | No real-time | Async iterators + callbacks | ⭐⭐ Live monitoring |
| **Multi-benchmark** | Manual | Built-in (sweep) | ⭐⭐⭐ Auto rate discovery |
| **Data sources** | Synthetic only | Synthetic + HF + files | ⭐ More flexibility |
| **Outputs** | JSON only | JSON/YAML/CSV/HTML | ⭐⭐ Better visualization |
| **Extensibility** | None | Backend/strategy plugins | ⭐ Future-proofing |
| **Type safety** | JSON parsing | Pydantic models | ⭐⭐ Fewer bugs |

**Overall verdict:** GuideLLM is significantly more powerful and better suited for programmatic integration.

## Recommended Integration Strategy

### Phase 1: Side-by-Side (Low Risk)

1. **Add GuideLLM as optional backend**
   - Keep existing genai-bench integration
   - Add new `GuideLLMBenchmarkController`
   - User selects via config: `benchmark_engine: "guidellm"` or `"genai-bench"`

2. **Map existing parameters**
   ```python
   # Current task config
   "benchmark": {
       "traffic_scenarios": ["D(100,100)"],
       "num_concurrency": [1, 4, 8],
       "max_time_per_iteration": 10
   }

   # Maps to GuideLLM
   BenchmarkGenerativeTextArgs(
       data=["prompt_tokens=100,output_tokens=100"],
       profile="concurrent",
       rate=[1, 4, 8],  # Concurrency levels
       max_seconds=10
   )
   ```

3. **Unified metric schema**
   - Extract metrics from `GenerativeBenchmarksReport`
   - Convert to existing format for database storage
   - Maintain backward compatibility

### Phase 2: Enhanced Features (Medium Risk)

1. **Enable sweep profile**
   ```python
   "benchmark": {
       "profile": "sweep",  # New option
       "num_benchmarks": 10,  # Intermediate points
       "traffic_scenarios": ["D(100,100)"]
   }
   ```

2. **Add rate-based strategies**
   ```python
   "benchmark": {
       "profile": "constant",  # or "poisson"
       "rates": [5.0, 10.0, 15.0],  # requests/second
       "traffic_scenarios": ["D(100,100)"]
   }
   ```

3. **Real-time progress streaming**
   - Implement `WebUIProgress` handler
   - Stream metrics via WebSocket to React frontend
   - Live charts during benchmark execution

### Phase 3: Advanced Integration (Higher Risk)

1. **Unified benchmark orchestration**
   - Remove genai-bench dependency
   - GuideLLM becomes primary benchmark engine
   - Simplified codebase

2. **Enhanced SLO validation**
   - Use GuideLLM constraints for early stopping
   - More granular error handling
   - Better resource efficiency

3. **Advanced data pipelines**
   - Support custom datasets
   - User-provided conversation logs
   - Realistic workload replay

## Implementation Example

### New GuideLLMBenchmarkController

```python
# src/controllers/guidellm_benchmark_controller.py

from guidellm.benchmark import benchmark_generative_text, BenchmarkGenerativeTextArgs
import asyncio

class GuideLLMBenchmarkController:
    def __init__(self):
        self.reports = {}

    async def run_benchmark_async(
        self,
        task_name: str,
        experiment_id: int,
        service_name: str,
        namespace: str,
        benchmark_config: Dict[str, Any],
        timeout: int = 600,
        endpoint_url: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Run benchmark using GuideLLM."""

        # Convert config
        args = self._convert_config(benchmark_config, endpoint_url, timeout)

        # Run benchmark
        try:
            report, outputs = await benchmark_generative_text(args)

            # Store report
            benchmark_name = f"{task_name}-exp{experiment_id}"
            self.reports[benchmark_name] = report

            # Convert to our metrics format
            metrics = self._extract_metrics(report)
            return metrics

        except Exception as e:
            print(f"[GuideLLM] Benchmark failed: {e}")
            return None

    def run_benchmark(self, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Synchronous wrapper."""
        return asyncio.run(self.run_benchmark_async(*args, **kwargs))

    def _convert_config(
        self,
        benchmark_config: Dict[str, Any],
        endpoint_url: str,
        timeout: int
    ) -> BenchmarkGenerativeTextArgs:
        """Convert our config to GuideLLM format."""

        # Parse traffic scenario (e.g., "D(100,100)")
        scenario = benchmark_config.get("traffic_scenarios", ["D(100,100)"])[0]
        prompt_tokens, output_tokens = self._parse_scenario(scenario)

        # Determine profile
        profile = benchmark_config.get("profile", "concurrent")

        # Get rates/concurrency
        if profile == "concurrent":
            rate = benchmark_config.get("num_concurrency", [1])
        elif profile in ["constant", "poisson"]:
            rate = benchmark_config.get("rates", [10.0])
        elif profile == "sweep":
            rate = [benchmark_config.get("num_benchmarks", 10)]
        else:
            rate = None

        return BenchmarkGenerativeTextArgs(
            target=endpoint_url,
            data=[f"prompt_tokens={prompt_tokens},output_tokens={output_tokens}"],
            profile=profile,
            rate=rate,
            backend="openai_http",
            model=benchmark_config.get("model_name"),
            processor=benchmark_config.get("model_tokenizer"),
            max_seconds=benchmark_config.get("max_time_per_iteration", timeout),
            max_requests=benchmark_config.get("max_requests_per_iteration"),
            max_error_rate=benchmark_config.get("max_error_rate", 0.1),
            output_path=f"results/{task_name}-exp{experiment_id}.json",
            random_seed=42
        )

    def _extract_metrics(
        self,
        report: GenerativeBenchmarksReport
    ) -> Dict[str, Any]:
        """Convert GuideLLM report to our metrics format."""

        all_metrics = []

        for benchmark in report.benchmarks:
            metrics = benchmark.metrics
            totals = benchmark.request_totals

            # Build metrics dict compatible with our schema
            benchmark_metrics = {
                "scenario": "guidellm",
                "num_concurrency": benchmark.scheduler.strategy.rate
                    if hasattr(benchmark.scheduler.strategy, 'rate') else None,
                "batch_size": 1,
                "iteration_type": benchmark.scheduler.strategy.type_,
                "run_duration": benchmark.duration,

                # Throughput
                "mean_output_throughput_tokens_per_s":
                    metrics.output_token_throughput.successful.mean,
                "mean_input_throughput_tokens_per_s":
                    metrics.input_token_count.successful.mean / benchmark.duration,
                "mean_total_tokens_throughput_tokens_per_s":
                    (metrics.input_token_count.successful.sum +
                     metrics.output_token_count.successful.sum) / benchmark.duration,

                "requests_per_second": metrics.requests_per_second.successful.mean,

                # Request stats
                "num_completed_requests": totals.successful,
                "num_error_requests": totals.errored,
                "num_requests": totals.total,
                "error_rate": totals.errored / totals.total if totals.total > 0 else 0,

                # Latency statistics
                "stats": {
                    "ttft": self._distribution_to_dict(
                        metrics.time_to_first_token.successful
                    ),
                    "tpot": self._distribution_to_dict(
                        metrics.inter_token_latency.successful
                    ),
                    "e2e_latency": self._distribution_to_dict(
                        metrics.request_latency.successful
                    ),
                    "output_latency": self._distribution_to_dict(
                        metrics.decode_time.successful
                    ),
                    "num_input_tokens": self._distribution_to_dict(
                        metrics.input_token_count.successful
                    ),
                    "num_output_tokens": self._distribution_to_dict(
                        metrics.output_token_count.successful
                    ),
                }
            }

            all_metrics.append(benchmark_metrics)

        # Aggregate results
        return {
            "num_result_files": len(all_metrics),
            "raw_results": all_metrics,
            "mean_ttft": sum(m["stats"]["ttft"]["mean"] for m in all_metrics) / len(all_metrics),
            "mean_tpot": sum(m["stats"]["tpot"]["mean"] for m in all_metrics) / len(all_metrics),
            "mean_output_throughput": sum(m["mean_output_throughput_tokens_per_s"] for m in all_metrics) / len(all_metrics),
            "max_output_throughput": max(m["mean_output_throughput_tokens_per_s"] for m in all_metrics),
            # ... more aggregations
        }

    def _distribution_to_dict(self, dist) -> Dict[str, float]:
        """Convert GuideLLM distribution to dict."""
        return {
            "min": dist.min,
            "max": dist.max,
            "mean": dist.mean,
            "median": dist.median,
            "stddev": dist.stdev,
            "sum": dist.sum,
            "p25": dist.p25,
            "p50": dist.p50,
            "p75": dist.p75,
            "p90": dist.p90,
            "p95": dist.p95,
            "p99": dist.p99,
        }
```

## Benefits Summary

### Immediate Benefits (Phase 1)
1. ✅ **Type-safe benchmarking**: Pydantic models eliminate JSON parsing bugs
2. ✅ **Better error handling**: Native Python exceptions vs parsing stderr
3. ✅ **Richer metrics**: Additional percentiles and distributions
4. ✅ **Async native**: Better performance with async/await

### Medium-term Benefits (Phase 2)
1. ✅ **Auto rate discovery**: Sweep profile finds optimal operating ranges automatically
2. ✅ **Real-time monitoring**: Live progress updates to frontend
3. ✅ **Better SLO validation**: Early stopping on constraint violations
4. ✅ **HTML reports**: Interactive visualization for experiment results

### Long-term Benefits (Phase 3)
1. ✅ **Simplified codebase**: Remove genai-bench subprocess management
2. ✅ **Advanced scheduling**: Poisson distribution for realistic workloads
3. ✅ **Custom datasets**: Support user-provided conversation logs
4. ✅ **Extensibility**: Plugin system for future enhancements

## Risks and Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking changes | High | Keep genai-bench as fallback in Phase 1 |
| Performance overhead | Medium | Benchmark both engines, compare overhead |
| Learning curve | Low | GuideLLM API is well-documented |
| Dependency conflicts | Medium | Use virtualenv isolation, test compatibility |
| Migration effort | High | Phased approach, start with optional integration |

## Next Steps

1. **Proof of Concept** (1-2 days)
   - [ ] Create standalone script using GuideLLM
   - [ ] Run identical benchmark with genai-bench and GuideLLM
   - [ ] Compare metrics output
   - [ ] Validate compatibility

2. **Phase 1 Implementation** (1 week)
   - [ ] Implement `GuideLLMBenchmarkController`
   - [ ] Add config option: `benchmark_engine: "guidellm"`
   - [ ] Metric conversion layer
   - [ ] Integration tests

3. **Phase 2 Rollout** (2 weeks)
   - [ ] Enable sweep profile
   - [ ] Real-time progress streaming
   - [ ] HTML report generation
   - [ ] Frontend integration

4. **Phase 3 Migration** (3-4 weeks)
   - [ ] Make GuideLLM default
   - [ ] Deprecate genai-bench
   - [ ] Advanced features (custom datasets, Poisson)
   - [ ] Documentation update

## Conclusion

GuideLLM is a superior choice for benchmarking in the inference-autotuner project:

- **Better API**: Native Python vs subprocess CLI
- **More strategies**: 6 scheduling strategies vs 1
- **Richer metrics**: Comprehensive distributions + additional metrics
- **Auto-discovery**: Sweep profile eliminates manual rate testing
- **Real-time feedback**: Progress tracking for better UX
- **Extensible**: Plugin architecture for future needs

The phased integration strategy minimizes risk while enabling incremental value delivery. Phase 1 provides immediate benefits with low risk, while Phase 2-3 unlock advanced capabilities as confidence grows.

**Recommendation:** Proceed with Phase 1 implementation to evaluate GuideLLM in production workloads before committing to full migration.
