# GenAI-Bench Logs - Display Options

This document explains how to view genai-bench logs during autotuner execution.

## Overview

The autotuner now supports multiple ways to view genai-bench output during benchmark execution:

1. **Default Mode** - Shows output after completion
2. **Verbose Mode** - Streams output in real-time
3. **Manual Inspection** - View logs from benchmark output directory

## Option 1: Default Mode (Post-Completion)

By default, the autotuner shows genai-bench stdout/stderr after each benchmark completes:

```bash
python src/run_autotuner.py examples/docker_task.json --mode docker --direct
```

**Output example:**
```
[Benchmark] Completed in 236.6s
[Benchmark] Exit code: 0
[Benchmark] STDOUT:
<genai-bench output here>
[Benchmark] STDERR:
<any warnings or errors>
```

**Pros:**
- Clean output, easier to read
- Default behavior, no extra flags needed
- Full output available in logs

**Cons:**
- No real-time feedback during long benchmarks
- Cannot see progress until completion

## Option 2: Verbose Mode (Real-Time Streaming)

Use the `--verbose` or `-v` flag to stream genai-bench output in real-time:

```bash
python src/run_autotuner.py examples/docker_task.json --mode docker --direct --verbose
```

**Output example:**
```
[Benchmark] Starting genai-bench (streaming output)...
[genai-bench] Loading tokenizer...
[genai-bench] Connecting to API endpoint...
[genai-bench] Running scenario D(100,100) with concurrency 1...
[genai-bench] Progress: 10/50 requests completed
[genai-bench] Progress: 20/50 requests completed
...
[Benchmark] Completed in 236.6s
[Benchmark] Exit code: 0
```

**Pros:**
- Real-time feedback
- See progress during long benchmarks
- Useful for debugging connection/API issues
- Can detect problems early

**Cons:**
- More verbose output
- Harder to grep/filter in logs

## Option 3: Manual Inspection

View genai-bench logs directly from the output directory:

```bash
# View all benchmark result files
ls -R benchmark_results/

# View specific experiment results
cat benchmark_results/docker-simple-tune-exp1/experiment_metadata.json

# View genai-bench detailed logs (if available)
cat benchmark_results/docker-simple-tune-exp1/*.log
```

## Usage Examples

### Example 1: Quick Testing (Default)
```bash
# Standard run - see output after completion
python src/run_autotuner.py examples/docker_task.json --mode docker --direct
```

### Example 2: Debugging Connection Issues (Verbose)
```bash
# Stream output to see where it fails
python src/run_autotuner.py examples/docker_task.json --mode docker --direct --verbose
```

### Example 3: Long-Running Benchmarks (Verbose + Log File)
```bash
# Stream output and save to file for later analysis
python src/run_autotuner.py examples/docker_task.json --mode docker --direct --verbose 2>&1 | tee autotuner.log
```

### Example 4: Debugging with Reduced Workload
```bash
# Use verbose mode with smaller benchmark for faster iteration
# First, edit examples/docker_task.json to reduce max_requests_per_iteration
python src/run_autotuner.py examples/docker_task.json --mode docker --direct --verbose
```

## Implementation Details

### Code Location

The verbose functionality is implemented in:
- `src/controllers/direct_benchmark_controller.py` - Benchmark execution with streaming
- `src/run_autotuner.py` - CLI argument parsing and orchestration

### How It Works

**Default Mode:**
```python
result = subprocess.run(cmd, capture_output=True, ...)
print(result.stdout)  # After completion
```

**Verbose Mode:**
```python
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, ...)
for line in process.stdout:
    print(f"[genai-bench] {line}")  # Real-time streaming
```

## Troubleshooting

### No Output in Verbose Mode

If you see no output in verbose mode, genai-bench might be buffering. This is automatically handled by setting `bufsize=1`.

### Mixed Output Order

When using verbose mode, stdout and stderr are merged. This is intentional for simpler real-time viewing.

### Log Files Not Created

The autotuner doesn't create separate log files by default. Use shell redirection:
```bash
python src/run_autotuner.py ... 2>&1 | tee output.log
```

## Best Practices

1. **Use verbose mode for:**
   - Initial testing of new configurations
   - Debugging connection/API issues
   - Long-running benchmarks (>5 minutes)
   - When you need to monitor progress

2. **Use default mode for:**
   - Production runs
   - CI/CD pipelines
   - When output needs post-processing
   - Multiple parallel experiments

3. **Save logs for analysis:**
   ```bash
   python src/run_autotuner.py ... --verbose 2>&1 | tee "run_$(date +%Y%m%d_%H%M%S).log"
   ```

## See Also

- [Docker Mode Documentation](DOCKER_MODE.md)
- [Troubleshooting Guide](../README.md#troubleshooting)
- [GenAI-Bench Documentation](https://github.com/sgl-project/genai-bench)
