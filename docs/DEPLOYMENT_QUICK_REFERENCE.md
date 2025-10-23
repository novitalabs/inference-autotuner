# Inference-Autotuner: Quick Reference - Deployment Architecture

## One-Page Summary

### What This Project Does
- Automatically tunes LLM inference engine parameters by running multiple experiments
- Deploys different parameter configurations via Kubernetes (OME framework)
- Benchmarks each configuration with genai-bench
- Reports optimal parameters based on latency or throughput

### Deployment Architecture (Simple View)

```
Task JSON → Orchestrator → Deploy InferenceService → Run Benchmark → Results JSON
                              (OME CRD)              (genai-bench)
```

### Two Benchmark Modes

| Mode | Command | Where genai-bench Runs | Pros | Cons |
|------|---------|----------------------|------|------|
| **Direct CLI** (RECOMMENDED) | `--direct` | Local Python process | Simple, no Docker issues | Requires kubectl port-forward |
| **K8s BenchmarkJob** | (default) | In Kubernetes pod | Native K8s experience | Requires Docker image, PVC |

### Entry Points

1. **Orchestrator:** `python src/run_autotuner.py examples/simple_task.json [--direct]`
2. **Installation:** `./install.sh [--install-ome]`

### Deployment Infrastructure

- **Deployment Engine:** Kubernetes (v1.28+)
- **Framework:** OME (Open Model Engine) - provides CRDs
- **Model Runtime:** SGLang (v0.5.2)
- **Benchmarking:** genai-bench
- **Configuration:** Jinja2 templates (YAML)
- **Dependency:** OME must be pre-installed

### Key Components & Files

| Component | File | Purpose |
|-----------|------|---------|
| Orchestrator | `src/run_autotuner.py` | Main entry point, experiment loop |
| OME Controller | `src/controllers/ome_controller.py` | InferenceService CRUD |
| Benchmark (K8s) | `src/controllers/benchmark_controller.py` | BenchmarkJob management |
| Benchmark (CLI) | `src/controllers/direct_benchmark_controller.py` | Port-forward + genai-bench CLI |
| InferenceService Template | `src/templates/inference_service.yaml.j2` | SGLang deployment config |
| BenchmarkJob Template | `src/templates/benchmark_job.yaml.j2` | Benchmark pod config |
| Task Config | `examples/simple_task.json` | User-provided parameters |
| Installation | `install.sh` | Setup + OME installation |

### Experiment Lifecycle (Per Configuration)

```
1. Render InferenceService YAML with parameters (tp_size, mem_frac, etc.)
2. POST to Kubernetes → Create InferenceService CRD
3. Poll status.conditions until Ready=True (timeout: 600s)
4. Run benchmark:
   - Direct Mode: kubectl port-forward 8080:8000 → genai-bench CLI → localhost:8080
   - K8s Mode: Create BenchmarkJob CRD → Wait for completion → Read PVC results
5. Extract metrics & calculate score
6. DELETE InferenceService & BenchmarkJob
```

### Hardcoded Key Values

| Setting | Value |
|---------|-------|
| OME CRD Group | `ome.io` |
| Autotuner Namespace | `autotuner` |
| OME Namespace | `ome` |
| Model Path | `/mnt/data/models/{model_name}` |
| Ports | 8080 (InferenceService), 8000 (SGLang) |
| Poll Intervals | 10s (ISVC), 15s (Benchmark) |
| Default Timeouts | 600s (ISVC), 1800s (Benchmark) |
| PVC Name | `benchmark-results-pvc` |
| SGLang Image | `docker.io/lmsysorg/sglang:v0.5.2-cu126` |
| Genai-bench Image | `kllambda/genai-bench:v251014` |

### Configuration Flow

1. **User defines:** `examples/simple_task.json`
   - Model name & namespace
   - Parameters to tune (tp_size, mem_frac, etc.)
   - Optimization objective (minimize_latency / maximize_throughput)
   - Benchmark settings (traffic, concurrency, etc.)

2. **System generates:** All combinations (Cartesian product)
   - Example: tp_size∈[1,2] × mem_frac∈[0.8,0.9] = 4 combinations

3. **System executes:** Sequential experiments
   - Each experiment: Deploy → Wait → Benchmark → Score → Cleanup
   - Results saved to `results/{task_name}_results.json`

### Results Structure

```json
{
  "task_name": "simple-tune",
  "total_experiments": 4,
  "successful_experiments": 4,
  "best_result": {
    "experiment_id": 2,
    "parameters": {"tp_size": 1, "mem_frac": 0.9},
    "objective_score": 89.2
  },
  "all_results": [...]
}
```

### Dependencies (OME Stack)

- **Required:** Kubernetes cluster with OME installed
- **OME CRDs:** InferenceService, BenchmarkJob, ClusterBaseModel, ClusterServingRuntime
- **OME Dependencies:** cert-manager, KEDA
- **Python:** kubernetes-client, PyYAML, Jinja2
- **Tools:** kubectl, genai-bench CLI

### API Usage

**Kubernetes API (via Python client):**
```python
# Create CRD
self.custom_api.create_namespaced_custom_object(
    group="ome.io",
    version="v1beta1",
    namespace=namespace,
    plural="inferenceservices",  # or "benchmarkjobs"
    body=manifest_dict
)

# Poll status
isvc = self.custom_api.get_namespaced_custom_object(...)
status = isvc.get("status", {})
conditions = status.get("conditions", [])  # Check Ready condition
```

**kubectl CLI (for port-forward):**
```bash
kubectl port-forward pod/{pod_name} 8080:8000 -n {namespace} &
# or
kubectl port-forward svc/{service_name} 8080:8000 -n {namespace} &
```

**genai-bench CLI:**
```bash
genai-bench benchmark \
  --api-backend openai \
  --api-base http://localhost:8080 \
  --task text-to-text \
  --traffic-scenario "D(100,100)" \
  --num-concurrency 1
```

### Zero Deployment Automation Features

- No automatic scheduling (manual CLI invocation only)
- No CI/CD integration
- No webhooks or event-driven execution
- No containerized orchestrator
- No operator pattern
- Deployment logic: Pure Python orchestrator script

### Extension Points for Production

1. **Deployment:**
   - Containerize orchestrator
   - Deploy as K8s CronJob or Operator
   - Add REST API for job submission

2. **Optimization:**
   - Bayesian optimization (instead of grid search)
   - Parallel experiment execution
   - Early stopping strategies

3. **Storage:**
   - PostgreSQL for experiment history
   - InfluxDB for metrics
   - S3 for result artifacts

4. **Monitoring:**
   - Prometheus metrics export
   - Real-time status API (WebSocket)
   - Visualization dashboard

### File Statistics

- **Total Lines:** ~670 core code + ~460 install script
- **Python Modules:** 5 (orchestrator, 2 controllers, 1 utility, 1 init)
- **Templates:** 2 Jinja2 YAML templates
- **Configurations:** 3 example/config files
- **Documentation:** ~1000 lines

---

## Quick Start (Under 5 minutes)

```bash
# 1. Verify OME is installed
kubectl get namespace ome

# 2. Activate environment & run
source env/bin/activate
python src/run_autotuner.py examples/simple_task.json --direct

# 3. View results
cat results/simple-tune_results.json
```

---

## See Also

- Full architecture: `/root/work/inference-autotuner/DEPLOYMENT_ARCHITECTURE.md`
- README: `/root/work/inference-autotuner/README.md`
- Installation: `/root/work/inference-autotuner/docs/OME_INSTALLATION.md`

