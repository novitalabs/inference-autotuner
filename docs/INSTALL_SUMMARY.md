# Installation Summary - LLM Inference Autotuner

This document summarizes the installation process, components installed, and common troubleshooting steps for the inference-autotuner project.

## Quick Start

```bash
# Clone and install with OME
git clone <repository-url>
cd inference-autotuner
./install.sh --install-ome

# Apply model resources
kubectl apply -f third_party/ome/config/models/meta/Llama-3.2-1B-Instruct.yaml

# Run test
python src/run_autotuner.py examples/simple_task.json --direct
```

## Components Installed

### 1. Python Environment

**Location:** `./env/` (virtual environment)

**Packages Installed:**
- `kubernetes>=28.1.0` - Kubernetes Python client
- `pyyaml>=6.0` - YAML parsing
- `jinja2>=3.0` - Template rendering
- `genai-bench` - Benchmark CLI (editable install from `third_party/genai-bench`)

**Verification:**
```bash
source env/bin/activate
python --version
pip list | grep -E "(kubernetes|pyyaml|jinja2|genai-bench)"
genai-bench --version
```

**Expected Output:**
```
Python 3.8+
genai-bench    0.0.2    /path/to/third_party/genai-bench
jinja2         3.1.6
kubernetes     34.1.0
PyYAML         6.0.3
```

### 2. Git Submodules

**OME (Open Model Engine):**
- Path: `third_party/ome/`
- Version: v0.1.3+
- Contents: Kubernetes operator, CRDs, Helm charts, example configs

**genai-bench:**
- Path: `third_party/genai-bench/`
- Version: v0.0.2+
- Contents: Benchmark CLI tool, task definitions, metrics collectors

**Verification:**
```bash
git submodule status
ls third_party/ome/
ls third_party/genai-bench/
```

### 3. Kubernetes Prerequisites

#### cert-manager (OME Dependency)

**Installation:**
```bash
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --create-namespace \
  --set crds.enabled=true
```

**Verification:**
```bash
kubectl get pods -n cert-manager
```

**Expected:** 3 pods running (controller, webhook, cainjector)

**Common Issue:** Webhook timeout during installation
```bash
# Workaround: Delete webhook configurations
kubectl delete validatingwebhookconfiguration cert-manager-webhook
kubectl delete mutatingwebhookconfiguration cert-manager-webhook
```

#### KEDA (Kubernetes Event-Driven Autoscaling)

**Why Required:** OME uses KEDA's ScaledObject CRD for InferenceService autoscaling

**Installation:**
```bash
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda \
  --namespace keda \
  --create-namespace
```

**Verification:**
```bash
kubectl get pods -n keda
kubectl get crd | grep keda
```

**Expected:**
- `keda-operator` and `keda-operator-metrics-apiserver` pods running
- `scaledobjects.keda.sh` CRD present

**Common Issue:** OME controller crashes with "no kind is registered for type v1alpha1.ScaledObject"
```bash
# Fix: Install KEDA and restart OME controller
kubectl rollout restart deployment/ome-controller-manager -n ome
```

### 4. OME Operator

**Installation:**
```bash
# Install CRDs from OCI registry
helm upgrade --install ome-crd \
  oci://ghcr.io/moirai-internal/charts/ome-crd \
  --version 0.1.3

# Install OME resources from local charts
cd third_party/ome
helm upgrade --install ome charts/ome-resources \
  --namespace ome \
  --create-namespace \
  --timeout 7m

cd ../..
```

**Components Installed:**
- **Namespace:** `ome`
- **Controller Manager:** 3 replicas (HA configuration)
- **Model Agent:** 1 DaemonSet pod per node
- **CRDs:** 7 custom resource definitions

**CRDs Installed:**
1. `inferenceservices.ome.io` - Model deployment specifications
2. `benchmarkjobs.ome.io` - Benchmark execution resources
3. `clusterbasemodels.ome.io` - Model metadata (cluster-scoped)
4. `basemodels.ome.io` - Model metadata (namespace-scoped)
5. `clusterservingruntimes.ome.io` - Runtime configurations (cluster-scoped)
6. `servingruntimes.ome.io` - Runtime configurations (namespace-scoped)
7. `finetunedweights.ome.io` - Fine-tuned model weights

**Verification:**
```bash
# Check pods
kubectl get pods -n ome

# Check CRDs
kubectl get crd | grep ome.io

# Check controllers
kubectl logs -n ome deployment/ome-controller-manager --tail=50
```

**Expected Output:**
```
NAME                                      READY   STATUS    AGE
ome-controller-manager-xxxxx-xxxxx        1/1     Running   5m
ome-controller-manager-xxxxx-xxxxx        1/1     Running   5m
ome-controller-manager-xxxxx-xxxxx        1/1     Running   5m
ome-model-agent-xxxxx                     1/1     Running   5m
```

### 5. Kubernetes Resources Created

**Namespace:** `autotuner`
```bash
kubectl get namespace autotuner
```

**PersistentVolumeClaim:** `benchmark-results-pvc` (1Gi)
```bash
kubectl get pvc -n autotuner
```

**Purpose:** Stores benchmark results from BenchmarkJob executions

### 6. Directory Structure

```
inference-autotuner/
├── env/                          # Python virtual environment
├── third_party/
│   ├── ome/                      # OME submodule
│   └── genai-bench/              # genai-bench submodule
├── src/
│   ├── controllers/              # OME and benchmark controllers
│   ├── templates/                # Jinja2 YAML templates
│   ├── utils/                    # Optimizer utilities
│   └── run_autotuner.py          # Main orchestrator
├── examples/
│   └── simple_task.json          # Example tuning task
├── config/
│   ├── benchmark-pvc.yaml        # PVC for benchmark results
│   └── examples/                 # Example ClusterBaseModel/Runtime configs
├── results/                      # JSON result files (created at runtime)
└── benchmark_results/            # Benchmark output data (created at runtime)
```

## Model and Runtime Setup

After installation, you need at least one model and runtime configured.

### Option A: Use Example Resources

```bash
# Apply Llama 3.2 1B model
kubectl apply -f third_party/ome/config/models/meta/Llama-3.2-1B-Instruct.yaml

# Apply SGLang runtime
kubectl apply -f config/examples/clusterservingruntime-sglang.yaml
```

### Option B: Download Model Manually

**For gated models (e.g., Llama):**

```bash
# 1. Accept license on HuggingFace
# Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct

# 2. Create HuggingFace token
# Visit: https://huggingface.co/settings/tokens

# 3. Login and download
huggingface-cli login --token <your-token>
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /tmp/llama-3.2-1b-instruct

# 4. For Minikube: Transfer model files
tar czf /tmp/model.tar.gz -C /tmp llama-3.2-1b-instruct
scp -i $(minikube ssh-key) /tmp/model.tar.gz docker@$(minikube ip):~/
minikube ssh "sudo mkdir -p /mnt/data/models && \
  sudo tar xzf ~/model.tar.gz -C /mnt/data/models/"

# 5. Create ClusterBaseModel pointing to the path
kubectl apply -f config/examples/clusterbasemodel-llama-3.2-1b.yaml
```

### Verification

```bash
# Check models
kubectl get clusterbasemodels

# Check runtimes
kubectl get clusterservingruntimes

# Expected output
NAME                      READY   AGE
llama-3-2-1b-instruct    True    5m

NAME                          ACTIVE   AGE
llama-3-2-1b-instruct-rt     True     5m
```

## Testing Installation

### Test 1: Verify Components

```bash
# Activate environment
source env/bin/activate

# Check Python packages
python -c "import kubernetes; import yaml; import jinja2; print('✓ All imports successful')"

# Check genai-bench CLI
genai-bench --version

# Check kubectl access
kubectl cluster-info

# Check OME
kubectl get pods -n ome
kubectl get crd | grep ome.io
```

### Test 2: Run Simple Task

```bash
# Update examples/simple_task.json with your model/runtime names
vi examples/simple_task.json

# Run autotuner in direct CLI mode (recommended)
python src/run_autotuner.py examples/simple_task.json --direct

# Or run with Kubernetes BenchmarkJob mode
python src/run_autotuner.py examples/simple_task.json
```

**Expected Behavior:**
1. InferenceService created and becomes Ready (60-90 seconds)
2. Benchmark executes (time varies by configuration)
3. Results saved to `results/simple-tune_results.json`
4. Resources automatically cleaned up

## Common Issues and Solutions

### Issue 1: OME Not Installed

**Error:**
```
ERROR: OME (Open Model Engine) is a required prerequisite
```

**Solution:**
```bash
./install.sh --install-ome
```

### Issue 2: cert-manager Webhook Timeout

**Error:**
```
Error: INSTALLATION FAILED: context deadline exceeded
```

**Solution:**
```bash
kubectl delete validatingwebhookconfiguration cert-manager-webhook
kubectl delete mutatingwebhookconfiguration cert-manager-webhook
helm install cert-manager jetstack/cert-manager --set crds.enabled=true
```

### Issue 3: KEDA Not Installed

**Error:**
```
no kind is registered for the type v1alpha1.ScaledObject
```

**Solution:**
```bash
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda --namespace keda --create-namespace
kubectl rollout restart deployment/ome-controller-manager -n ome
```

### Issue 4: No GPU Resources in Minikube

**Error:**
```
0/1 nodes are available: 1 Insufficient nvidia.com/gpu
```

**Root Cause:** Minikube with Docker driver cannot access host GPUs (nested containerization limitation)

**Solutions:**

**A. Use Minikube with --driver=none (bare metal):**
```bash
minikube start --driver=none
```

**B. Use direct Docker deployment for testing:**
```bash
# Download model
huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir /tmp/llama-3.2-1b-instruct

# Check available GPUs
nvidia-smi

# Run SGLang directly
docker run --gpus '"device=1"' -d --name sglang-llama \
  -p 8000:8080 \
  -v /tmp/llama-3.2-1b-instruct:/model \
  lmsysorg/sglang:v0.5.2-cu126 \
  python3 -m sglang.launch_server \
  --model-path /model \
  --host 0.0.0.0 \
  --port 8080 \
  --mem-frac 0.6

# Test
curl http://localhost:8000/health
```

**C. Use production Kubernetes cluster with GPU support**

### Issue 5: Model Download Access Denied

**Error:**
```
401 Client Error: Unauthorized
Access to model meta-llama/Llama-3.2-1B-Instruct is restricted
```

**Solution:**
```bash
# Accept license on HuggingFace website first
# Then create token and login
huggingface-cli login --token <your-token>

# For OME model agent
kubectl create secret generic hf-token \
  --from-literal=token=<your-token> \
  -n ome
```

### Issue 6: Network/Proxy Issues

**Symptoms:**
- ImagePullBackOff errors
- OME model agent can't download models
- Connection timeouts

**Solution:** Configure proxy in Minikube

```bash
minikube ssh

sudo mkdir -p /etc/systemd/system/docker.service.d
sudo tee /etc/systemd/system/docker.service.d/http-proxy.conf <<EOF
[Service]
Environment="HTTP_PROXY=http://YOUR_PROXY:PORT"
Environment="HTTPS_PROXY=http://YOUR_PROXY:PORT"
Environment="NO_PROXY=localhost,127.0.0.1,10.96.0.0/12"
EOF

sudo systemctl daemon-reload
sudo systemctl restart docker
exit

minikube stop && minikube start
```

Patch OME model agent:
```bash
kubectl set env daemonset/ome-model-agent -n ome \
  HTTP_PROXY=http://YOUR_PROXY:PORT \
  HTTPS_PROXY=http://YOUR_PROXY:PORT
```

### Issue 7: GPU Out of Memory

**Error:**
```
torch.OutOfMemoryError: CUDA out of memory
```

**Solution:**

A. Check GPU status:
```bash
nvidia-smi
```

B. Select different GPU:
```bash
docker run --gpus '"device=1"' ...  # Use GPU 1 instead of 0
```

C. Reduce memory allocation:
```bash
--mem-frac 0.6  # Instead of 0.8
```

**Memory Guidelines:**
- 1-3B models: `--mem-frac 0.6-0.7`
- 7-13B models: `--mem-frac 0.8-0.85`
- 70B+ models: `--mem-frac 0.9-0.95`

## Monitoring and Debugging

### Watch Resources

```bash
# All autotuner resources
watch kubectl get inferenceservices,benchmarkjobs,pods -n autotuner

# Pod logs
kubectl logs -f <pod-name> -n autotuner

# OME controller logs
kubectl logs -n ome deployment/ome-controller-manager --tail=100
```

### Check InferenceService Status

```bash
# Get InferenceService details
kubectl describe inferenceservice <name> -n autotuner

# Check pod events
kubectl get events -n autotuner --sort-by='.lastTimestamp'

# Common status conditions
# - Ready=True: Service is fully operational
# - Ready=False: Check "reason" field for details
# - ComponentNotReady: Target service not ready
```

### Check BenchmarkJob Status

```bash
# Get BenchmarkJob status
kubectl describe benchmarkjob <name> -n autotuner

# Check benchmark pod logs
kubectl get pods -n autotuner | grep bench
kubectl logs <benchmark-pod> -n autotuner

# Status fields
# - state: "Running", "Complete", or "Failed"
# - failureMessage: Error details if failed
```

## Uninstallation

### Remove Autotuner Resources

```bash
# Delete namespace (removes all InferenceServices, BenchmarkJobs, pods)
kubectl delete namespace autotuner

# Remove Python environment
rm -rf env/

# Remove result files
rm -rf results/ benchmark_results/
```

### Remove OME (Optional)

```bash
# Delete OME resources
helm uninstall ome -n ome

# Delete OME CRDs
helm uninstall ome-crd

# Delete namespace
kubectl delete namespace ome

# Delete KEDA
helm uninstall keda -n keda
kubectl delete namespace keda

# Delete cert-manager
helm uninstall cert-manager -n cert-manager
kubectl delete namespace cert-manager
```

## Next Steps

1. **Customize Task Configuration:** Edit `examples/simple_task.json` for your model and parameters
2. **Review Documentation:**
   - [README.md](../README.md) - Main usage documentation
   - [OME_INSTALLATION.md](OME_INSTALLATION.md) - Detailed OME setup
   - [SGLANG_RUNTIME_METRICS.md](SGLANG_RUNTIME_METRICS.md) - SGLang tuning guide
3. **Run Experiments:** Test different parameter combinations
4. **Analyze Results:** Review JSON output in `results/` directory

## Support and Troubleshooting

For detailed troubleshooting, see:
- [README.md - Troubleshooting Section](../README.md#troubleshooting)
- [prompts.md](../prompts.md) - Complete development history and solutions

For issues not covered in this documentation:
1. Check OME controller logs: `kubectl logs -n ome deployment/ome-controller-manager`
2. Review pod events: `kubectl get events -n autotuner`
3. Verify all prerequisites are installed correctly
4. Consult OME documentation: https://github.com/sgl-project/ome
