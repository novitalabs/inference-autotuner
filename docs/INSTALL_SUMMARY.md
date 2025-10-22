# Installation Summary

This document summarizes the environment installation for the LLM Inference Autotuner project.

## Installation Script

A comprehensive installation script `install.sh` has been created to automate the setup process.

### Features

- **Prerequisite Checking**: Verifies Python, pip, kubectl, and git installations
- **Git Submodules**: Automatically initializes OME and genai-bench submodules
- **Virtual Environment**: Creates and configures Python virtual environment
- **Dependency Installation**: Installs all required Python packages
- **genai-bench Setup**: Installs genai-bench from local submodule in editable mode
- **Kubernetes Resources**: Creates namespace and PersistentVolumeClaim for benchmarks
- **Verification**: Tests all installations and provides status summary
- **Flexible Options**: Supports custom configurations and skip flags

### Usage

```bash
# Full installation (recommended)
./install.sh

# Skip virtual environment creation
./install.sh --skip-venv

# Skip Kubernetes resource creation
./install.sh --skip-k8s

# Custom virtual environment path
./install.sh --venv-path /path/to/venv

# Show help
./install.sh --help
```

## Installation Results

### ✅ Successfully Installed Components

1. **Python Environment**
   - Python version: 3.10.12
   - Virtual environment: `./env/`
   - pip upgraded to latest version

2. **Python Dependencies**
   - `kubernetes` v34.1.0 - Kubernetes Python client
   - `pyyaml` v6.0.3 - YAML parsing
   - `jinja2` v3.1.6 - Template rendering
   - All transitive dependencies installed

3. **genai-bench**
   - Version: 0.0.2
   - Installation mode: Editable (linked to `third_party/genai-bench`)
   - CLI available at: `env/bin/genai-bench`
   - Commands verified:
     - `genai-bench --version` ✅
     - `genai-bench --help` ✅
     - `genai-bench benchmark --help` ✅

4. **Git Submodules**
   - OME: `third_party/ome` (v0.1.3-69-g78587ad)
   - genai-bench: `third_party/genai-bench` (v0.0.2-15-ga11575b)

5. **Directory Structure**
   - `results/` - Created for storing benchmark results
   - `benchmark_results/` - Created for intermediate files

6. **Kubernetes Resources**
   - Namespace: `autotuner` ✅ Created
   - PersistentVolumeClaim: `benchmark-results-pvc` ✅ Created (1Gi)

### ⚠️ Components Requiring Manual Setup

1. **OME Installation**
   - Status: Not found in current cluster
   - Impact: Cannot use Kubernetes BenchmarkJob mode
   - Workaround: Use direct CLI mode (`--direct` flag)
   - Installation: Follow OME documentation to install operator

2. **ClusterBaseModels & ClusterServingRuntimes**
   - Status: No CRDs found
   - Impact: Cannot deploy InferenceServices
   - Required for: Both benchmark modes
   - Setup: Install OME operator first

## Verification Commands

### Python Environment

```bash
# Activate virtual environment
source env/bin/activate

# Check installed packages
pip list | grep -E "(kubernetes|pyyaml|jinja2|genai-bench)"

# Test genai-bench
genai-bench --version
genai-bench benchmark --help
```

### Kubernetes Resources

```bash
# Check cluster connectivity
kubectl cluster-info

# Check autotuner namespace
kubectl get namespace autotuner

# Check PVC
kubectl get pvc -n autotuner

# Check OME (if installed)
kubectl get namespace ome
kubectl get crd | grep ome.io
kubectl get clusterbasemodels
kubectl get clusterservingruntimes
```

## Current System Status

### Working Components

✅ Python dependencies installed
✅ genai-bench CLI functional
✅ Kubernetes cluster accessible
✅ autotuner namespace created
✅ PVC for benchmark results created
✅ Virtual environment configured
✅ Git submodules initialized

### Components Needing Setup

❌ OME operator not installed
❌ No ClusterBaseModels available
❌ No ClusterServingRuntimes available

## Next Steps

### Option 1: Direct CLI Mode (Recommended for Current Setup)

Since OME is not installed, use the direct CLI mode which bypasses Kubernetes BenchmarkJob:

```bash
# Activate environment
source env/bin/activate

# Configure task
vi examples/simple_task.json

# Run autotuner in direct mode
python src/run_autotuner.py examples/simple_task.json --direct
```

**Requirements:**
- ✅ genai-bench installed (done)
- ✅ kubectl configured (done)
- ⚠️ OME InferenceService deployed (requires OME installation)

### Option 2: Full Kubernetes Mode

To use the full Kubernetes BenchmarkJob mode:

1. **Install OME Operator**
   ```bash
   # Follow OME installation guide
   # https://github.com/sgl-project/ome
   ```

2. **Create Model and Runtime Resources**
   ```bash
   kubectl apply -f <clusterbasemodel.yaml>
   kubectl apply -f <clusterservingruntime.yaml>
   ```

3. **Verify Installation**
   ```bash
   kubectl get clusterbasemodels
   kubectl get clusterservingruntimes
   ```

4. **Run Autotuner**
   ```bash
   python src/run_autotuner.py examples/simple_task.json
   ```

## Environment Variables

The installation script uses these paths:

- `VENV_PATH`: `./env` (default, customizable with `--venv-path`)
- `SCRIPT_DIR`: Auto-detected from script location
- Python: System Python 3.10.12

## Troubleshooting

### Issue: genai-bench not found after installation

**Solution:**
```bash
source env/bin/activate
which genai-bench  # Should show: ./env/bin/genai-bench
```

### Issue: kubectl cannot connect to cluster

**Solution:**
```bash
# Check kubeconfig
kubectl cluster-info

# Set kubeconfig if needed
export KUBECONFIG=/path/to/kubeconfig
```

### Issue: OME CRDs not found

**Solution:**
This is expected if OME is not installed. Either:
1. Install OME operator (see OME documentation)
2. Use direct CLI mode (`--direct` flag)

## Summary

The installation script successfully set up all **local components** required for the autotuner:
- ✅ Python environment with all dependencies
- ✅ genai-bench CLI tool
- ✅ Kubernetes namespace and storage resources
- ✅ Project structure and directories

**OME installation** is the only remaining prerequisite for full functionality. Until then, the project can be used in **direct CLI mode** for benchmarking deployed InferenceServices.

## Files Created

- `/root/work/inference-autotuner/install.sh` - Main installation script (executable)
- `/root/work/inference-autotuner/env/` - Python virtual environment
- Kubernetes resources:
  - Namespace: `autotuner`
  - PVC: `benchmark-results-pvc`

## Installation Log

The installation completed successfully with the following summary:

```
[SUCCESS] python3 is installed
[SUCCESS] pip3 is installed
[SUCCESS] kubectl is installed
[SUCCESS] git is installed
[SUCCESS] Git submodules initialized
[SUCCESS] Virtual environment activated
[SUCCESS] Python dependencies installed
[SUCCESS] genai-bench installed
[SUCCESS] genai-bench CLI available at: /root/work/inference-autotuner/env/bin/genai-bench
[SUCCESS] Directories created
[SUCCESS] Kubernetes cluster is accessible
[SUCCESS] PVC created/updated
[WARNING] OME namespace not found - OME may not be installed
```

Total installation time: ~2-3 minutes (depending on network speed for package downloads)
