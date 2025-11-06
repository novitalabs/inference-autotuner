#!/bin/bash

##############################################################################
# LLM Inference Autotuner - Environment Installation Script
#
# This script installs all prerequisites for the inference-autotuner project.
# It sets up:
#   - Python dependencies
#   - genai-bench CLI
#   - Kubernetes resources (optional)
#   - Web API dependencies (FastAPI, SQLAlchemy, ARQ)
#   - Database directory (~/.local/share/inference-autotuner/)
#
# Usage:
#   ./install.sh [OPTIONS]
#
# Options:
#   --skip-venv         Skip virtual environment creation (use system Python)
#   --skip-k8s          Skip Kubernetes resource creation
#   --install-ome       Install OME operator automatically
#   --venv-path PATH    Specify custom virtual environment path (default: ./env)
#   --help              Show this help message
#
##############################################################################

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/env"
SKIP_VENV=false
SKIP_K8S=false
INSTALL_OME=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-venv)
            SKIP_VENV=true
            shift
            ;;
        --skip-k8s)
            SKIP_K8S=true
            shift
            ;;
        --install-ome)
            INSTALL_OME=true
            shift
            ;;
        --venv-path)
            VENV_PATH="$2"
            shift 2
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help to see available options"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        log_success "$1 is installed"
        return 0
    else
        log_error "$1 is not installed"
        return 1
    fi
}

##############################################################################
# 1. Check Prerequisites
##############################################################################

log_info "Checking prerequisites..."

# Check Python
if ! check_command python3; then
    log_error "Python 3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
log_info "Python version: $PYTHON_VERSION"

# Check pip
if ! check_command pip3; then
    log_error "pip3 is required but not installed"
    exit 1
fi

# Check kubectl (optional for K8s mode)
if [ "$SKIP_K8S" = false ]; then
    if ! check_command kubectl; then
        log_warning "kubectl is not installed - Kubernetes features will not work"
        log_warning "Install kubectl to use the autotuner with Kubernetes"
    else
        KUBECTL_VERSION=$(kubectl version --client --short 2>/dev/null | head -1)
        log_info "kubectl version: $KUBECTL_VERSION"
    fi
fi

# Check git
if ! check_command git; then
    log_error "git is required but not installed"
    exit 1
fi

##############################################################################
# 2. Initialize Git Submodules
##############################################################################

log_info "Initializing git submodules (OME and genai-bench)..."

cd "$SCRIPT_DIR"

if [ ! -d ".git" ]; then
    log_warning "Not a git repository - skipping submodule initialization"
else
    git submodule update --init --recursive
    log_success "Git submodules initialized"
fi

##############################################################################
# 3. Setup Python Virtual Environment
##############################################################################

if [ "$SKIP_VENV" = false ]; then
    log_info "Setting up Python virtual environment at: $VENV_PATH"

    # Check Python version compatibility with genai-bench
    PYTHON_MAJOR=$(python3 --version | awk '{print $2}' | cut -d. -f1)
    PYTHON_MINOR=$(python3 --version | awk '{print $2}' | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 13 ]; then
        log_warning "Python 3.13+ detected - genai-bench requires Python <3.13"
        
        # Try to find Python 3.10, 3.11, or 3.12
        for py_version in python3.12 python3.11 python3.10; do
            if command -v $py_version &> /dev/null; then
                log_info "Found compatible Python: $py_version"
                PYTHON_CMD=$py_version
                break
            fi
        done
        
        if [ -z "$PYTHON_CMD" ]; then
            log_error "No compatible Python version found (need 3.10, 3.11, or 3.12)"
            log_error "genai-bench requires Python <3.13 for compatibility"
            log_info "Please install Python 3.10, 3.11, or 3.12 and run:"
            log_info "  python3.10 -m venv env"
            log_info "  source env/bin/activate"
            log_info "  pip install -r requirements.txt"
            exit 1
        fi
    else
        PYTHON_CMD=python3
    fi

    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_PATH" ]; then
        log_info "Creating virtual environment with $PYTHON_CMD..."
        $PYTHON_CMD -m venv "$VENV_PATH"
        log_success "Virtual environment created"
    else
        log_warning "Virtual environment already exists at $VENV_PATH"
        
        # Check if it uses compatible Python
        VENV_PYTHON_VERSION=$("$VENV_PATH/bin/python" --version | awk '{print $2}')
        log_info "Existing venv Python version: $VENV_PYTHON_VERSION"
        
        VENV_MAJOR=$(echo $VENV_PYTHON_VERSION | cut -d. -f1)
        VENV_MINOR=$(echo $VENV_PYTHON_VERSION | cut -d. -f2)
        
        if [ "$VENV_MAJOR" -eq 3 ] && [ "$VENV_MINOR" -ge 13 ]; then
            log_warning "Existing venv uses Python 3.13+ which is incompatible with genai-bench"
            log_warning "Recreating virtual environment with compatible Python..."
            rm -rf "$VENV_PATH"
            $PYTHON_CMD -m venv "$VENV_PATH"
            log_success "Virtual environment recreated with $PYTHON_CMD"
        fi
    fi

    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    log_success "Virtual environment activated"

    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip

else
    log_warning "Skipping virtual environment creation (--skip-venv)"
fi

##############################################################################
# 4. Install Python Dependencies
##############################################################################

log_info "Installing Python dependencies from requirements.txt..."

if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    pip install -r "$SCRIPT_DIR/requirements.txt"
    log_success "Python dependencies installed"
else
    log_error "requirements.txt not found"
    exit 1
fi

##############################################################################
# 5. Install genai-bench
##############################################################################

log_info "Installing genai-bench from third_party/genai-bench..."

if [ -d "$SCRIPT_DIR/third_party/genai-bench" ]; then
    # Install in editable mode for development
    pip install -e "$SCRIPT_DIR/third_party/genai-bench"
    log_success "genai-bench installed"
else
    log_error "genai-bench submodule not found at third_party/genai-bench"
    log_error "Run: git submodule update --init --recursive"
    exit 1
fi

##############################################################################
# 6. Verify Installation
##############################################################################

log_info "Verifying installation..."

# Check Python packages
REQUIRED_PACKAGES=("kubernetes" "yaml" "jinja2" "docker")
for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        log_success "Python package '$package' is available"
    else
        log_error "Python package '$package' is not available"
    fi
done

# Check genai-bench CLI
if [ "$SKIP_VENV" = false ]; then
    GENAI_BENCH_PATH="$VENV_PATH/bin/genai-bench"
else
    GENAI_BENCH_PATH="$(which genai-bench)"
fi

if [ -f "$GENAI_BENCH_PATH" ]; then
    log_success "genai-bench CLI available at: $GENAI_BENCH_PATH"
    log_info "Testing genai-bench CLI..."
    if "$GENAI_BENCH_PATH" --version &> /dev/null; then
        GENAI_BENCH_VERSION=$("$GENAI_BENCH_PATH" --version 2>&1 | grep "version" | awk '{print $NF}')
        log_success "genai-bench version: $GENAI_BENCH_VERSION"
    else
        log_warning "genai-bench --version failed (may still work)"
    fi
else
    log_error "genai-bench CLI not found"
    log_error "This is required for benchmarking. Please check installation."
fi

# Check GPU availability (for Docker mode)
log_info "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        log_success "Found $GPU_COUNT GPU(s):"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null | while read line; do
            log_info "  GPU $line"
        done
        log_info "Docker mode can use GPUs directly"
    else
        log_warning "nvidia-smi found but no GPUs detected"
    fi
else
    log_warning "nvidia-smi not found - GPU support not available"
    log_warning "For GPU inference, install NVIDIA drivers and CUDA toolkit"
fi

# Check Docker (for Docker mode)
log_info "Checking Docker availability..."
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null; then
        log_success "Docker is installed and accessible"
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
        log_info "Docker version: $DOCKER_VERSION"
        
        # Check if Docker can access GPUs
        if command -v nvidia-smi &> /dev/null; then
            log_info "Testing Docker GPU access..."
            if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
                log_success "Docker can access GPUs (NVIDIA Container Toolkit is configured)"
                log_info "Docker mode with GPU is fully supported"
            else
                log_warning "Docker cannot access GPUs"
                log_warning "To enable GPU in Docker:"
                log_warning "  1. Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
                log_warning "  2. Restart Docker: sudo systemctl restart docker"
            fi
        fi
    else
        log_warning "Docker is installed but cannot connect to Docker daemon"
        log_warning "Start Docker or check permissions: sudo usermod -aG docker $USER"
    fi
else
    log_warning "Docker is not installed (optional for Docker mode)"
    log_info "To use Docker deployment mode:"
    log_info "  - Install Docker: https://docs.docker.com/engine/install/"
    log_info "  - Install NVIDIA Container Toolkit for GPU support"
fi

##############################################################################
# 7. Create Required Directories
##############################################################################

log_info "Creating required directories..."

mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/benchmark_results"

# Create database directory (XDG Base Directory standard)
DB_DIR="$HOME/.local/share/inference-autotuner"
mkdir -p "$DB_DIR"
log_success "Database directory created at: $DB_DIR"

log_success "Directories created"

##############################################################################
# 8. Setup Web API Dependencies (Optional)
##############################################################################

log_info "Checking Web API dependencies..."

# Check if Redis is needed (for background workers)
log_info "Web API requires Redis for background job processing"
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        log_success "Redis is running and accessible"
    else
        log_warning "Redis is installed but not running"
        log_info "Start Redis with: redis-server"
        log_info "Or using Docker: docker run -d -p 6379:6379 redis:alpine"
    fi
else
    log_warning "Redis is not installed (optional for Web API)"
    log_info "To enable background job processing:"
    log_info "  - Install Redis: apt-get install redis-server (Ubuntu/Debian)"
    log_info "  - Or use Docker: docker run -d -p 6379:6379 redis:alpine"
fi

# Verify web API dependencies
WEB_PACKAGES=("fastapi" "uvicorn" "sqlalchemy" "aiosqlite" "arq")
MISSING_WEB_PACKAGES=0
for package in "${WEB_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        log_success "Web API package '$package' is available"
    else
        log_warning "Web API package '$package' is not available"
        MISSING_WEB_PACKAGES=$((MISSING_WEB_PACKAGES + 1))
    fi
done

if [ "$MISSING_WEB_PACKAGES" -eq 0 ]; then
    log_success "All Web API dependencies are installed"
else
    log_warning "$MISSING_WEB_PACKAGES Web API package(s) missing"
    log_info "Reinstall with: pip install -r requirements.txt"
fi

##############################################################################
# 9. Setup Kubernetes Resources (Optional)
##############################################################################

if [ "$SKIP_K8S" = false ]; then
    log_info "Setting up Kubernetes resources..."

    # Check if kubectl is available and cluster is accessible
    if command -v kubectl &> /dev/null; then
        if kubectl cluster-info &> /dev/null; then
            log_success "Kubernetes cluster is accessible"

            # Create namespace
            log_info "Creating 'autotuner' namespace..."
            kubectl create namespace autotuner 2>/dev/null || log_warning "Namespace 'autotuner' already exists"

            # Create PVC for benchmark results
            if [ -f "$SCRIPT_DIR/config/benchmark-pvc.yaml" ]; then
                log_info "Creating PersistentVolumeClaim for benchmark results..."
                kubectl apply -f "$SCRIPT_DIR/config/benchmark-pvc.yaml" || log_warning "PVC creation failed"
                log_success "PVC created/updated"
            else
                log_warning "PVC config not found at config/benchmark-pvc.yaml"
            fi

            # Install OME if requested
            if [ "$INSTALL_OME" = true ]; then
                log_info "Installing OME (Open Model Engine)..."

                # Check if Helm is installed
                if ! command -v helm &> /dev/null; then
                    log_error "Helm is required to install OME but not found"
                    log_error "Install Helm: https://helm.sh/docs/intro/install/"
                    exit 1
                fi

                # Install cert-manager (required dependency)
                if ! kubectl get namespace cert-manager &> /dev/null; then
                    log_info "Installing cert-manager (OME dependency)..."
                    helm repo add jetstack https://charts.jetstack.io --force-update
                    helm repo update
                    helm install cert-manager jetstack/cert-manager \
                        --namespace cert-manager \
                        --create-namespace \
                        --set crds.enabled=true \
                        --wait --timeout=5m

                    # Delete webhook configurations to avoid issues
                    kubectl delete validatingwebhookconfiguration cert-manager-webhook 2>/dev/null || true
                    kubectl delete mutatingwebhookconfiguration cert-manager-webhook 2>/dev/null || true

                    log_success "cert-manager installed"
                else
                    log_info "cert-manager already installed"
                fi

                # Install OME CRDs
                if ! kubectl get crd inferenceservices.ome.io &> /dev/null; then
                    log_info "Installing OME CRDs..."
                    helm upgrade --install ome-crd \
                        oci://ghcr.io/moirai-internal/charts/ome-crd \
                        --namespace ome \
                        --create-namespace
                    log_success "OME CRDs installed"
                else
                    log_info "OME CRDs already installed"
                fi

                # Install KEDA (required for OME autoscaling)
                if ! kubectl get namespace keda &> /dev/null; then
                    log_info "Installing KEDA (OME dependency for autoscaling)..."
                    helm repo add kedacore https://kedacore.github.io/charts --force-update
                    helm repo update
                    helm install keda kedacore/keda \
                        --namespace keda \
                        --create-namespace \
                        --wait --timeout=5m
                    log_success "KEDA installed"
                else
                    log_info "KEDA already installed"
                fi

                # Install OME resources
                if ! kubectl get deployment -n ome ome-controller-manager &> /dev/null; then
                    log_info "Installing OME resources..."
                    cd "$SCRIPT_DIR/third_party/ome"
                    helm upgrade --install ome charts/ome-resources \
                        --namespace ome \
                        --wait --timeout=7m
                    cd "$SCRIPT_DIR"
                    log_success "OME resources installed"
                else
                    log_info "OME resources already installed"
                fi

                log_success "OME installation completed"
            fi

            # Verify OME installation (REQUIRED)
            log_info "Verifying OME installation (required prerequisite)..."
            if kubectl get namespace ome &> /dev/null; then
                log_success "OME namespace exists"

                # Check OME CRDs (all required)
                CRDS=("inferenceservices.ome.io" "benchmarkjobs.ome.io" "clusterbasemodels.ome.io" "clusterservingruntimes.ome.io")
                MISSING_CRDS=0
                for crd in "${CRDS[@]}"; do
                    if kubectl get crd "$crd" &> /dev/null; then
                        log_success "CRD '$crd' is available"
                    else
                        log_error "CRD '$crd' is not available"
                        MISSING_CRDS=$((MISSING_CRDS + 1))
                    fi
                done

                if [ "$MISSING_CRDS" -gt 0 ]; then
                    log_error "OME installation is incomplete - missing $MISSING_CRDS CRD(s)"
                    echo ""
                    echo "OME must be properly installed to use this autotuner."
                    echo "Please install OME following the instructions in docs/OME_INSTALLATION.md"
                    echo ""
                    exit 1
                fi

                # Check available models and runtimes
                log_info "Checking available models..."
                MODEL_COUNT=$(kubectl get clusterbasemodels --no-headers 2>/dev/null | wc -l)
                log_info "Found $MODEL_COUNT ClusterBaseModel(s)"

                log_info "Checking available runtimes..."
                RUNTIME_COUNT=$(kubectl get clusterservingruntimes --no-headers 2>/dev/null | wc -l)
                log_info "Found $RUNTIME_COUNT ClusterServingRuntime(s)"

                if [ "$MODEL_COUNT" -eq 0 ] || [ "$RUNTIME_COUNT" -eq 0 ]; then
                    log_warning "No models or runtimes found"
                    log_warning "You need to create at least one ClusterBaseModel and ClusterServingRuntime before using the autotuner"
                    log_warning "See docs/OME_INSTALLATION.md for setup instructions"
                fi
            else
                log_error "OME namespace not found - OME is NOT installed"
                echo ""
                echo "================================================================================"
                echo "ERROR: OME (Open Model Engine) is a required prerequisite for this autotuner"
                echo "================================================================================"
                echo ""
                echo "The inference-autotuner requires OME to:"
                echo "  - Deploy InferenceServices with different parameter configurations"
                echo "  - Manage SGLang runtime instances"
                echo "  - Execute automated parameter tuning experiments"
                echo ""
                echo "Installation options:"
                echo ""
                echo "  Option 1: Automatic installation (Recommended)"
                echo "    ./install.sh --install-ome"
                echo ""
                echo "  Option 2: Manual installation"
                echo "    See detailed guide: docs/OME_INSTALLATION.md"
                echo ""
                echo "  Option 3: Quick install with Helm"
                echo "    helm upgrade --install ome-crd oci://ghcr.io/moirai-internal/charts/ome-crd --namespace ome --create-namespace"
                echo "    helm upgrade --install ome third_party/ome/charts/ome-resources --namespace ome"
                echo ""
                exit 1
            fi
        else
            log_warning "Cannot connect to Kubernetes cluster"
            log_warning "Make sure kubectl is configured with valid credentials"
        fi
    else
        log_warning "kubectl not installed - skipping Kubernetes resource creation"
    fi
else
    log_warning "Skipping Kubernetes resource creation (--skip-k8s)"
fi

##############################################################################
# 10. Summary
##############################################################################

echo ""
echo "============================================================================"
log_success "Installation completed successfully!"
echo "============================================================================"
echo ""

# Determine best deployment mode based on environment
RECOMMENDED_MODE=""
if command -v docker &> /dev/null && docker ps &> /dev/null && command -v nvidia-smi &> /dev/null; then
    if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
        RECOMMENDED_MODE="docker"
        echo "🎉 GPU-enabled Docker mode is available (RECOMMENDED for GPU workloads)"
        echo ""
    fi
fi

if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
    if kubectl get namespace ome &> /dev/null 2>&1; then
        if [ -z "$RECOMMENDED_MODE" ]; then
            RECOMMENDED_MODE="ome"
            echo "✅ OME mode is available"
        else
            echo "✅ OME mode is also available"
        fi
        
        # Check if Kubernetes has GPU resources
        GPU_NODES=$(kubectl get nodes -o json 2>/dev/null | grep -c "nvidia.com/gpu" || echo "0")
        if [ "$GPU_NODES" -gt 0 ]; then
            ALLOCATABLE_GPUS=$(kubectl describe nodes 2>/dev/null | grep "nvidia.com/gpu" | grep "Allocatable" | awk '{sum+=$2} END {print sum}')
            if [ "$ALLOCATABLE_GPUS" -gt 0 ]; then
                echo "   🎉 Kubernetes has $ALLOCATABLE_GPUS GPU(s) available - OME mode can use GPUs"
                RECOMMENDED_MODE="ome"
            else
                echo "   ⚠️  Kubernetes has no allocatable GPUs (Minikube limitation)"
                echo "   ℹ️  Use Docker mode for GPU workloads, OME for orchestration testing"
            fi
        else
            echo "   ⚠️  Kubernetes cannot access GPUs (Minikube Docker driver limitation)"
            echo "   ℹ️  See docs/GPU_DEPLOYMENT_STRATEGY.md for solutions"
        fi
        echo ""
    fi
fi

if [ -z "$RECOMMENDED_MODE" ]; then
    log_warning "Neither Docker nor OME mode is fully configured"
    echo ""
fi

echo "Next steps:"
echo ""
echo "1. Activate the virtual environment (if not already activated):"
echo "   source $VENV_PATH/bin/activate"
echo ""

if [ "$RECOMMENDED_MODE" = "docker" ]; then
    echo "2. RECOMMENDED: Run with Docker mode (GPU support):"
    echo "   python src/run_autotuner.py examples/docker_task.json --mode docker --verbose"
    echo ""
    echo "   Or create your own task:"
    echo "   cp examples/docker_task.json my_task.json"
    echo "   # Edit my_task.json with your parameters"
    echo "   python src/run_autotuner.py my_task.json --mode docker --verbose"
    echo ""
elif [ "$RECOMMENDED_MODE" = "ome" ]; then
    echo "2. Run with OME mode (Kubernetes orchestration):"
    echo "   python src/run_autotuner.py examples/simple_ome_task.json --mode ome --direct --verbose"
    echo ""
    echo "   Note: Use --direct flag for more reliable benchmarking"
    echo ""
else
    echo "2. Configure your deployment mode:"
    echo ""
    echo "   For Docker mode (GPU support):"
    echo "     - Install Docker: https://docs.docker.com/engine/install/"
    echo "     - Install NVIDIA Container Toolkit"
    echo "     - Run: python src/run_autotuner.py examples/docker_task.json --mode docker"
    echo ""
    echo "   For OME mode (Kubernetes):"
    echo "     - Setup Kubernetes cluster with GPU support"
    echo "     - Install OME: ./install.sh --install-ome"
    echo "     - Run: python src/run_autotuner.py examples/simple_ome_task.json --mode ome --direct"
    echo ""
fi

echo "3. Monitor results:"
echo "   # Real-time logs"
echo "   tail -f logs/autotuner.log"
echo ""
echo "   # View results"
echo "   cat results/<task_name>_results.json"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "4. Monitor GPU usage:"
    echo "   watch -n 1 nvidia-smi"
    echo ""
fi

echo "5. Optional: Start Web API (for UI and REST API):"
echo "   # Terminal 1: Start Redis"
echo "   redis-server  # Or: docker run -d -p 6379:6379 redis:alpine"
echo ""
echo "   # Terminal 2: Start backend"
echo "   ./scripts/start_dev.sh"
echo ""
echo "   # Terminal 3: Start frontend"
echo "   cd frontend && npm run dev"
echo ""
echo "   # Access UI at: http://localhost:5173"
echo "   # Access API at: http://localhost:8000/docs"
echo ""

if [ "$SKIP_K8S" = false ] && command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
    echo "Kubernetes verification commands:"
    echo "   kubectl get clusterbasemodels"
    echo "   kubectl get clusterservingruntimes"
    echo "   kubectl get inferenceservices -n autotuner"
    echo "   kubectl get pods -n autotuner"
    echo ""
fi

echo "Database location: $DB_DIR/autotuner.db"
echo ""
echo "Documentation:"
echo "   README.md                              - Project overview"
echo "   docs/QUICK_START_VERIFIED.md          - Quick start commands"
echo "   docs/TASK_COMPLETION_SUMMARY.md       - Full deliverables"
echo "   docs/GPU_DEPLOYMENT_STRATEGY.md       - GPU deployment options"
echo "   docs/DOCKER_MODE.md                   - Docker mode guide"
echo "   docs/OME_INSTALLATION.md              - OME setup guide"
echo "   docs/TROUBLESHOOTING.md               - Common issues"
echo ""

# Final environment check
echo "Environment summary:"
echo "   Python: $(python3 --version | awk '{print $2}')"
if [ -f "$GENAI_BENCH_PATH" ]; then
    echo "   genai-bench: ✅ Installed"
else
    echo "   genai-bench: ❌ Not found"
fi
if command -v kubectl &> /dev/null && kubectl cluster-info &> /dev/null; then
    echo "   Kubernetes: ✅ Accessible"
else
    echo "   Kubernetes: ❌ Not accessible"
fi
if command -v docker &> /dev/null && docker ps &> /dev/null; then
    echo "   Docker: ✅ Running"
else
    echo "   Docker: ❌ Not running"
fi
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "   GPUs: ✅ $GPU_COUNT available"
else
    echo "   GPUs: ❌ Not detected"
fi
echo ""
