# Deployment Architecture Documentation Index

This directory contains comprehensive documentation of the inference-autotuner's deployment architecture and infrastructure.

## Documentation Files

### 1. [DEPLOYMENT_QUICK_REFERENCE.md](DEPLOYMENT_QUICK_REFERENCE.md)
**Best for:** Quick lookup and high-level understanding
- One-page summary of deployment architecture
- Two benchmark modes comparison table
- Key components and files list
- Hardcoded configuration values table
- Quick start instructions
- Entry points and CLI usage
- ~217 lines

### 2. [DEPLOYMENT_ARCHITECTURE.md](DEPLOYMENT_ARCHITECTURE.md)
**Best for:** Comprehensive technical understanding
- Executive summary
- Detailed architecture patterns
- Component specifications (OME, InferenceService, BenchmarkJob)
- Deployment logic implementation details
- Configuration file schemas
- Entry points and CLI interfaces
- Deployment assumptions and constraints
- Results output and storage
- Key files and their roles
- ~799 lines

### 3. [DEPLOYMENT_SUMMARY.txt](DEPLOYMENT_SUMMARY.txt)
**Best for:** Structured reference and detailed workflows
- Project purpose and deployment model
- Core architecture components
- Two benchmark execution modes (detailed flows)
- Key configuration files
- Infrastructure dependencies
- Complete deployment workflow with ASCII diagrams
- Hardcoded configuration values
- File statistics and summaries
- Extension opportunities
- ~482 lines

## What You'll Learn

### System Overview
- How the autotuner orchestrates Kubernetes resources
- The role of OME (Open Model Engine) framework
- Two different benchmark execution modes and their tradeoffs
- Parameter flow from user configuration to Kubernetes deployment

### Deployment Mechanisms
- Jinja2 template-based YAML generation
- Kubernetes API interactions (InferenceService and BenchmarkJob CRDs)
- Port-forwarding for direct CLI mode
- Status polling and timeout handling

### Key Components
- `run_autotuner.py` - Main orchestrator
- `ome_controller.py` - InferenceService lifecycle management
- `benchmark_controller.py` - K8s BenchmarkJob management
- `direct_benchmark_controller.py` - CLI mode with port-forwarding
- `inference_service.yaml.j2` - Deployment template
- `benchmark_job.yaml.j2` - Benchmark template

### Configuration & Dependencies
- Task definition JSON schema
- Kubernetes resource requirements (CRDs, namespace, PVC)
- OME framework prerequisites
- Python library dependencies
- Hardcoded values and extension points

### Practical Workflows
- Complete end-to-end experiment lifecycle
- Parameter grid generation (Cartesian product)
- Status polling with timeouts
- Results collection and storage

## Quick Navigation

| Topic | Location |
|-------|----------|
| Quick Start | DEPLOYMENT_QUICK_REFERENCE.md |
| Entry Points | DEPLOYMENT_ARCHITECTURE.md #5 |
| Deployment Modes | DEPLOYMENT_SUMMARY.txt |
| Configuration | DEPLOYMENT_ARCHITECTURE.md #4 |
| Workflow Diagrams | DEPLOYMENT_SUMMARY.txt |
| Hardcoded Values | DEPLOYMENT_QUICK_REFERENCE.md or DEPLOYMENT_SUMMARY.txt |
| Key Files | DEPLOYMENT_ARCHITECTURE.md #9 |
| Infrastructure | DEPLOYMENT_ARCHITECTURE.md #2 |

## Reading Recommendations

**For 5-minute overview:**
→ Read DEPLOYMENT_QUICK_REFERENCE.md

**For implementation details:**
→ Read DEPLOYMENT_ARCHITECTURE.md sections 2-3

**For complete workflow understanding:**
→ Read DEPLOYMENT_SUMMARY.txt "Deployment Workflow" section

**For integration or extension:**
→ Read all three documents, focusing on:
- Hardcoded configuration values
- Extension opportunities section
- API usage patterns

## File Statistics

| Document | Lines | Focus |
|----------|-------|-------|
| DEPLOYMENT_QUICK_REFERENCE.md | 217 | Quick lookup |
| DEPLOYMENT_ARCHITECTURE.md | 799 | Comprehensive reference |
| DEPLOYMENT_SUMMARY.txt | 482 | Structured workflow |
| **Total** | **1,498** | **Complete coverage** |

## Key Findings Summary

### Current Deployment
- **Framework:** Kubernetes + OME (Open Model Engine)
- **Language:** Python orchestrator
- **Configuration:** JSON (tasks) + Jinja2 (templates) + YAML (resources)
- **Execution:** Sequential experiments (no parallelization)
- **Automation:** Manual CLI or externally scheduled

### Two Benchmark Modes
1. **K8s BenchmarkJob** (Default) - Runs in Kubernetes pods
2. **Direct CLI** (Recommended) - Runs locally with port-forwarding

### Core Logic
- Load → Plan → Deploy → Wait → Benchmark → Cleanup → Repeat
- ~670 lines of core Python code
- 2 Jinja2 templates
- Comprehensive error handling and logging

### Infrastructure Assumptions
- Kubernetes v1.28+
- OME pre-installed in `ome` namespace
- GPU support required
- Model path: `/mnt/data/models/{model_name}`
- Namespace: `autotuner`

### Zero Automation Features
- No automatic scheduling (manual invocation)
- No REST API (command-line only)
- No containerized orchestrator
- No Kubernetes Operator
- Extension point: Deploy as CronJob or operator

## Related Documentation

- **Main README:** [README.md](README.md)
- **OME Installation:** [docs/OME_INSTALLATION.md](docs/OME_INSTALLATION.md)
- **Quick Start:** [docs/QUICKSTART.md](docs/QUICKSTART.md)
- **Installation Summary:** [docs/INSTALL_SUMMARY.md](docs/INSTALL_SUMMARY.md)
- **Source Code:** 
  - `src/run_autotuner.py` (orchestrator)
  - `src/controllers/` (controllers)
  - `src/templates/` (YAML templates)
  - `src/utils/` (utilities)

---

**Generated:** October 23, 2025
**Project:** inference-autotuner
**Scope:** Deployment architecture analysis
