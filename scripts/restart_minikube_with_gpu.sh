#!/bin/bash
# Script to restart Minikube with GPU support

set -e

echo "⚠️  WARNING: This will restart Minikube and temporarily disrupt Kubernetes services"
echo "Current Minikube will be deleted and recreated with GPU access"
echo ""
read -p "Continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo ""
echo "Step 1: Stopping current Minikube..."
minikube stop

echo ""
echo "Step 2: Deleting current Minikube cluster..."
minikube delete

echo ""
echo "Step 3: Starting Minikube with GPU support (--driver=none)..."
echo "Note: This requires running as root and removes containerization isolation"

# Start Minikube with --driver=none to access host GPUs
sudo minikube start \
    --driver=none \
    --kubernetes-version=v1.28.0 \
    --extra-config=kubelet.cgroup-driver=systemd

echo ""
echo "Step 4: Installing NVIDIA device plugin..."
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

echo ""
echo "Step 5: Waiting for device plugin to be ready..."
sleep 30

echo ""
echo "Step 6: Verifying GPU access in Kubernetes..."
kubectl describe nodes | grep nvidia.com/gpu

echo ""
echo "✅ Minikube restarted with GPU support!"
echo ""
echo "Next steps:"
echo "1. Reinstall OME operator (it was deleted with the cluster)"
echo "2. Recreate ClusterBaseModel and ClusterServingRuntime resources"
echo "3. Run OME autotuner test"
