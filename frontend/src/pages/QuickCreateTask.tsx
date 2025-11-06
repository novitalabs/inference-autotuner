import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import toast from "react-hot-toast";
import apiClient from "@/services/api";
import ProfileSelector from "@/components/ProfileSelector";
import { navigateTo } from "@/components/Layout";
import { useEscapeKey } from "@/hooks/useEscapeKey";
import type { TaskContextCreate } from "@/types/api";

export default function QuickCreateTask() {
	// Handle Escape key to go back to tasks page
	useEscapeKey(() => navigateTo("tasks"));
	const queryClient = useQueryClient();
	const [selectedProfiles, setSelectedProfiles] = useState<string[]>([]);
	const [formData, setFormData] = useState<Partial<TaskContextCreate>>({
		model_name: "llama-3-2-1b-instruct",
		base_runtime: "sglang",
		deployment_mode: "docker",
		profiles: []
	});

	const createTaskMutation = useMutation({
		mutationFn: (context: TaskContextCreate) => apiClient.createTaskFromContext(context),
		onSuccess: (response) => {
			toast.success(`Task created: ${response.task.task_name}`);
			queryClient.invalidateQueries({ queryKey: ["tasks"] });
			// Navigate to tasks page after creation
			navigateTo("tasks");
		},
		onError: (error: any) => {
			console.error("Failed to create task:", error);
			// Error toast is handled by apiClient interceptor
		}
	});

	const handleSelectProfile = (profileName: string) => {
		const newProfiles = [...selectedProfiles, profileName];
		setSelectedProfiles(newProfiles);
		setFormData({ ...formData, profiles: newProfiles });
	};

	const handleDeselectProfile = (profileName: string) => {
		const newProfiles = selectedProfiles.filter(p => p !== profileName);
		setSelectedProfiles(newProfiles);
		setFormData({ ...formData, profiles: newProfiles });
	};

	const handleInputChange = (field: keyof TaskContextCreate, value: any) => {
		setFormData({ ...formData, [field]: value });
	};

	const handleSubmit = (e: React.FormEvent) => {
		e.preventDefault();

		// Validation
		if (!formData.model_name) {
			toast.error("Model name is required");
			return;
		}
		if (!formData.base_runtime) {
			toast.error("Base runtime is required");
			return;
		}
		if (selectedProfiles.length === 0) {
			toast.error("Please select at least one profile");
			return;
		}

		// Create task
		createTaskMutation.mutate(formData as TaskContextCreate);
	};

	const handleReset = () => {
		setFormData({
			model_name: "llama-3-2-1b-instruct",
			base_runtime: "sglang",
			deployment_mode: "docker",
			profiles: []
		});
		setSelectedProfiles([]);
	};

	return (
		<div className="space-y-6">
			{/* Header */}
			<div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
				<div className="flex items-center justify-between">
					<div>
						<h2 className="text-2xl font-bold text-gray-900">Quick Create Task</h2>
						<p className="mt-1 text-sm text-gray-600">
							Create a tuning task using configuration profiles - fast and easy!
						</p>
					</div>
					<button
						onClick={() => navigateTo("new-task")}
						className="px-4 py-2 text-sm text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
					>
						Advanced Mode
					</button>
				</div>
			</div>

			{/* Form */}
			<form onSubmit={handleSubmit} className="space-y-6">
				{/* Basic Configuration */}
				<div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
					<h3 className="text-lg font-semibold text-gray-900 mb-4">Basic Configuration</h3>

					<div className="grid grid-cols-1 md:grid-cols-2 gap-6">
						{/* Model Name */}
						<div>
							<label className="block text-sm font-medium text-gray-700 mb-2">
								Model Name <span className="text-red-500">*</span>
							</label>
							<input
								type="text"
								value={formData.model_name || ""}
								onChange={(e) => handleInputChange("model_name", e.target.value)}
								className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
								placeholder="e.g., llama-3-2-1b-instruct"
								required
							/>
							<p className="mt-1 text-xs text-gray-500">
								Directory name in /mnt/data/models/ or HuggingFace model ID
							</p>
						</div>

						{/* Base Runtime */}
						<div>
							<label className="block text-sm font-medium text-gray-700 mb-2">
								Base Runtime <span className="text-red-500">*</span>
							</label>
							<select
								value={formData.base_runtime || "sglang"}
								onChange={(e) => handleInputChange("base_runtime", e.target.value)}
								className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
								required
							>
								<option value="sglang">SGLang</option>
								<option value="vllm">vLLM</option>
							</select>
							<p className="mt-1 text-xs text-gray-500">
								Inference engine to use for serving
							</p>
						</div>

						{/* Deployment Mode */}
						<div>
							<label className="block text-sm font-medium text-gray-700 mb-2">
								Deployment Mode <span className="text-red-500">*</span>
							</label>
							<select
								value={formData.deployment_mode || "docker"}
								onChange={(e) => handleInputChange("deployment_mode", e.target.value)}
								className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
								required
							>
								<option value="docker">Docker (Standalone)</option>
								<option value="ome">OME (Kubernetes)</option>
							</select>
							<p className="mt-1 text-xs text-gray-500">
								Docker for development, OME for production
							</p>
						</div>

						{/* GPU Type (Optional) */}
						<div>
							<label className="block text-sm font-medium text-gray-700 mb-2">
								GPU Type (Optional)
							</label>
							<input
								type="text"
								value={formData.gpu_type || ""}
								onChange={(e) => handleInputChange("gpu_type", e.target.value)}
								className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
								placeholder="e.g., A100, V100"
							/>
							<p className="mt-1 text-xs text-gray-500">
								Limits tp-size based on available GPUs
							</p>
						</div>
					</div>
				</div>

				{/* Profile Selection */}
				<div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
					<ProfileSelector
						selectedProfiles={selectedProfiles}
						onSelectProfile={handleSelectProfile}
						onDeselectProfile={handleDeselectProfile}
						multiSelect={true}
					/>
					<p className="mt-4 text-sm text-gray-500">
						💡 <strong>Tip:</strong> Select multiple profiles to combine configurations.
						Later profiles override earlier ones.
					</p>
				</div>

				{/* Advanced Options (Optional) */}
				<div className="bg-white rounded-lg shadow-sm p-6 border border-gray-200">
					<details className="group">
						<summary className="cursor-pointer text-lg font-semibold text-gray-900 list-none flex items-center justify-between">
							<span>Advanced Options (Optional)</span>
							<svg
								className="w-5 h-5 text-gray-500 group-open:rotate-180 transition-transform"
								fill="none"
								viewBox="0 0 24 24"
								stroke="currentColor"
							>
								<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
							</svg>
						</summary>

						<div className="mt-4 space-y-4">
							{/* Override Mode */}
							<div>
								<label className="block text-sm font-medium text-gray-700 mb-2">
									Override Mode
								</label>
								<select
									value={formData.override_mode || "patch"}
									onChange={(e) => handleInputChange("override_mode", e.target.value as "patch" | "replace")}
									className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
								>
									<option value="patch">Patch (Merge with profiles)</option>
									<option value="replace">Replace (Override profiles completely)</option>
								</select>
								<p className="mt-1 text-xs text-gray-500">
									How to handle user overrides
								</p>
							</div>

							{/* Total GPUs */}
							<div>
								<label className="block text-sm font-medium text-gray-700 mb-2">
									Total GPUs Available
								</label>
								<input
									type="number"
									min="1"
									value={formData.total_gpus || ""}
									onChange={(e) => handleInputChange("total_gpus", parseInt(e.target.value) || undefined)}
									className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
									placeholder="e.g., 1, 2, 4, 8"
								/>
								<p className="mt-1 text-xs text-gray-500">
									Constrains tp-size parameter choices
								</p>
							</div>
						</div>
					</details>
				</div>

				{/* Action Buttons */}
				<div className="flex items-center justify-between bg-white rounded-lg shadow-sm p-6 border border-gray-200">
					<button
						type="button"
						onClick={handleReset}
						className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
					>
						Reset Form
					</button>

					<div className="flex items-center space-x-3">
						<button
							type="button"
							onClick={() => navigateTo("tasks")}
							className="px-4 py-2 text-sm font-medium text-gray-700 bg-white border border-gray-300 rounded-md hover:bg-gray-50"
						>
							Cancel
						</button>
						<button
							type="submit"
							disabled={createTaskMutation.isPending || selectedProfiles.length === 0}
							className="px-6 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
						>
							{createTaskMutation.isPending ? "Creating..." : "Create Task"}
						</button>
					</div>
				</div>

				{/* Profile Summary */}
				{selectedProfiles.length > 0 && (
					<div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
						<div className="flex items-start">
							<svg
								className="w-5 h-5 text-blue-600 mt-0.5 mr-3 flex-shrink-0"
								fill="currentColor"
								viewBox="0 0 20 20"
							>
								<path
									fillRule="evenodd"
									d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
									clipRule="evenodd"
								/>
							</svg>
							<div>
								<h4 className="text-sm font-semibold text-blue-900">Selected Profiles</h4>
								<p className="mt-1 text-sm text-blue-700">
									{selectedProfiles.join(" + ")}
								</p>
								<p className="mt-2 text-xs text-blue-600">
									These profiles will be applied in order to generate the final task configuration.
								</p>
							</div>
						</div>
					</div>
				)}
			</form>
		</div>
	);
}
