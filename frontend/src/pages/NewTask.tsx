import { useState, useEffect } from 'react';
import { useMutation, useQueryClient, useQuery } from '@tanstack/react-query';
import { apiClient } from '../services/api';
import toast from 'react-hot-toast';
import { navigateTo } from '../components/Layout';
import { getEditingTaskId } from '../utils/editTaskStore';
import type { Task } from '../types/api';
import PresetSelector from '../components/PresetSelector';

interface TaskFormData {
  task_name: string;
  description: string;
  deployment_mode: string;
  base_runtime: string;
  runtime_image_tag?: string;
  model: {
    id_or_path: string;
    namespace: string;
  };
  parameters: Record<string, any[]>;
  optimization: {
    strategy: string;
    objective: string;
    max_iterations: number;
    timeout_per_iteration: number;
  };
  benchmark: {
    task: string;
    model_name: string;
    model_tokenizer: string;
    traffic_scenarios: string[];
    num_concurrency: number[];
    max_time_per_iteration: number;
    max_requests_per_iteration: number;
    additional_params: Record<string, any>;
  };
}

interface ParamField {
  name: string;
  values: string;
}

export default function NewTask() {
  const queryClient = useQueryClient();
  // Edit mode support
  const [editingTaskId, setEditingTaskId] = useState<number | null>(null);
  const [originalTask, setOriginalTask] = useState<Task | null>(null);

  // Check for edit mode on mount
  useEffect(() => {
    const taskId = getEditingTaskId();
    if (taskId) {
      setEditingTaskId(taskId);
    }
  }, []);

  // Fetch task if editing
  const { data: taskToEdit } = useQuery({
    queryKey: ['task', editingTaskId],
    queryFn: () => editingTaskId ? apiClient.getTask(editingTaskId) : null,
    enabled: editingTaskId !== null,
  });

  // Pre-populate form when task data is loaded
  useEffect(() => {
    if (taskToEdit) {
      setOriginalTask(taskToEdit);
      
      // Basic info
      setTaskName(taskToEdit.task_name);
      setDescription(taskToEdit.description || '');
      setDeploymentMode(taskToEdit.deployment_mode);
      setBaseRuntime(taskToEdit.base_runtime);
      setRuntimeImageTag(taskToEdit.runtime_image_tag || '');

      // Model config
      setModelIdOrPath(taskToEdit.model?.id_or_path || '');
      setModelNamespace(taskToEdit.model?.namespace || 'autotuner');

      // Parameters - convert from API format to form format
      const params: ParamField[] = [];
      if (taskToEdit.parameters) {
        for (const [key, value] of Object.entries(taskToEdit.parameters)) {
          if (Array.isArray(value)) {
            params.push({ name: key, values: value.join(', ') });
          }
        }
      }
      if (params.length > 0) {
        setParameters(params);
      }

      // Optimization
      if (taskToEdit.optimization) {
        setStrategy(taskToEdit.optimization.strategy || 'grid_search');
        setObjective(taskToEdit.optimization.objective || 'minimize_latency');
        setMaxIterations(taskToEdit.optimization.max_iterations || 2);
        setTimeoutPerIteration(taskToEdit.optimization.timeout_per_iteration || 600);
      }

      // Benchmark
      if (taskToEdit.benchmark) {
        setBenchmarkTask(taskToEdit.benchmark.task || 'text-to-text');
        setBenchmarkModelName(taskToEdit.benchmark.model_name || '');
        setModelTokenizer(taskToEdit.benchmark.model_tokenizer || '');
        setTrafficScenarios(taskToEdit.benchmark.traffic_scenarios?.join(', ') || 'D(100,100)');
        setNumConcurrency(taskToEdit.benchmark.num_concurrency?.join(', ') || '1, 4');
        setMaxTimePerIteration(taskToEdit.benchmark.max_time_per_iteration || 10);
        setMaxRequestsPerIteration(taskToEdit.benchmark.max_requests_per_iteration || 50);
        setTemperature(taskToEdit.benchmark.additional_params?.temperature?.toString() || '0.0');
      }
    }
  }, [taskToEdit]);

  // Basic info
  const [taskName, setTaskName] = useState('');
  const [description, setDescription] = useState('');
  const [deploymentMode, setDeploymentMode] = useState('docker');
  const [baseRuntime, setBaseRuntime] = useState('sglang');
  const [runtimeImageTag, setRuntimeImageTag] = useState('');

  // Model config
  const [modelIdOrPath, setModelIdOrPath] = useState('');
  const [modelNamespace, setModelNamespace] = useState('autotuner');

  // Parameters (dynamic list)
  const [parameters, setParameters] = useState<ParamField[]>([
    { name: 'tp-size', values: '1' },
    { name: 'mem-fraction-static', values: '0.7, 0.8' },
  ]);
  const [usePresets, setUsePresets] = useState(false);

  // Optimization
  const [strategy, setStrategy] = useState('grid_search');
  const [objective, setObjective] = useState('minimize_latency');
  const [maxIterations, setMaxIterations] = useState(2);
  const [timeoutPerIteration, setTimeoutPerIteration] = useState(600);

  // Benchmark
  const [benchmarkTask, setBenchmarkTask] = useState('text-to-text');
  const [benchmarkModelName, setBenchmarkModelName] = useState('');
  const [modelTokenizer, setModelTokenizer] = useState('');
  const [trafficScenarios, setTrafficScenarios] = useState('D(100,100)');
  const [numConcurrency, setNumConcurrency] = useState('1, 4');
  const [maxTimePerIteration, setMaxTimePerIteration] = useState(10);
  const [maxRequestsPerIteration, setMaxRequestsPerIteration] = useState(50);
  const [temperature, setTemperature] = useState('0.0');

  // Auto-update benchmarkModelName when modelIdOrPath changes
  useEffect(() => {
    if (modelIdOrPath && !benchmarkModelName) {
      const derivedName = modelIdOrPath.split('/').pop() || modelIdOrPath;
      setBenchmarkModelName(derivedName);
    }
  }, [modelIdOrPath, benchmarkModelName]);

  // Auto-update modelTokenizer when modelIdOrPath changes
  useEffect(() => {
    if (modelIdOrPath && !modelTokenizer) {
      setModelTokenizer(modelIdOrPath);
    }
  }, [modelIdOrPath, modelTokenizer]);

  const createTaskMutation = useMutation({
    mutationFn: async (data: TaskFormData) => {
      if (originalTask) {
        // Update existing task
        return await apiClient.updateTask(originalTask.id, data);
      } else {
        // Create new task
        return await apiClient.createTask(data);
      }
    },
    onSuccess: (response) => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
      toast.success(`Task "${response.task_name}" ${originalTask ? 'updated' : 'created'} successfully`);
      navigateTo('tasks');
    },
    onError: (error: any) => {
      const message = error.response?.data?.detail || `Failed to ${originalTask ? 'update' : 'create'} task`;
      toast.error(message);
    },
  });

  const addParameter = () => {
    setParameters([...parameters, { name: '', values: '' }]);
  };

  const removeParameter = (index: number) => {
    setParameters(parameters.filter((_, i) => i !== index));
  };

  const updateParameter = (index: number, field: 'name' | 'values', value: string) => {
    const newParams = [...parameters];
    newParams[index][field] = value;
    setParameters(newParams);
  };

  const handlePresetParametersChange = (presetParams: Record<string, any[]>) => {
    // Convert preset parameters to ParamField format
    const paramFields: ParamField[] = Object.entries(presetParams).map(([name, values]) => ({
      name,
      values: values.join(', ')
    }));

    if (paramFields.length > 0) {
      setParameters(paramFields);
    } else {
      // Reset to default when no presets selected
      setParameters([
        { name: 'tp-size', values: '1' },
        { name: 'mem-fraction-static', values: '0.7, 0.8' },
      ]);
    }
  };

  const parseNumberArray = (str: string): number[] => {
    return str
      .split(',')
      .map((s) => parseFloat(s.trim()))
      .filter((n) => !isNaN(n));
  };

  const parseParameterArray = (str: string): any[] => {
    // Parse comma-separated values, handling numbers, strings, and booleans
    return str
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
      .map((val) => {
        // Try to parse as number first
        const num = parseFloat(val);
        if (!isNaN(num)) {
          return num;
        }
        // Parse boolean
        if (val.toLowerCase() === 'true') {
          return true;
        }
        if (val.toLowerCase() === 'false') {
          return false;
        }
        // Keep as string
        return val;
      });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    // Parse parameters
    const parsedParams: Record<string, any[]> = {};
    for (const param of parameters) {
      if (param.name && param.values) {
        parsedParams[param.name] = parseParameterArray(param.values);
      }
    }

    // Parse traffic scenarios - split by comma but respect parentheses
    // D(100,100), D(200,200) should become ["D(100,100)", "D(200,200)"]
    const trafficScenariosList = trafficScenarios
      .split(/,\s*(?![^()]*\))/)  // Split on comma not inside parentheses
      .map((s) => s.trim())
      .filter(Boolean);

    const formData: TaskFormData = {
      task_name: taskName,
      description,
      deployment_mode: deploymentMode,
      base_runtime: baseRuntime,
      ...(runtimeImageTag && { runtime_image_tag: runtimeImageTag }),
      model: {
        id_or_path: modelIdOrPath,
        namespace: modelNamespace,
      },
      parameters: parsedParams,
      optimization: {
        strategy,
        objective,
        max_iterations: maxIterations,
        timeout_per_iteration: timeoutPerIteration,
      },
      benchmark: {
        task: benchmarkTask,
        model_name: benchmarkModelName || modelIdOrPath.split('/').pop() || modelIdOrPath, // Use editable field or fallback
        model_tokenizer: modelTokenizer || modelIdOrPath, // Auto-fill with full id_or_path if empty
        traffic_scenarios: trafficScenariosList,
        num_concurrency: parseNumberArray(numConcurrency),
        max_time_per_iteration: maxTimePerIteration,
        max_requests_per_iteration: maxRequestsPerIteration,
        additional_params: {
          temperature: parseFloat(temperature),
        },
      },
    };

    createTaskMutation.mutate(formData);
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900">{originalTask ? "Edit Task" : "Create New Task"}</h1>
        <p className="mt-2 text-gray-600">Configure a new autotuning task</p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-8">
        {/* Basic Information */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Basic Information</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Task Name *
              </label>
              <input
                type="text"
                value={taskName}
                onChange={(e) => setTaskName(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="docker-simple-tune"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Deployment Mode *
              </label>
              <select
                value={deploymentMode}
                onChange={(e) => setDeploymentMode(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="docker">Docker</option>
                <option value="ome">OME (Kubernetes)</option>
              </select>
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Description
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                rows={2}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Describe the purpose of this autotuning task"
              />
            </div>
          </div>
        </div>

        {/* Runtime Configuration */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Runtime Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Base Runtime *
              </label>
              <select
                value={baseRuntime}
                onChange={(e) => setBaseRuntime(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="sglang">SGLang</option>
                <option value="vllm">vLLM</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Runtime Image Tag
              </label>
              <input
                type="text"
                value={runtimeImageTag}
                onChange={(e) => setRuntimeImageTag(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="v0.5.2-cu126 (optional)"
              />
              <p className="text-sm text-gray-500 mt-1">
                Docker image tag for the runtime (Docker mode only)
              </p>
            </div>
          </div>
        </div>

        {/* Model Configuration */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Model Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model ID or Path *
              </label>
              <input
                type="text"
                value={modelIdOrPath}
                onChange={(e) => setModelIdOrPath(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="llama-3-2-1b-instruct or meta-llama/Llama-3.2-1B-Instruct"
              />
              <p className="text-sm text-gray-500 mt-1">
                Local model directory name or HuggingFace model ID
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Namespace *
              </label>
              <input
                type="text"
                value={modelNamespace}
                onChange={(e) => setModelNamespace(e.target.value)}
                required
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="autotuner"
              />
            </div>
          </div>
        </div>

        {/* Parameters */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold">Parameters to Tune</h2>
            <div className="flex items-center gap-4">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={usePresets}
                  onChange={(e) => setUsePresets(e.target.checked)}
                  className="rounded"
                />
                <span className="text-gray-700">Use Parameter Presets</span>
              </label>
              <button
                type="button"
                onClick={addParameter}
                className="px-3 py-1 bg-blue-600 text-white rounded-md hover:bg-blue-700 text-sm"
              >
                Add Parameter
              </button>
            </div>
          </div>

          {/* Preset Selector */}
          {usePresets && (
            <div className="mb-6">
              <PresetSelector
                onParametersChange={handlePresetParametersChange}
                className="mb-4"
              />
            </div>
          )}

          <div className="space-y-3">
            {parameters.map((param, index) => (
              <div key={index} className="flex gap-3 items-start">
                <div className="flex-1">
                  <input
                    type="text"
                    value={param.name}
                    onChange={(e) => updateParameter(index, 'name', e.target.value)}
                    placeholder="Parameter name (e.g., tp-size)"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div className="flex-1">
                  <input
                    type="text"
                    value={param.values}
                    onChange={(e) => updateParameter(index, 'values', e.target.value)}
                    placeholder="Values (e.g., 1, 2, 4)"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <button
                  type="button"
                  onClick={() => removeParameter(index)}
                  className="px-3 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                  disabled={parameters.length === 1}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
          <p className="text-sm text-gray-500 mt-2">
            {usePresets
              ? "Parameters from presets are pre-filled below. You can still edit them manually."
              : "Enter parameter values as comma-separated numbers (e.g., 0.7, 0.8, 0.9)"}
          </p>
        </div>

        {/* Optimization Settings */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Optimization Settings</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Strategy
              </label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="grid_search">Grid Search</option>
                <option value="random_search">Random Search</option>
                <option value="bayesian">Bayesian Optimization</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Objective
              </label>
              <select
                value={objective}
                onChange={(e) => setObjective(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="minimize_latency">Minimize Latency</option>
                <option value="maximize_throughput">Maximize Throughput</option>
                <option value="balanced">Balanced</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Iterations
              </label>
              <input
                type="number"
                value={maxIterations}
                onChange={(e) => setMaxIterations(parseInt(e.target.value))}
                min="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Timeout Per Iteration (seconds)
              </label>
              <input
                type="number"
                value={timeoutPerIteration}
                onChange={(e) => setTimeoutPerIteration(parseInt(e.target.value))}
                min="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        </div>

        {/* Benchmark Configuration */}
        <div className="bg-white shadow-sm rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4">Benchmark Configuration</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Benchmark Task
              </label>
              <input
                type="text"
                value={benchmarkTask}
                onChange={(e) => setBenchmarkTask(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="text-to-text"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Name
              </label>
              <input
                type="text"
                value={benchmarkModelName}
                onChange={(e) => setBenchmarkModelName(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Auto-filled from Model ID/Path"
              />
              <p className="text-sm text-gray-500 mt-1">
                Display name for benchmark results (auto-filled but editable)
              </p>
            </div>

            <div className="md:col-span-2">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Model Tokenizer (HuggingFace ID)
              </label>
              <input
                type="text"
                value={modelTokenizer}
                onChange={(e) => setModelTokenizer(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Leave empty to auto-fill with Model ID/Path"
              />
              <p className="text-sm text-gray-500 mt-1">
                HuggingFace model ID for tokenizer. Linked to Model ID/Path above (auto-fills if empty).
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Traffic Scenarios
              </label>
              <input
                type="text"
                value={trafficScenarios}
                onChange={(e) => setTrafficScenarios(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="D(100,100)"
              />
              <p className="text-sm text-gray-500 mt-1">
                Comma-separated list (e.g., D(100,100), D(200,200))
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Concurrency Levels
              </label>
              <input
                type="text"
                value={numConcurrency}
                onChange={(e) => setNumConcurrency(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="1, 4, 8"
              />
              <p className="text-sm text-gray-500 mt-1">
                Comma-separated numbers
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Time Per Iteration (seconds)
              </label>
              <input
                type="number"
                value={maxTimePerIteration}
                onChange={(e) => setMaxTimePerIteration(parseInt(e.target.value))}
                min="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Max Requests Per Iteration
              </label>
              <input
                type="number"
                value={maxRequestsPerIteration}
                onChange={(e) => setMaxRequestsPerIteration(parseInt(e.target.value))}
                min="1"
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Temperature
              </label>
              <input
                type="text"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="0.0"
              />
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-4 justify-end">
          <button
            type="button"
            onClick={() => navigateTo('tasks')}
            className="px-6 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={createTaskMutation.isPending}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-blue-400"
          >
            {createTaskMutation.isPending ? (originalTask ? 'Saving...' : 'Creating...') : (originalTask ? 'Save Changes' : 'Create Task')}
          </button>
        </div>
      </form>
    </div>
  );
}
