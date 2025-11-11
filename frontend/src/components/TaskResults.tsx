import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/services/api';
import type { Task } from '@/types/api';
import toast from 'react-hot-toast';
import { useEscapeKey } from '@/hooks/useEscapeKey';
import { useState, useMemo, useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  ScatterChart,
  Scatter,
  ZAxis,
} from 'recharts';

interface TaskResultsProps {
  task: Task;
  onClose: () => void;
}

export default function TaskResults({ task, onClose }: TaskResultsProps) {
  // Handle Escape key to close modal
  useEscapeKey(onClose);

  // State for scatter plot axis selection (initialized from localStorage or defaults)
  const [scatterXAxis, setScatterXAxis] = useState<string>(() => {
    return localStorage.getItem('taskResults.scatterXAxis') || 'mean_output_throughput_tokens_per_s';
  });
  const [scatterYAxis, setScatterYAxis] = useState<string>(() => {
    return localStorage.getItem('taskResults.scatterYAxis') || 'num_concurrency';
  });
  const [hoveredExperiment, setHoveredExperiment] = useState<number | null>(null);

  // Save axis selections to localStorage whenever they change
  useEffect(() => {
    localStorage.setItem('taskResults.scatterXAxis', scatterXAxis);
  }, [scatterXAxis]);

  useEffect(() => {
    localStorage.setItem('taskResults.scatterYAxis', scatterYAxis);
  }, [scatterYAxis]);

  // Fetch experiments for this task
  const {
    data: experiments = [],
    isLoading,
  } = useQuery({
    queryKey: ['experiments', task.id],
    queryFn: () => apiClient.getExperimentsByTask(task.id),
  });

  if (isLoading) {
    return (
      <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading results...</p>
        </div>
      </div>
    );
  }

  const successfulExperiments = experiments.filter((exp) => exp.status === 'success');
  const bestExperiment = experiments.find((exp) => exp.id === task.best_experiment_id);

  // Helper function to get parameter differences
  const getParameterDiff = (expParams: any, bestParams: any): string[] => {
    if (!expParams || !bestParams) return [];
    const diffs: string[] = [];

    // Get all parameter keys from both experiments
    const allKeys = new Set([...Object.keys(expParams), ...Object.keys(bestParams)]);

    for (const key of allKeys) {
      const expValue = expParams[key];
      const bestValue = bestParams[key];

      if (expValue !== bestValue) {
        diffs.push(`${key}: ${expValue} (best: ${bestValue})`);
      }
    }

    return diffs;
  };

  // Helper function to format metric value for display
  const formatMetricValue = (value: any): string => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') return value.toFixed(2);
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (typeof value === 'object') return JSON.stringify(value);
    return String(value);
  };

  // Filter metrics to only include primitive values for charts
  const getPrimitiveMetrics = (metrics: any): Record<string, number> => {
    if (!metrics) return {};
    const result: Record<string, number> = {};
    for (const [key, value] of Object.entries(metrics)) {
      if (typeof value === 'number') {
        result[key] = value;
      }
    }
    return result;
  };

  // Prepare data for charts (only primitive numeric values)
  const chartData = successfulExperiments.map((exp) => ({
    name: `Exp ${exp.experiment_id}`,
    experiment_id: exp.experiment_id,
    objective_score: exp.objective_score || 0,
    parameters: exp.parameters, // Include parameters for comparison
    ...getPrimitiveMetrics(exp.metrics),
  }));

  const formatDuration = (seconds: number | null) => {
    if (!seconds) return 'N/A';
    const hours = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    if (hours > 0) return `${hours}h ${mins}m`;
    if (mins > 0) return `${mins}m ${secs}s`;
    return `${secs}s`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'deploying':
        return 'bg-blue-100 text-blue-800';
      case 'benchmarking':
        return 'bg-yellow-100 text-yellow-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  // Extract raw_results for scatter plot (sub-rounds of focused experiment)
  const getScatterDataForExperiment = (exp: any) => {
    if (!exp?.metrics?.raw_results || !Array.isArray(exp.metrics.raw_results)) {
      return [];
    }

    return exp.metrics.raw_results.map((rawResult: any, index: number) => {
      // Flatten nested stats into top-level for easier access
      const flatData: any = {
        round_index: index,
        round_name: `Round ${index + 1}`,
        num_concurrency: rawResult.num_concurrency,
        batch_size: rawResult.batch_size,
        scenario: rawResult.scenario,
        mean_output_throughput_tokens_per_s: rawResult.mean_output_throughput_tokens_per_s,
        mean_input_throughput_tokens_per_s: rawResult.mean_input_throughput_tokens_per_s,
        mean_total_tokens_throughput_tokens_per_s: rawResult.mean_total_tokens_throughput_tokens_per_s,
        requests_per_second: rawResult.requests_per_second,
        error_rate: rawResult.error_rate,
        num_requests: rawResult.num_requests,
        num_completed_requests: rawResult.num_completed_requests,
      };

      // Flatten stats.* fields
      if (rawResult.stats) {
        // Extract mean values from each stat category
        if (rawResult.stats.ttft) {
          Object.entries(rawResult.stats.ttft).forEach(([key, value]) => {
            flatData[`ttft_${key}`] = value;
          });
        }
        if (rawResult.stats.tpot) {
          Object.entries(rawResult.stats.tpot).forEach(([key, value]) => {
            flatData[`tpot_${key}`] = value;
          });
        }
        if (rawResult.stats.e2e_latency) {
          Object.entries(rawResult.stats.e2e_latency).forEach(([key, value]) => {
            flatData[`e2e_latency_${key}`] = value;
          });
        }
        if (rawResult.stats.num_input_tokens) {
          Object.entries(rawResult.stats.num_input_tokens).forEach(([key, value]) => {
            flatData[`input_tokens_${key}`] = value;
          });
        }
        if (rawResult.stats.num_output_tokens) {
          Object.entries(rawResult.stats.num_output_tokens).forEach(([key, value]) => {
            flatData[`output_tokens_${key}`] = value;
          });
        }
      }

      return flatData;
    });
  };

  // Get scatter data for best experiment (always show in green)
  const bestExperimentData = useMemo(() => {
    return bestExperiment ? getScatterDataForExperiment(bestExperiment) : [];
  }, [bestExperiment]);

  // Get scatter data for hovered experiment (show in blue if different from best)
  const hoveredExperimentData = useMemo(() => {
    if (!hoveredExperiment || hoveredExperiment === bestExperiment?.experiment_id) {
      return [];
    }
    const hoveredExp = experiments.find(exp => exp.experiment_id === hoveredExperiment);
    return hoveredExp ? getScatterDataForExperiment(hoveredExp) : [];
  }, [hoveredExperiment, experiments, bestExperiment]);

  // Get all available numeric fields from scatter data for axis selection
  const scatterAxisOptions = useMemo(() => {
    const data = bestExperimentData.length > 0 ? bestExperimentData : hoveredExperimentData;
    return data.length > 0
      ? Object.keys(data[0]).filter(key =>
          typeof data[0][key] === 'number' && key !== 'round_index'
        )
      : [];
  }, [bestExperimentData, hoveredExperimentData]);

  return (
    <div className="fixed inset-0 bg-gray-600 bg-opacity-50 overflow-y-auto z-50">
      <div className="min-h-screen px-4 py-8">
        <div className="max-w-7xl mx-auto bg-white rounded-lg shadow-xl">
          {/* Header */}
          <div className="border-b border-gray-200 px-6 py-4 flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">{task.task_name} - Results</h2>
              <p className="text-sm text-gray-500 mt-1">
                {task.successful_experiments} / {task.total_experiments} successful experiments
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>

          <div className="p-6 space-y-6">
            {/* Best Configuration Card */}
            {bestExperiment && (
              <div className="bg-gradient-to-r from-green-50 to-emerald-50 border-2 border-green-200 rounded-lg p-4">
                <div className="flex items-center mb-3">
                  <svg className="w-5 h-5 text-green-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <h3 className="text-lg font-bold text-green-900">Best Configuration</h3>
                  <span className="ml-auto text-xs text-green-700">Exp #{bestExperiment.experiment_id}</span>
                </div>

                {/* Objective Score on Top */}
                <div className="mb-3">
                  <h4 className="text-xs font-medium text-green-700 mb-1.5">Objective Score</h4>
                  <div className="bg-white rounded-md p-3 border border-green-200 text-center">
                    <div className="text-4xl font-bold text-green-900">
                      {bestExperiment.objective_score?.toFixed(4) || 'N/A'}
                    </div>
                    <p className="text-xs text-green-600 mt-1">
                      {task.optimization?.objective === 'minimize_latency' && 'Lower is better'}
                      {task.optimization?.objective === 'maximize_throughput' && 'Higher is better'}
                      {task.optimization?.objective === 'balanced' && 'Balanced score'}
                    </p>
                  </div>
                </div>

                {/* Parameters and Metrics in a Row */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {/* Parameters Section - Table */}
                  <div>
                    <div className="flex items-center justify-between mb-1.5">
                      <h4 className="text-xs font-medium text-green-700">Optimal Parameters</h4>
                      <button
                        onClick={() => {
                          const paramsText = Object.entries(bestExperiment.parameters)
                            .map(([key, value]) => `${key}=${value}`)
                            .join(' ');
                          navigator.clipboard.writeText(paramsText);
                          toast.success('Parameters copied to clipboard!');
                        }}
                        className="text-xs px-2 py-0.5 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                        title="Copy parameters to clipboard"
                      >
                        Copy
                      </button>
                    </div>
                    <div className="bg-white rounded-md border border-green-200 overflow-hidden">
                      <table className="min-w-full divide-y divide-green-200">
                        <thead className="bg-green-50">
                          <tr>
                            <th className="px-2 py-1 text-left text-xs font-medium text-green-700">Parameter</th>
                            <th className="px-2 py-1 text-right text-xs font-medium text-green-700">Value</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-green-100">
                          {Object.entries(bestExperiment.parameters).map(([key, value]) => (
                            <tr key={key} className="hover:bg-green-50">
                              <td className="px-2 py-1 text-xs font-medium text-gray-700">{key}</td>
                              <td className="px-2 py-1 text-right text-xs font-mono font-bold text-green-800">
                                {String(value)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  {/* Metrics Section - Table */}
                  {bestExperiment.metrics && Object.keys(bestExperiment.metrics).length > 0 && (
                    <div>
                      <h4 className="text-xs font-medium text-green-700 mb-1.5">Performance Metrics</h4>
                      <div className="bg-white rounded-md border border-green-200 overflow-hidden">
                        <table className="min-w-full divide-y divide-green-200">
                          <thead className="bg-green-50">
                            <tr>
                              <th className="px-2 py-1 text-left text-xs font-medium text-green-700">Metric</th>
                              <th className="px-2 py-1 text-right text-xs font-medium text-green-700">Value</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-green-100">
                            {Object.entries(bestExperiment.metrics)
                              .filter(([_, value]) => typeof value === 'number')
                              .map(([key, value]) => (
                                <tr key={key} className="hover:bg-green-50">
                                  <td className="px-2 py-1 text-xs text-gray-700">{key}</td>
                                  <td className="px-2 py-1 text-right text-xs font-semibold text-gray-900">
                                    {formatMetricValue(value)}
                                  </td>
                                </tr>
                              ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Charts */}
            {successfulExperiments.length > 0 && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Objective Score Bar Chart */}
                <div className="bg-white border border-gray-200 rounded-lg p-4">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">
                    Objective Scores by Experiment
                    <span className="ml-3 text-sm font-normal text-gray-500">
                      <span className="inline-block w-3 h-3 bg-green-500 rounded mr-1"></span>
                      Best
                    </span>
                  </h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip
                        content={({ active, payload }) => {
                          if (active && payload && payload.length) {
                            const data = payload[0].payload;
                            const isBest = data.experiment_id === bestExperiment?.experiment_id;
                            const paramDiffs = !isBest && bestExperiment
                              ? getParameterDiff(data.parameters, bestExperiment.parameters)
                              : [];

                            return (
                              <div className="bg-white border border-gray-200 rounded shadow-lg p-3 max-w-sm">
                                <p className="text-sm font-semibold text-gray-900">{data.name}</p>
                                <p className="text-sm text-gray-600">
                                  Score: <span className="font-mono">{(payload[0].value as number).toFixed(4)}</span>
                                </p>
                                {isBest && (
                                  <p className="text-xs text-green-600 font-semibold mt-1">⭐ Best Experiment</p>
                                )}
                                {paramDiffs.length > 0 && (
                                  <div className="mt-2 pt-2 border-t border-gray-200">
                                    <p className="text-xs font-semibold text-gray-700 mb-1">Parameter Differences vs Best:</p>
                                    <div className="space-y-1">
                                      {paramDiffs.map((diff, idx) => (
                                        <p key={idx} className="text-xs text-gray-600 font-mono">
                                          {diff}
                                        </p>
                                      ))}
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          }
                          return null;
                        }}
                      />
                      <Bar
                        dataKey="objective_score"
                        name="Objective Score"
                        onMouseEnter={(data: any) => {
                          if (data && data.experiment_id) {
                            setHoveredExperiment(data.experiment_id);
                          }
                        }}
                        onMouseLeave={() => {
                          setHoveredExperiment(null);
                        }}
                        label={(props: any) => {
                        const { x, y, width, index } = props;
                        if (x === undefined || y === undefined || width === undefined || index === undefined) return null;
                        const isBest = chartData[index]?.experiment_id === bestExperiment?.experiment_id;
                        if (isBest) {
                          return (
                            <text x={Number(x) + Number(width) / 2} y={Number(y) - 5} fill="#10b981" textAnchor="middle" fontSize={16}>
                              ⭐
                            </text>
                          );
                        }
                        return null;
                      }}>
                        {chartData.map((entry, index) => (
                          <Cell
                            key={`cell-${index}`}
                            fill={entry.experiment_id === bestExperiment?.experiment_id ? '#10b981' : '#3b82f6'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Metrics Comparison */}
                {(bestExperimentData.length > 0 || hoveredExperimentData.length > 0) && scatterAxisOptions.length > 0 && (
                  <div className="bg-white border border-gray-200 rounded-lg p-4">
                    {/* Title and Legend */}
                    <div className="mb-3">
                      <h3 className="text-lg font-semibold text-gray-900">
                        Performance Metrics - Sub-Rounds
                      </h3>
                      <div className="flex items-center gap-4 mt-2">
                        {bestExperimentData.length > 0 && (
                          <p className="text-xs text-green-600">
                            <span className="inline-block w-3 h-3 bg-green-500 rounded-full mr-1"></span>
                            Best Exp #{bestExperiment?.experiment_id} ({bestExperimentData.length} rounds)
                          </p>
                        )}
                        {hoveredExperimentData.length > 0 && (
                          <p className="text-xs text-blue-600">
                            <span className="inline-block w-3 h-3 bg-blue-500 rounded-full mr-1"></span>
                            Hovered Exp #{hoveredExperiment} ({hoveredExperimentData.length} rounds)
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Axis Controls */}
                    <div className="flex items-center gap-4 mb-4 pb-3 border-b border-gray-200">
                      <div className="flex items-center gap-2">
                        <label className="text-xs font-medium text-gray-700">X-Axis:</label>
                        <select
                          value={scatterXAxis}
                          onChange={(e) => setScatterXAxis(e.target.value)}
                          className="text-xs border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          {scatterAxisOptions.map((key) => (
                            <option key={key} value={key}>
                              {key}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="flex items-center gap-2">
                        <label className="text-xs font-medium text-gray-700">Y-Axis:</label>
                        <select
                          value={scatterYAxis}
                          onChange={(e) => setScatterYAxis(e.target.value)}
                          className="text-xs border border-gray-300 rounded px-3 py-1.5 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          {scatterAxisOptions.map((key) => (
                            <option key={key} value={key}>
                              {key}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>

                    {/* Chart */}
                    <ResponsiveContainer width="100%" height={300}>
                      <ScatterChart key={`${bestExperiment?.id}-${hoveredExperiment}`}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          type="number"
                          dataKey={scatterXAxis}
                          name={scatterXAxis}
                          label={{ value: scatterXAxis, position: 'insideBottom', offset: -5, fontSize: 11 }}
                        />
                        <YAxis
                          type="number"
                          dataKey={scatterYAxis}
                          name={scatterYAxis}
                          label={{ value: scatterYAxis, angle: -90, position: 'insideLeft', fontSize: 11 }}
                        />
                        <ZAxis range={[100, 400]} />
                        <Tooltip
                          content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                              const data = payload[0].payload;
                              const isFromBest = payload[0].name === 'Best Experiment';
                              return (
                                <div className="bg-white border border-gray-200 rounded shadow-lg p-3 max-w-sm">
                                  <p className="text-sm font-semibold text-gray-900">
                                    {data.round_name} (Concurrency: {data.num_concurrency})
                                  </p>
                                  {isFromBest && (
                                    <p className="text-xs text-green-600 font-semibold">⭐ Best Experiment #{bestExperiment?.experiment_id}</p>
                                  )}
                                  {!isFromBest && (
                                    <p className="text-xs text-blue-600 font-semibold">Experiment #{hoveredExperiment}</p>
                                  )}
                                  <div className="mt-2 space-y-1">
                                    <p className="text-xs text-gray-600">
                                      {scatterXAxis}: <span className="font-mono font-semibold">{data[scatterXAxis]?.toFixed(2)}</span>
                                    </p>
                                    <p className="text-xs text-gray-600">
                                      {scatterYAxis}: <span className="font-mono font-semibold">{data[scatterYAxis]?.toFixed(2)}</span>
                                    </p>
                                    <p className="text-xs text-gray-600">
                                      Scenario: <span className="font-mono">{data.scenario}</span>
                                    </p>
                                    <p className="text-xs text-gray-600">
                                      Requests: <span className="font-mono">{data.num_completed_requests}/{data.num_requests}</span>
                                    </p>
                                    {data.error_rate > 0 && (
                                      <p className="text-xs text-red-600">
                                        Error Rate: <span className="font-mono">{(data.error_rate * 100).toFixed(2)}%</span>
                                      </p>
                                    )}
                                  </div>
                                </div>
                              );
                            }
                            return null;
                          }}
                        />
                        <Legend />

                        {/* Best experiment data (green dots) */}
                        {bestExperimentData.length > 0 && (
                          <Scatter
                            name="Best Experiment"
                            data={bestExperimentData}
                            fill="#10b981"
                            shape={(props: any) => {
                              const { cx, cy } = props;
                              return (
                                <circle
                                  cx={cx}
                                  cy={cy}
                                  r={6}
                                  fill="#10b981"
                                  stroke="#059669"
                                  strokeWidth={1.5}
                                  opacity={0.8}
                                />
                              );
                            }}
                          />
                        )}

                        {/* Hovered experiment data (blue dots) */}
                        {hoveredExperimentData.length > 0 && (
                          <Scatter
                            name="Hovered Experiment"
                            data={hoveredExperimentData}
                            fill="#3b82f6"
                            shape={(props: any) => {
                              const { cx, cy } = props;
                              return (
                                <circle
                                  cx={cx}
                                  cy={cy}
                                  r={6}
                                  fill="#3b82f6"
                                  stroke="#2563eb"
                                  strokeWidth={1.5}
                                  opacity={0.8}
                                />
                              );
                            }}
                          />
                        )}
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            )}

            {/* Experiments Table */}
            <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
              <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">All Experiments</h3>
              </div>
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">ID</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Parameters</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Objective Score</th>
                      <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Duration</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {experiments.map((exp) => (
                      <tr
                        key={exp.id}
                        className={exp.id === task.best_experiment_id ? 'bg-green-50' : ''}
                      >
                        <td className="px-4 py-3 whitespace-nowrap">
                          <div className="flex items-center">
                            <span className="text-sm font-medium text-gray-900">#{exp.experiment_id}</span>
                            {exp.id === task.best_experiment_id && (
                              <span className="ml-2 text-xs bg-green-100 text-green-800 px-2 py-0.5 rounded-full">
                                Best
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStatusColor(exp.status)}`}>
                            {exp.status}
                          </span>
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm space-y-0.5">
                            {Object.entries(exp.parameters).map(([key, value]) => (
                              <div key={key} className="flex gap-2">
                                <span className="text-gray-500">{key}:</span>
                                <span className="font-mono text-gray-900">{String(value)}</span>
                              </div>
                            ))}
                          </div>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap">
                          <span className="text-sm font-mono text-gray-900">
                            {exp.objective_score !== null ? exp.objective_score.toFixed(4) : 'N/A'}
                          </span>
                        </td>
                        <td className="px-4 py-3 whitespace-nowrap text-sm text-gray-500">
                          {formatDuration(exp.elapsed_time)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Task Summary */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="text-sm text-blue-600 font-medium">Total Experiments</div>
                <div className="text-2xl font-bold text-blue-900 mt-1">{task.total_experiments}</div>
              </div>
              <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                <div className="text-sm text-green-600 font-medium">Successful</div>
                <div className="text-2xl font-bold text-green-900 mt-1">{task.successful_experiments}</div>
              </div>
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="text-sm text-red-600 font-medium">Failed</div>
                <div className="text-2xl font-bold text-red-900 mt-1">
                  {task.total_experiments - task.successful_experiments}
                </div>
              </div>
              <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                <div className="text-sm text-purple-600 font-medium">Total Duration</div>
                <div className="text-2xl font-bold text-purple-900 mt-1">{formatDuration(task.elapsed_time)}</div>
              </div>
            </div>

            {/* Close Button */}
            <div className="flex justify-end pt-4 border-t border-gray-200">
              <button
                onClick={onClose}
                className="px-6 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700 transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
