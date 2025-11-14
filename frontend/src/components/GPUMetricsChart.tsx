import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface GPUMonitoringData {
  monitoring_duration_seconds: number;
  sample_count: number;
  gpu_stats: {
    [gpu_index: string]: {
      name: string;
      utilization: {
        min: number;
        max: number;
        mean: number;
        samples: number;
      };
      memory_used_mb: {
        min: number;
        max: number;
        mean: number;
      };
      memory_usage_percent: {
        min: number;
        max: number;
        mean: number;
      };
      temperature_c: {
        min: number;
        max: number;
        mean: number;
      };
      power_draw_w: {
        min: number;
        max: number;
        mean: number;
      };
    };
  };
}

interface GPUMetricsChartProps {
  gpuMonitoring: GPUMonitoringData;
}

export default function GPUMetricsChart({ gpuMonitoring }: GPUMetricsChartProps) {
  // Transform GPU stats into chart data
  const chartData = Object.entries(gpuMonitoring.gpu_stats || {}).map(([gpuIndex, stats]) => ({
    gpu: `GPU ${gpuIndex}`,
    utilization: stats.utilization.mean,
    memory: stats.memory_usage_percent.mean,
    temperature: stats.temperature_c.mean,
    power: stats.power_draw_w.mean,
  }));

  if (chartData.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-4 text-center text-gray-500">
        No GPU monitoring data available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary Statistics */}
      <div className="bg-blue-50 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-blue-900 mb-2">Monitoring Summary</h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-gray-600">Duration:</span>{' '}
            <span className="font-medium">{gpuMonitoring.monitoring_duration_seconds.toFixed(1)}s</span>
          </div>
          <div>
            <span className="text-gray-600">Samples:</span>{' '}
            <span className="font-medium">{gpuMonitoring.sample_count}</span>
          </div>
        </div>
      </div>

      {/* GPU Stats Table */}
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                GPU
              </th>
              <th className="px-4 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Model
              </th>
              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Util (%)
              </th>
              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Memory (%)
              </th>
              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Temp (°C)
              </th>
              <th className="px-4 py-2 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                Power (W)
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {Object.entries(gpuMonitoring.gpu_stats).map(([gpuIndex, stats]) => (
              <tr key={gpuIndex} className="hover:bg-gray-50">
                <td className="px-4 py-2 whitespace-nowrap font-medium text-gray-900">
                  GPU {gpuIndex}
                </td>
                <td className="px-4 py-2 whitespace-nowrap text-gray-600 text-xs">
                  {stats.name}
                </td>
                <td className="px-4 py-2 whitespace-nowrap text-right">
                  <div className="flex flex-col items-end">
                    <span className="font-medium">{stats.utilization.mean.toFixed(1)}%</span>
                    <span className="text-xs text-gray-500">
                      {stats.utilization.min.toFixed(0)}-{stats.utilization.max.toFixed(0)}%
                    </span>
                  </div>
                </td>
                <td className="px-4 py-2 whitespace-nowrap text-right">
                  <div className="flex flex-col items-end">
                    <span className="font-medium">{stats.memory_usage_percent.mean.toFixed(1)}%</span>
                    <span className="text-xs text-gray-500">
                      {stats.memory_used_mb.mean.toFixed(0)} MB
                    </span>
                  </div>
                </td>
                <td className="px-4 py-2 whitespace-nowrap text-right">
                  <div className="flex flex-col items-end">
                    <span className="font-medium">{stats.temperature_c.mean.toFixed(0)}°C</span>
                    <span className="text-xs text-gray-500">
                      {stats.temperature_c.min.toFixed(0)}-{stats.temperature_c.max.toFixed(0)}°C
                    </span>
                  </div>
                </td>
                <td className="px-4 py-2 whitespace-nowrap text-right">
                  <div className="flex flex-col items-end">
                    <span className="font-medium">{stats.power_draw_w.mean.toFixed(1)}W</span>
                    <span className="text-xs text-gray-500">
                      {stats.power_draw_w.min.toFixed(0)}-{stats.power_draw_w.max.toFixed(0)}W
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Utilization Chart */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">GPU Utilization</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="gpu" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="utilization"
                stroke="#3b82f6"
                strokeWidth={2}
                name="Utilization (%)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Memory Chart */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Memory Usage</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="gpu" tick={{ fontSize: 12 }} />
              <YAxis domain={[0, 100]} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="memory"
                stroke="#10b981"
                strokeWidth={2}
                name="Memory (%)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Temperature Chart */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Temperature</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="gpu" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="temperature"
                stroke="#f59e0b"
                strokeWidth={2}
                name="Temperature (°C)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Power Chart */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">Power Draw</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="gpu" tick={{ fontSize: 12 }} />
              <YAxis tick={{ fontSize: 12 }} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="power"
                stroke="#ef4444"
                strokeWidth={2}
                name="Power (W)"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
