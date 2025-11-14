import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../services/api';
import type { Experiment } from '../types/api';

interface ExperimentProgressBarProps {
  taskId: number;
  totalExperiments: number;
  successfulExperiments: number;
  onExperimentClick?: (taskId: number, experimentId: number) => void;
}

export default function ExperimentProgressBar({
  taskId,
  totalExperiments,
  successfulExperiments,
  onExperimentClick
}: ExperimentProgressBarProps) {
  // Fetch experiments for this task
  // Note: No polling needed here - WebSocket updates in parent page will trigger cache invalidation
  const { data: experiments = [] } = useQuery<Experiment[]>({
    queryKey: ['experiments', taskId],
    queryFn: () => apiClient.getExperimentsByTask(taskId),
  });

  const getBlockColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'bg-gray-300';
      case 'deploying':
        return 'bg-blue-400';
      case 'benchmarking':
        return 'bg-blue-500';
      case 'success':
        return 'bg-green-500';
      case 'failed':
        return 'bg-red-500';
      default:
        return 'bg-gray-200';
    }
  };

  const getBlockTitle = (exp: Experiment) => {
    const paramStr = Object.entries(exp.parameters || {})
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ');
    return `Experiment ${exp.experiment_id}\nStatus: ${exp.status}\nParams: ${paramStr}`;
  };

  // Combine actual experiments with planned placeholders
  // Create a map of experiment_id -> experiment for quick lookup
  const experimentsMap = new Map<number, Experiment>();
  experiments.forEach(exp => {
    experimentsMap.set(exp.experiment_id, exp);
  });

  // Build display blocks: actual experiments + planned placeholders up to totalExperiments
  const displayBlocks: Experiment[] = [];
  for (let i = 1; i <= totalExperiments; i++) {
    if (experimentsMap.has(i)) {
      // Use actual experiment
      displayBlocks.push(experimentsMap.get(i)!);
    } else {
      // Create placeholder for planned experiment
      displayBlocks.push({
        id: -i, // Negative ID to distinguish from real experiments
        task_id: taskId,
        experiment_id: i,
        status: 'pending',
        parameters: {},
        created_at: new Date().toISOString(),
      } as Experiment);
    }
  }

  // Sort by experiment_id ascending (earliest first) for display
  const sortedBlocks = [...displayBlocks].sort((a, b) => a.experiment_id - b.experiment_id);

  // Limit blocks to reasonable number for display (max 50)
  const maxBlocks = 50;
  const blocksToShow = sortedBlocks.slice(0, maxBlocks);
  const hasMore = sortedBlocks.length > maxBlocks;

  return (
    <div className="flex flex-col gap-1">
      {/* Progress text */}
      <div className="text-xs text-gray-600">
        {successfulExperiments} / {totalExperiments} success
      </div>

      {/* Progress blocks */}
      {blocksToShow.length > 0 && (
        <div className="flex flex-wrap gap-0.5">
          {blocksToShow.map((exp, idx) => {
            // Only allow clicking on real experiments (not placeholders)
            const isRealExperiment = exp.id > 0;
            // Only show logs for finished or running experiments
            const isClickable = isRealExperiment && onExperimentClick &&
              (exp.status === 'success' || exp.status === 'failed' || exp.status === 'deploying' || exp.status === 'benchmarking');

            return (
              <div
                key={exp.id || idx}
                className={`w-1.5 h-3 rounded-sm ${getBlockColor(exp.status)} transition-colors duration-300 hover:scale-125 ${isClickable ? 'cursor-pointer' : 'cursor-default'}`}
                title={isClickable ? `${getBlockTitle(exp)}\n\nClick to view logs` : getBlockTitle(exp)}
                onClick={() => {
                  if (isClickable) {
                    onExperimentClick(taskId, exp.experiment_id);
                  }
                }}
              />
            );
          })}
          {hasMore && (
            <div
              className="text-xs text-gray-500 ml-1 self-center"
              title={`${sortedBlocks.length - maxBlocks} more experiments`}
            >
              +{sortedBlocks.length - maxBlocks}
            </div>
          )}
        </div>
      )}

      {/* Legend (optional, shown on hover) */}
      <div className="hidden group-hover:flex text-xs text-gray-500 gap-2 mt-1">
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 bg-gray-300 rounded-sm"></div>
          Pending
        </span>
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 bg-blue-500 rounded-sm"></div>
          Running
        </span>
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 bg-green-500 rounded-sm"></div>
          Success
        </span>
        <span className="flex items-center gap-1">
          <div className="w-2 h-2 bg-red-500 rounded-sm"></div>
          Failed
        </span>
      </div>
    </div>
  );
}
