import { useQuery } from '@tanstack/react-query';
import { apiClient } from '../services/api';
import type { Experiment } from '../types/api';

interface ExperimentProgressBarProps {
  taskId: number;
  totalExperiments: number;
  successfulExperiments: number;
}

export default function ExperimentProgressBar({
  taskId,
  totalExperiments,
  successfulExperiments
}: ExperimentProgressBarProps) {
  // Fetch experiments for this task
  const { data: experiments = [] } = useQuery<Experiment[]>({
    queryKey: ['experiments', taskId],
    queryFn: () => apiClient.getExperimentsByTask(taskId),
    refetchInterval: 5000, // Auto-refresh every 5 seconds
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

  // If no experiments yet, show placeholder blocks based on total_experiments
  const displayBlocks = experiments.length > 0 ? experiments :
    totalExperiments > 0 ? Array(totalExperiments).fill(null).map((_, i) => ({
      id: i,
      experiment_id: i + 1,
      status: 'pending',
      parameters: {}
    } as Experiment)) : [];

  // Limit blocks to reasonable number for display (max 50)
  const maxBlocks = 50;
  const blocksToShow = displayBlocks.slice(0, maxBlocks);
  const hasMore = displayBlocks.length > maxBlocks;

  return (
    <div className="flex flex-col gap-1">
      {/* Progress text */}
      <div className="text-xs text-gray-600">
        {successfulExperiments} / {totalExperiments} success
      </div>

      {/* Progress blocks */}
      {blocksToShow.length > 0 && (
        <div className="flex flex-wrap gap-0.5">
          {blocksToShow.map((exp, idx) => (
            <div
              key={exp.id || idx}
              className={`w-1.5 h-3 rounded-sm ${getBlockColor(exp.status)} transition-colors duration-300 hover:scale-125 cursor-pointer`}
              title={getBlockTitle(exp)}
            />
          ))}
          {hasMore && (
            <div
              className="text-xs text-gray-500 ml-1 self-center"
              title={`${displayBlocks.length - maxBlocks} more experiments`}
            >
              +{displayBlocks.length - maxBlocks}
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
