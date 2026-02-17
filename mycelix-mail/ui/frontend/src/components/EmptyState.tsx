import { ReactNode } from 'react';

interface EmptyStateProps {
  icon: ReactNode;
  title: string;
  description: string;
  action?: ReactNode;
  tip?: string;
  tipIcon?: string;
}

export default function EmptyState({ icon, title, description, action, tip, tipIcon = '💡' }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 min-h-[300px] animate-fade-in">
      {/* Icon with subtle animation */}
      <div className="text-6xl mb-6 animate-scale-in opacity-50 dark:opacity-40">
        {icon}
      </div>

      {/* Title */}
      <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
        {title}
      </h3>

      {/* Description */}
      <p className="text-gray-600 dark:text-gray-400 mb-6 text-center max-w-md">
        {description}
      </p>

      {/* Action button */}
      {action && <div className="mb-6">{action}</div>}

      {/* Helpful tip */}
      {tip && (
        <div className="mt-4 px-4 py-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg max-w-md">
          <div className="flex items-start space-x-2">
            <span className="text-lg flex-shrink-0">{tipIcon}</span>
            <p className="text-sm text-blue-800 dark:text-blue-200">
              <span className="font-semibold">Tip: </span>
              {tip}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
