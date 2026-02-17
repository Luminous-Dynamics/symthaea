interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular';
  width?: string;
  height?: string;
}

export default function Skeleton({
  className = '',
  variant = 'text',
  width,
  height,
}: SkeletonProps) {
  const baseClass = 'skeleton'; // Uses shimmer animation from animations.css

  const variantClass = {
    text: 'h-4 rounded',
    circular: 'rounded-full',
    rectangular: 'rounded-md',
  }[variant];

  const style: React.CSSProperties = {};
  if (width) style.width = width;
  if (height) style.height = height;

  return <div className={`${baseClass} ${variantClass} ${className}`} style={style} />;
}

export function EmailListSkeleton() {
  return (
    <div className="divide-y divide-gray-200 dark:divide-gray-700">
      {[...Array(8)].map((_, i) => (
        <div key={i} className="px-4 py-3 flex items-start space-x-3">
          {/* Checkbox placeholder */}
          <Skeleton variant="rectangular" width="20px" height="20px" className="mt-1" />

          {/* Avatar placeholder */}
          <Skeleton variant="circular" width="40px" height="40px" className="flex-shrink-0 mt-1" />

          {/* Email content */}
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between mb-2">
              <Skeleton width="35%" height="16px" />
              <Skeleton width="70px" height="14px" />
            </div>
            <Skeleton width="75%" height="14px" className="mb-2" />
            <Skeleton width="90%" height="12px" />
          </div>
        </div>
      ))}
    </div>
  );
}

export function EmailViewSkeleton() {
  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center justify-between mb-4">
          <Skeleton width="100px" height="36px" />
          <div className="flex space-x-2">
            <Skeleton width="80px" height="36px" />
            <Skeleton width="90px" height="36px" />
            <Skeleton width="80px" height="36px" />
          </div>
        </div>
      </div>

      {/* Email Header */}
      <div className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-6">
        <Skeleton width="70%" height="28px" className="mb-4" />

        <div className="space-y-3">
          {/* From with avatar */}
          <div className="flex items-center space-x-3">
            <Skeleton width="60px" className="mr-1" />
            <Skeleton variant="circular" width="48px" height="48px" />
            <Skeleton width="250px" />
          </div>

          {/* To */}
          <div className="flex items-center space-x-3">
            <Skeleton width="60px" className="mr-1" />
            <div className="flex space-x-2">
              <Skeleton variant="circular" width="32px" height="32px" />
              <Skeleton variant="circular" width="32px" height="32px" />
            </div>
            <Skeleton width="200px" />
          </div>

          {/* Date */}
          <div className="flex items-center">
            <Skeleton width="60px" className="mr-4" />
            <Skeleton width="180px" />
          </div>
        </div>
      </div>

      {/* Email Body */}
      <div className="flex-1 p-6 bg-white dark:bg-gray-800 space-y-3 overflow-auto">
        <Skeleton width="100%" height="16px" />
        <Skeleton width="95%" height="16px" />
        <Skeleton width="98%" height="16px" />
        <Skeleton width="92%" height="16px" />
        <div className="h-4" />
        <Skeleton width="97%" height="16px" />
        <Skeleton width="100%" height="16px" />
        <Skeleton width="88%" height="16px" />
        <Skeleton width="94%" height="16px" />
      </div>
    </div>
  );
}

export function FolderListSkeleton() {
  return (
    <div className="space-y-2">
      {[...Array(6)].map((_, i) => (
        <div key={i} className="flex items-center justify-between px-3 py-2">
          <Skeleton width="60%" />
          {i < 2 && <Skeleton width="24px" height="20px" variant="rectangular" />}
        </div>
      ))}
    </div>
  );
}
