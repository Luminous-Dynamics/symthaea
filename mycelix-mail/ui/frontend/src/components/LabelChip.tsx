import type { Label } from '@/store/labelStore';

interface LabelChipProps {
  label: Label;
  onRemove?: () => void;
  onClick?: () => void;
  size?: 'sm' | 'md' | 'lg';
  removable?: boolean;
  clickable?: boolean;
}

export default function LabelChip({
  label,
  onRemove,
  onClick,
  size = 'sm',
  removable = false,
  clickable = false,
}: LabelChipProps) {
  // Adjust opacity of the color for background
  const hexToRgba = (hex: string, alpha: number) => {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  };

  const backgroundColor = hexToRgba(label.color, 0.15);
  const borderColor = hexToRgba(label.color, 0.3);

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-2.5 py-1',
    lg: 'text-base px-3 py-1.5',
  };

  const handleClick = (e: React.MouseEvent) => {
    if (clickable && onClick) {
      e.stopPropagation();
      onClick();
    }
  };

  const handleRemove = (e: React.MouseEvent) => {
    e.stopPropagation();
    onRemove?.();
  };

  return (
    <span
      className={`inline-flex items-center rounded-full font-medium border ${sizeClasses[size]} ${
        clickable ? 'cursor-pointer hover:opacity-80' : ''
      } transition-opacity`}
      style={{
        backgroundColor,
        borderColor,
        color: label.color,
      }}
      onClick={handleClick}
      title={label.name}
    >
      {label.icon && <span className="mr-1">{label.icon}</span>}
      <span>{label.name}</span>
      {removable && onRemove && (
        <button
          onClick={handleRemove}
          className="ml-1 hover:bg-white hover:bg-opacity-30 rounded-full w-4 h-4 flex items-center justify-center transition-colors"
          aria-label={`Remove ${label.name} label`}
        >
          <svg
            className="w-3 h-3"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M6 18L18 6M6 6l12 12"
            />
          </svg>
        </button>
      )}
    </span>
  );
}
