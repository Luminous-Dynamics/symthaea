import { useState } from 'react';
import { md5 } from '@/utils/hash';

interface AvatarProps {
  email: string;
  name?: string;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
  showTooltip?: boolean;
  onClick?: () => void;
}

// Generate consistent color based on email
const getColorForEmail = (email: string): string => {
  const colors = [
    '#3B82F6', // Blue
    '#10B981', // Green
    '#F59E0B', // Amber
    '#EF4444', // Red
    '#8B5CF6', // Purple
    '#EC4899', // Pink
    '#06B6D4', // Cyan
    '#F97316', // Orange
    '#14B8A6', // Teal
    '#6366F1', // Indigo
    '#84CC16', // Lime
    '#F43F5E', // Rose
  ];

  // Simple hash based on email
  let hash = 0;
  for (let i = 0; i < email.length; i++) {
    hash = email.charCodeAt(i) + ((hash << 5) - hash);
  }

  return colors[Math.abs(hash) % colors.length];
};

// Get initials from name or email
const getInitials = (email: string, name?: string): string => {
  if (name) {
    const parts = name.trim().split(/\s+/);
    if (parts.length >= 2) {
      return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
    }
    return name.substring(0, 2).toUpperCase();
  }

  // Use email username
  const username = email.split('@')[0];
  if (username.length >= 2) {
    return username.substring(0, 2).toUpperCase();
  }
  return username.toUpperCase();
};

// Get size classes
const getSizeClasses = (size: 'sm' | 'md' | 'lg' | 'xl') => {
  switch (size) {
    case 'sm':
      return {
        container: 'w-8 h-8 text-xs',
        text: 'text-xs',
      };
    case 'md':
      return {
        container: 'w-10 h-10 text-sm',
        text: 'text-sm',
      };
    case 'lg':
      return {
        container: 'w-12 h-12 text-base',
        text: 'text-base',
      };
    case 'xl':
      return {
        container: 'w-16 h-16 text-xl',
        text: 'text-xl',
      };
  }
};

export default function Avatar({
  email,
  name,
  size = 'md',
  className = '',
  showTooltip = true,
  onClick,
}: AvatarProps) {
  const [gravatarFailed, setGravatarFailed] = useState(false);
  const [showTooltipState, setShowTooltipState] = useState(false);

  const initials = getInitials(email, name);
  const backgroundColor = getColorForEmail(email);
  const sizeClasses = getSizeClasses(size);

  // Generate Gravatar URL
  const gravatarUrl = `https://www.gravatar.com/avatar/${md5(email.toLowerCase().trim())}?d=404&s=${
    size === 'sm' ? 32 : size === 'md' ? 40 : size === 'lg' ? 48 : 64
  }`;

  const handleImageError = () => {
    setGravatarFailed(true);
  };

  const avatarContent = (
    <div
      className={`${sizeClasses.container} rounded-full flex items-center justify-center font-semibold flex-shrink-0 relative ${
        onClick ? 'cursor-pointer hover:opacity-80 transition-opacity' : ''
      } ${className}`}
      style={{
        backgroundColor: gravatarFailed ? backgroundColor : undefined,
        color: gravatarFailed ? 'white' : undefined,
      }}
      onClick={onClick}
      onMouseEnter={() => showTooltip && setShowTooltipState(true)}
      onMouseLeave={() => showTooltip && setShowTooltipState(false)}
    >
      {!gravatarFailed ? (
        <img
          src={gravatarUrl}
          alt={name || email}
          className="w-full h-full rounded-full object-cover"
          onError={handleImageError}
        />
      ) : (
        <span className={sizeClasses.text}>{initials}</span>
      )}

      {/* Tooltip */}
      {showTooltip && showTooltipState && (
        <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 dark:bg-gray-700 text-white text-xs rounded-lg shadow-lg whitespace-nowrap z-50">
          <div className="font-semibold">{name || email}</div>
          {name && <div className="text-gray-300 dark:text-gray-400">{email}</div>}
          {/* Arrow */}
          <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px">
            <div className="border-4 border-transparent border-t-gray-900 dark:border-t-gray-700" />
          </div>
        </div>
      )}
    </div>
  );

  return avatarContent;
}

// AvatarGroup component for showing multiple avatars
interface AvatarGroupProps {
  emails: Array<{ email: string; name?: string }>;
  max?: number;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  className?: string;
}

export function AvatarGroup({ emails, max = 3, size = 'md', className = '' }: AvatarGroupProps) {
  const displayedEmails = emails.slice(0, max);
  const remainingCount = emails.length - max;

  return (
    <div className={`flex -space-x-2 ${className}`}>
      {displayedEmails.map((item, index) => (
        <div key={item.email} className="ring-2 ring-white dark:ring-gray-800 rounded-full" style={{ zIndex: displayedEmails.length - index }}>
          <Avatar email={item.email} name={item.name} size={size} />
        </div>
      ))}
      {remainingCount > 0 && (
        <div
          className={`${
            getSizeClasses(size).container
          } rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center font-semibold text-gray-700 dark:text-gray-300 ring-2 ring-white dark:ring-gray-800`}
          style={{ zIndex: 0 }}
        >
          <span className={getSizeClasses(size).text}>+{remainingCount}</span>
        </div>
      )}
    </div>
  );
}
