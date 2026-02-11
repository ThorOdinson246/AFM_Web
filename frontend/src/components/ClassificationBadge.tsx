'use client';

import clsx from 'clsx';

interface ClassificationBadgeProps {
  className: 'dots' | 'lines' | 'mixed' | 'irregular' | string;
  confidence: number;
  size?: 'sm' | 'md' | 'lg';
}

export function ClassificationBadge({ 
  className, 
  confidence, 
  size = 'md' 
}: ClassificationBadgeProps) {
  const classMap: Record<string, { label: string; color: string; bgColor: string; borderColor: string }> = {
    dots: {
      label: 'Dots',
      color: '#22c55e',
      bgColor: 'rgba(34, 197, 94, 0.15)',
      borderColor: 'rgba(34, 197, 94, 0.3)',
    },
    lines: {
      label: 'Lines',
      color: '#a855f7',
      bgColor: 'rgba(168, 85, 247, 0.15)',
      borderColor: 'rgba(168, 85, 247, 0.3)',
    },
    mixed: {
      label: 'Mixed',
      color: '#fb923c',
      bgColor: 'rgba(251, 146, 60, 0.15)',
      borderColor: 'rgba(251, 146, 60, 0.3)',
    },
    irregular: {
      label: 'Irregular',
      color: '#fb923c',
      bgColor: 'rgba(251, 146, 60, 0.15)',
      borderColor: 'rgba(251, 146, 60, 0.3)',
    },
  };

  const config = classMap[className] || {
    label: className,
    color: '#9ba9bc',
    bgColor: 'rgba(155, 169, 188, 0.15)',
    borderColor: 'rgba(155, 169, 188, 0.3)',
  };

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-3 py-1 text-sm',
    lg: 'px-4 py-1.5 text-base',
  };

  return (
    <div className="flex items-center gap-3">
      <span
        className={clsx(
          'inline-flex items-center rounded-full font-medium uppercase tracking-wide',
          sizeClasses[size]
        )}
        style={{
          color: config.color,
          backgroundColor: config.bgColor,
          border: `1px solid ${config.borderColor}`,
        }}
      >
        {config.label}
      </span>
      <span className="text-gray-600">
        {(confidence * 100).toFixed(1)}% confidence
      </span>
    </div>
  );
}
