'use client';

import { useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface StatItem {
  label: string;
  value: number | string;
  color?: string;
}

interface Props {
  stats: StatItem[];
}

export function PlotlyStatsIndicators({ stats }: Props) {
  const data = useMemo(() => stats.map((stat, idx) => ({
    type: 'indicator' as const,
    mode: 'number' as const,
    value: typeof stat.value === 'number' ? stat.value : undefined,
    number: typeof stat.value === 'number' 
      ? { font: { size: 28, color: stat.color || '#111827' } }
      : { font: { size: 20, color: stat.color || '#111827' } },
    title: { text: stat.label, font: { size: 12, color: '#6b7280' } },
    domain: { x: [idx / stats.length, (idx + 1) / stats.length], y: [0, 1] },
  })), [stats]);

  const layout = useMemo(() => ({
    paper_bgcolor: 'transparent',
    margin: { t: 30, r: 10, l: 10, b: 10 },
    height: 100,
    grid: { rows: 1, columns: stats.length, pattern: 'independent' as const },
  }), [stats.length]);

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '100px' }}
    />
  );
}
