'use client';

import { useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

const colorMap: Record<string, string> = {
  dots: '#22c55e',
  lines: '#a855f7',
  mixed: '#fb923c',
  irregular: '#fb923c',
};

interface Props {
  predictedClass: string;
  confidence: number;
}

export function PlotlyClassificationBadge({ predictedClass, confidence }: Props) {
  const color = colorMap[predictedClass] || '#3b82f6';

  const data = useMemo(() => [{
    type: 'indicator' as const,
    mode: 'number' as const,
    value: confidence * 100,
    number: { 
      suffix: '%', 
      font: { size: 48, color: color, family: 'Inter, system-ui, sans-serif' },
    },
    title: { 
      text: '<b>' + predictedClass.toUpperCase() + '</b>', 
      font: { size: 24, color: color, family: 'Inter, system-ui, sans-serif' } 
    },
    domain: { x: [0, 1], y: [0, 1] },
  }], [predictedClass, confidence, color]);

  const layout = useMemo(() => ({
    paper_bgcolor: 'transparent',
    margin: { t: 60, r: 20, l: 20, b: 20 },
    height: 140,
  }), []);

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '140px' }}
    />
  );
}
