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
  confidence: number;
  predictedClass: string;
}

export function PlotlyConfidenceGauge({ confidence, predictedClass }: Props) {
  const color = colorMap[predictedClass] || '#3b82f6';

  const data = useMemo(() => [{
    type: 'indicator' as const,
    mode: 'gauge+number' as const,
    value: confidence * 100,
    number: { suffix: '%', font: { size: 32, color: color } },
    gauge: {
      axis: { range: [0, 100], tickwidth: 1, tickcolor: '#d1d5db' },
      bar: { color: color },
      bgcolor: '#f3f4f6',
      borderwidth: 0,
      steps: [
        { range: [0, 50], color: 'rgba(239, 68, 68, 0.1)' },
        { range: [50, 75], color: 'rgba(251, 191, 36, 0.1)' },
        { range: [75, 100], color: 'rgba(34, 197, 94, 0.1)' },
      ],
    },
    title: { text: 'Model Confidence', font: { size: 14, color: '#6b7280' } },
  }], [confidence, color]);

  const layout = useMemo(() => ({
    paper_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif' },
    margin: { t: 50, r: 30, l: 30, b: 20 },
    height: 200,
  }), []);

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '200px' }}
    />
  );
}
