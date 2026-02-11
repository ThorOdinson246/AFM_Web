'use client';

import dynamic from 'next/dynamic';
import { useMemo } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ConfidenceGaugeProps {
  confidence: number;
  predictedClass: string;
}

export function ConfidenceGauge({ confidence, predictedClass }: ConfidenceGaugeProps) {
  const colorMap: Record<string, string> = {
    dots: '#22c55e',
    lines: '#a855f7',
    mixed: '#fb923c',
    irregular: '#fb923c',
  };

  const color = colorMap[predictedClass] || '#3b82f6';

  const data = useMemo(() => [{
    type: 'indicator' as const,
    mode: 'gauge+number' as const,
    value: confidence * 100,
    number: {
      suffix: '%',
      font: { size: 28, color: '#111827' },
    },
    gauge: {
      axis: {
        range: [0, 100],
        tickwidth: 1,
        tickcolor: '#d1d5db',
        tickfont: { color: '#6b7280', size: 10 },
      },
      bar: { color: color, thickness: 0.8 },
      bgcolor: '#f3f4f6',
      borderwidth: 0,
      steps: [
        { range: [0, 50], color: '#e5e7eb' },
        { range: [50, 75], color: '#d1d5db' },
        { range: [75, 100], color: '#9ca3af' },
      ],
      threshold: {
        line: { color: color, width: 3 },
        thickness: 0.8,
        value: confidence * 100,
      },
    },
  }], [confidence, color]);

  const layout = useMemo(() => ({
    paper_bgcolor: 'transparent',
    font: { color: '#374151', family: 'Inter, system-ui, sans-serif' },
    margin: { t: 25, r: 25, l: 25, b: 25 },
    height: 180,
  }), []);

  const config = {
    displayModeBar: false,
    responsive: true,
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-900 mb-1 text-center">Model Confidence</h3>
      <Plot
        data={data}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '180px' }}
      />
    </div>
  );
}
