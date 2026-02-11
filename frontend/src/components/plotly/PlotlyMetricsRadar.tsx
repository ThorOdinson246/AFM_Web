'use client';

import { useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface Props {
  metrics: Record<string, unknown>;
  title: string;
}

export function PlotlyMetricsRadar({ metrics, title }: Props) {
  const numericMetrics = useMemo(() => {
    return Object.entries(metrics)
      .filter(([, v]) => typeof v === 'number' && !isNaN(v as number))
      .slice(0, 8);
  }, [metrics]);

  const data = useMemo(() => {
    if (numericMetrics.length < 3) return [];
    
    const labels = numericMetrics.map(([k]) => k);
    const values = numericMetrics.map(([, v]) => v as number);
    const maxVal = Math.max(...values.map(Math.abs)) || 1;
    const normalized = values.map(v => (v / maxVal) * 100);

    return [{
      type: 'scatterpolar' as const,
      r: [...normalized, normalized[0]],
      theta: [...labels, labels[0]],
      fill: 'toself',
      fillcolor: 'rgba(59, 130, 246, 0.2)',
      line: { color: '#3b82f6', width: 2 },
      hovertemplate: '%{theta}: %{text}<extra></extra>',
      text: [...values.map(v => v.toFixed(4)), values[0].toFixed(4)],
    }];
  }, [numericMetrics]);

  const layout = useMemo(() => ({
    polar: {
      radialaxis: { visible: true, range: [0, 100], gridcolor: 'rgba(229,231,235,1)' },
      angularaxis: { gridcolor: 'rgba(229,231,235,1)' },
      bgcolor: 'transparent',
    },
    paper_bgcolor: 'transparent',
    font: { family: 'Inter, system-ui, sans-serif', color: '#374151', size: 10 },
    margin: { t: 40, r: 60, l: 60, b: 40 },
    height: 300,
    showlegend: false,
    title: { text: title, font: { size: 14, color: '#374151' } },
  }), [title]);

  if (numericMetrics.length < 3) return null;

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '300px' }}
    />
  );
}
