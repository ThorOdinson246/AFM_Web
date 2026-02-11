'use client';

import { useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface Props {
  details: Record<string, unknown>;
}

export function PlotlyDetailsBar({ details }: Props) {
  const numericDetails = useMemo(() => {
    return Object.entries(details)
      .filter(([, v]) => typeof v === 'number' && !isNaN(v as number))
      .slice(0, 10);
  }, [details]);

  const data = useMemo(() => {
    if (numericDetails.length === 0) return [];
    
    const labels = numericDetails.map(([k]) => k);
    const values = numericDetails.map(([, v]) => v as number);
    const maxVal = Math.max(...values.map(Math.abs)) || 1;
    const normalized = values.map(v => (v / maxVal) * 100);

    return [{
      type: 'bar' as const,
      y: labels,
      x: normalized,
      orientation: 'h' as const,
      marker: { 
        color: normalized.map(v => v > 50 ? '#3b82f6' : '#93c5fd'),
      },
      text: values.map(v => v.toFixed(4)),
      textposition: 'outside' as const,
      textfont: { size: 10, color: '#374151' },
      hovertemplate: '%{y}: %{text}<extra></extra>',
    }];
  }, [numericDetails]);

  const chartHeight = useMemo(() => Math.max(200, numericDetails.length * 30), [numericDetails.length]);

  const layout = useMemo(() => ({
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 20, r: 80, l: 150, b: 30 },
    xaxis: { title: 'Normalized Value', range: [0, 120], gridcolor: 'rgba(229,231,235,1)' },
    yaxis: { gridcolor: 'rgba(229,231,235,1)', automargin: true },
    font: { family: 'Inter, system-ui, sans-serif', color: '#374151', size: 11 },
    height: chartHeight,
    showlegend: false,
  }), [chartHeight]);

  if (numericDetails.length === 0) return null;

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: chartHeight + 'px' }}
    />
  );
}
