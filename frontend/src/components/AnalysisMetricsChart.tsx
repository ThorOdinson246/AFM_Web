'use client';

import dynamic from 'next/dynamic';
import { useMemo } from 'react';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface AnalysisMetricsChartProps {
  metrics: Record<string, string | number>;
  title?: string;
}

export function AnalysisMetricsChart({ metrics, title = 'Analysis Metrics' }: AnalysisMetricsChartProps) {
  const chartData = useMemo(() => {
    // Filter for numeric metrics only
    const numericMetrics: Record<string, number> = {};
    
    Object.entries(metrics).forEach(([key, value]) => {
      if (typeof value === 'number' && !isNaN(value)) {
        numericMetrics[key] = value;
      } else if (typeof value === 'string') {
        const parsed = parseFloat(value);
        if (!isNaN(parsed)) {
          numericMetrics[key] = parsed;
        }
      }
    });

    const labels = Object.keys(numericMetrics).map(key => 
      key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
    );
    const values = Object.values(numericMetrics);

    if (labels.length === 0) {
      return null;
    }

    // Generate gradient colors
    const colors = values.map((_, i) => {
      const hue = 200 + (i * 30) % 60; // Blue to cyan range
      return `hsl(${hue}, 70%, 55%)`;
    });

    return [{
      type: 'bar' as const,
      x: values,
      y: labels,
      orientation: 'h' as const,
      marker: {
        color: colors,
        line: {
          color: colors.map(c => c.replace('55%', '65%')),
          width: 1,
        },
      },
      text: values.map(v => typeof v === 'number' ? v.toFixed(4) : v),
      textposition: 'outside' as const,
      textfont: {
        color: '#374151',
        size: 11,
      },
      hovertemplate: '%{y}: %{x:.4f}<extra></extra>',
    }];
  }, [metrics]);

  const layout = useMemo(() => ({
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      color: '#374151',
      family: 'Inter, system-ui, sans-serif',
    },
    margin: { t: 10, r: 80, b: 40, l: 150 },
    xaxis: {
      gridcolor: '#e5e7eb',
      linecolor: '#d1d5db',
      tickfont: { size: 10, color: '#6b7280' },
    },
    yaxis: {
      gridcolor: '#e5e7eb',
      linecolor: '#d1d5db',
      tickfont: { size: 11, color: '#6b7280' },
      automargin: true,
    },
    showlegend: false,
    bargap: 0.3,
  }), []);

  const config = {
    displayModeBar: false,
    responsive: true,
  };

  if (!chartData) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-900 mb-3">{title}</h3>
        <p className="text-gray-500 text-sm">No numeric metrics available</p>
      </div>
    );
  }

  const height = Math.max(200, Object.keys(metrics).length * 30 + 60);

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-900 mb-3">{title}</h3>
      <div style={{ height: `${height}px` }}>
        <Plot
          data={chartData}
          layout={layout}
          config={config}
          style={{ width: '100%', height: '100%' }}
        />
      </div>
    </div>
  );
}
