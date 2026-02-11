'use client';

import dynamic from 'next/dynamic';
import { useMemo } from 'react';

// Dynamic import for Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ProbabilityChartProps {
  probabilities: Record<string, number>;
  predictedClass: string;
}

export function ProbabilityChart({ probabilities, predictedClass }: ProbabilityChartProps) {
  const chartData = useMemo(() => {
    const labels = Object.keys(probabilities);
    const values = Object.values(probabilities);
    
    // Color mapping
    const colorMap: Record<string, string> = {
      dots: '#22c55e',
      lines: '#a855f7',
      mixed: '#fb923c',
      irregular: '#fb923c',
    };
    
    const colors = labels.map(label => 
      label === predictedClass 
        ? colorMap[label] || '#3b82f6'
        : 'rgba(155, 169, 188, 0.4)'
    );

    return [{
      type: 'bar' as const,
      x: labels.map(l => l.charAt(0).toUpperCase() + l.slice(1)),
      y: values.map(v => v * 100),
      marker: {
        color: colors,
        line: {
          color: colors.map(c => c.replace('0.4', '0.8')),
          width: 1,
        },
      },
      text: values.map(v => `${(v * 100).toFixed(1)}%`),
      textposition: 'outside' as const,
      textfont: {
        color: '#374151',
        size: 12,
      },
      hovertemplate: '%{x}: %{y:.1f}%<extra></extra>',
    }];
  }, [probabilities, predictedClass]);

  const layout = useMemo(() => ({
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {
      color: '#374151',
      family: 'Inter, system-ui, sans-serif',
    },
    margin: { t: 30, r: 20, b: 60, l: 50 },
    xaxis: {
      gridcolor: 'rgba(229, 231, 235, 1)',
      linecolor: '#d1d5db',
      tickfont: { size: 12, color: '#6b7280' },
    },
    yaxis: {
      title: 'Probability (%)',
      gridcolor: 'rgba(229, 231, 235, 1)',
      linecolor: '#d1d5db',
      range: [0, 100],
      tickfont: { size: 11, color: '#6b7280' },
      titlefont: { size: 12, color: '#374151' },
    },
    showlegend: false,
    bargap: 0.4,
  }), []);

  const config = {
    displayModeBar: false,
    responsive: true,
  };

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-900 mb-3">Classification Probabilities</h3>
      <div className="h-[200px]">
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
