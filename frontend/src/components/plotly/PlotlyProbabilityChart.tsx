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
  probabilities: Record<string, number>;
  predictedClass: string;
}

export function PlotlyProbabilityChart({ probabilities, predictedClass }: Props) {
  const data = useMemo(() => {
    const classes = Object.keys(probabilities);
    const values = classes.map(c => probabilities[c] * 100);
    const colors = classes.map(c => c === predictedClass ? colorMap[c] || '#3b82f6' : 'rgba(156, 163, 175, 0.5)');

    return [{
      type: 'bar' as const,
      x: classes.map(c => c.charAt(0).toUpperCase() + c.slice(1)),
      y: values,
      marker: { color: colors, line: { width: 0 } },
      text: values.map(v => v.toFixed(1) + '%'),
      textposition: 'outside' as const,
      textfont: { color: '#374151', size: 12, family: 'Inter, system-ui, sans-serif' },
      hovertemplate: '%{x}: %{y:.1f}%<extra></extra>',
    }];
  }, [probabilities, predictedClass]);

  const layout = useMemo(() => ({
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 30, r: 20, b: 50, l: 50 },
    yaxis: { title: 'Probability (%)', range: [0, 105], gridcolor: 'rgba(229,231,235,1)', linecolor: '#d1d5db' },
    xaxis: { gridcolor: 'rgba(229,231,235,1)', linecolor: '#d1d5db' },
    font: { family: 'Inter, system-ui, sans-serif', color: '#374151' },
    showlegend: false,
    bargap: 0.4,
    height: 220,
  }), []);

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '220px' }}
    />
  );
}
