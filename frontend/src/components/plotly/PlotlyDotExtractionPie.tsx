'use client';

import { useMemo } from 'react';
import dynamic from 'next/dynamic';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface Props {
  kept: number;
  rejected: number;
}

export function PlotlyDotExtractionPie({ kept, rejected }: Props) {
  const data = useMemo(() => [{
    type: 'pie' as const,
    values: [kept, rejected],
    labels: ['Kept', 'Rejected'],
    marker: { colors: ['#22c55e', '#f97316'] },
    textinfo: 'label+value' as const,
    textfont: { size: 12, color: '#ffffff' },
    hovertemplate: '%{label}: %{value}<extra></extra>',
    hole: 0.4,
  }], [kept, rejected]);

  const layout = useMemo(() => ({
    paper_bgcolor: 'transparent',
    margin: { t: 30, r: 20, l: 20, b: 30 },
    height: 200,
    showlegend: true,
    legend: { orientation: 'h' as const, y: -0.1 },
    font: { family: 'Inter, system-ui, sans-serif' },
    annotations: [{
      text: String(kept + rejected),
      showarrow: false,
      font: { size: 20, color: '#374151' },
    }],
  }), [kept, rejected]);

  return (
    <Plot
      data={data}
      layout={layout}
      config={{ displayModeBar: false, responsive: true }}
      style={{ width: '100%', height: '200px' }}
    />
  );
}
