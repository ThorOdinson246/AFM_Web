'use client';

interface AnalysisDetailsProps {
  details: Record<string, string | number>;
  title?: string;
}

export function AnalysisDetails({ details, title = 'Analysis Details' }: AnalysisDetailsProps) {
  if (Object.keys(details).length === 0) {
    return null;
  }

  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-900 mb-4">{title}</h3>
      <div className="space-y-2 max-h-[300px] overflow-y-auto">
        {Object.entries(details).map(([key, value]) => (
          <div 
            key={key} 
            className="flex justify-between items-center py-2 px-3 rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors"
          >
            <span className="text-sm text-gray-600 font-mono">
              {key.replace(/_/g, ' ')}
            </span>
            <span className="text-sm text-gray-900 font-mono">
              {typeof value === 'number' 
                ? value.toFixed(4) 
                : String(value).length > 50 
                  ? String(value).substring(0, 50) + '...'
                  : String(value)
              }
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
