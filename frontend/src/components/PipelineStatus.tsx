'use client';

import { CheckCircle2, Clock, AlertCircle, Loader2 } from 'lucide-react';
import clsx from 'clsx';

interface Step {
  id: string;
  label: string;
  status: 'pending' | 'active' | 'complete' | 'error';
}

interface PipelineStatusProps {
  steps: Step[];
}

export function PipelineStatus({ steps }: PipelineStatusProps) {
  return (
    <div className="bg-white border border-gray-200 rounded-xl p-4">
      <h3 className="text-sm font-medium text-gray-900 mb-4">Pipeline Progress</h3>
      <div className="space-y-3">
        {steps.map((step, index) => (
          <div key={step.id} className="flex items-center gap-3">
            <div className={clsx(
              'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
              step.status === 'complete' && 'bg-green-100',
              step.status === 'active' && 'bg-blue-100',
              step.status === 'pending' && 'bg-gray-100',
              step.status === 'error' && 'bg-red-100',
            )}>
              {step.status === 'complete' && (
                <CheckCircle2 className="w-5 h-5 text-green-600" />
              )}
              {step.status === 'active' && (
                <Loader2 className="w-5 h-5 text-blue-600 animate-spin" />
              )}
              {step.status === 'pending' && (
                <Clock className="w-5 h-5 text-gray-400" />
              )}
              {step.status === 'error' && (
                <AlertCircle className="w-5 h-5 text-red-500" />
              )}
            </div>
            
            <div className="flex-1 min-w-0">
              <p className={clsx(
                'text-sm font-medium truncate',
                step.status === 'complete' && 'text-green-600',
                step.status === 'active' && 'text-blue-600',
                step.status === 'pending' && 'text-gray-400',
                step.status === 'error' && 'text-red-500',
              )}>
                {step.label}
              </p>
            </div>

            {index < steps.length - 1 && (
              <div className={clsx(
                'absolute left-[1.45rem] h-3 w-0.5 mt-8',
                step.status === 'complete' ? 'bg-green-300' : 'bg-gray-200'
              )} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function getDefaultPipelineSteps(): Step[] {
  return [
    { id: 'upload', label: 'Image Upload', status: 'pending' },
    { id: 'segmentation', label: 'U-Net Segmentation', status: 'pending' },
    { id: 'classification', label: 'CNN Classification', status: 'pending' },
    { id: 'extraction', label: 'Feature Extraction', status: 'pending' },
    { id: 'analysis', label: 'Voronoi/ColorWheel Analysis', status: 'pending' },
  ];
}

export function updateStepStatus(
  steps: Step[], 
  stepId: string, 
  status: Step['status']
): Step[] {
  return steps.map(step => 
    step.id === stepId ? { ...step, status } : step
  );
}
