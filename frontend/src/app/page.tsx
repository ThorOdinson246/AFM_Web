'use client';

import { useState, useCallback, useEffect } from 'react';
import { 
  Cpu, 
  BarChart3, 
  Activity,
  CheckCircle2,
  XCircle,
  RefreshCw 
} from 'lucide-react';

import { UploadZone } from '@/components/UploadZone';
import { ImageViewer } from '@/components/ImageViewer';
import { ClassificationBadge } from '@/components/ClassificationBadge';
import { ProbabilityChart } from '@/components/ProbabilityChart';
import { AnalysisMetricsChart } from '@/components/AnalysisMetricsChart';
import { AnalysisDetails } from '@/components/AnalysisDetails';
import { 
  PipelineStatus, 
  getDefaultPipelineSteps, 
  updateStepStatus 
} from '@/components/PipelineStatus';

import { 
  analyzeImage, 
  checkHealth, 
  type AnalysisResult, 
  type HealthStatus 
} from '@/lib/api';

type Step = {
  id: string;
  label: string;
  status: 'pending' | 'active' | 'complete' | 'error';
};

export default function Home() {
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [isHealthChecking, setIsHealthChecking] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pipelineSteps, setPipelineSteps] = useState<Step[]>(getDefaultPipelineSteps());

  useEffect(() => {
    const check = async () => {
      setIsHealthChecking(true);
      try {
        const status = await checkHealth();
        setHealth(status);
      } catch (err) {
        setHealth(null);
      }
      setIsHealthChecking(false);
    };
    check();
  }, []);

  const handleFileSelect = useCallback(async (file: File) => {
    setIsAnalyzing(true);
    setError(null);
    setResult(null);
    
    let steps = getDefaultPipelineSteps();
    steps = updateStepStatus(steps, 'upload', 'active');
    setPipelineSteps(steps);

    try {
      steps = updateStepStatus(steps, 'upload', 'complete');
      steps = updateStepStatus(steps, 'segmentation', 'active');
      setPipelineSteps([...steps]);

      const analysisResult = await analyzeImage(file);

      steps = updateStepStatus(steps, 'segmentation', 'complete');
      steps = updateStepStatus(steps, 'classification', 'complete');
      steps = updateStepStatus(steps, 'extraction', 'complete');
      steps = updateStepStatus(steps, 'analysis', 'complete');
      setPipelineSteps([...steps]);

      setResult(analysisResult);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Analysis failed';
      setError(message);
      
      const activeStep = steps.find(s => s.status === 'active');
      if (activeStep) {
        steps = updateStepStatus(steps, activeStep.id, 'error');
        setPipelineSteps([...steps]);
      }
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  const refreshHealth = async () => {
    setIsHealthChecking(true);
    try {
      const status = await checkHealth();
      setHealth(status);
    } catch (err) {
      setHealth(null);
    }
    setIsHealthChecking(false);
  };

  return (
    <main className="min-h-screen py-6 px-4 sm:px-6 lg:px-8 bg-white">
      <div className="max-w-7xl mx-auto">
        {/* Backend Status - Compact */}
        <div className="mb-4 flex items-center justify-end gap-3">
          {isHealthChecking ? (
            <span className="text-sm text-gray-500">Checking backend...</span>
          ) : health ? (
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-green-600" />
              <span className="text-sm text-green-600">Backend connected</span>
            </div>
          ) : (
            <div className="flex items-center gap-2">
              <XCircle className="w-4 h-4 text-red-500" />
              <span className="text-sm text-red-500">Backend disconnected</span>
            </div>
          )}
          <button
            onClick={refreshHealth}
            disabled={isHealthChecking}
            className="p-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 text-gray-600 ${isHealthChecking ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Upload & Pipeline */}
          <div className="lg:col-span-1 space-y-4">
            <UploadZone 
              onFileSelect={handleFileSelect} 
              isLoading={isAnalyzing}
            />
            
            {(isAnalyzing || result) && (
              <PipelineStatus steps={pipelineSteps} />
            )}

            {error && (
              <div className="p-4 rounded-lg border border-red-200 bg-red-50">
                <p className="text-sm text-red-600">{error}</p>
              </div>
            )}
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-2 space-y-4">
            {result ? (
              <>
                {/* Classification Result */}
                <div className="bg-white border border-gray-200 rounded-xl p-4">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Cpu className="w-5 h-5 text-gray-500" />
                      <span className="text-sm font-medium text-gray-700">Classification</span>
                    </div>
                    <span className="text-xs text-gray-400 font-mono">Job: {result.job_id}</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <ClassificationBadge 
                        className={result.predicted_class} 
                        confidence={result.confidence}
                        size="lg"
                      />
                      {/* Probabilities as text */}
                      <div className="mt-3 pt-3 border-t border-gray-100">
                        <p className="text-xs text-gray-500 mb-2">Class Probabilities:</p>
                        <div className="grid grid-cols-2 gap-2">
                          {Object.entries(result.probabilities).map(([cls, prob]) => (
                            <div key={cls} className="text-xs">
                              <span className="font-medium text-gray-700">{cls}:</span>{' '}
                              <span className="text-gray-600">{(prob * 100).toFixed(1)}%</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="h-48">
                      <ProbabilityChart 
                        probabilities={result.probabilities}
                        predictedClass={result.predicted_class}
                      />
                    </div>
                  </div>
                </div>

                {/* Images Grid */}
                <div className="grid grid-cols-2 gap-4">
                  <ImageViewer
                    src={result.original_image}
                    alt="Original AFM image"
                    title="Original Image"
                  />
                  <ImageViewer
                    src={result.mask_image}
                    alt="U-Net segmentation mask"
                    title="U-Net Mask"
                  />
                </div>

                {/* Image Info */}
                {result.image_info && (
                  <div className="bg-white border border-gray-200 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Activity className="w-5 h-5 text-gray-500" />
                      <span className="text-sm font-medium text-gray-700">Image Information</span>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                      <div className="bg-gray-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1">Original Size</p>
                        <p className="text-sm font-semibold text-gray-900">
                          {result.image_info.original_width} × {result.image_info.original_height}
                        </p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1">CNN Input</p>
                        <p className="text-sm font-semibold text-gray-900">
                          {result.image_info.cnn_input_size} × {result.image_info.cnn_input_size}
                        </p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1">U-Net Input</p>
                        <p className="text-sm font-semibold text-gray-900">
                          {result.image_info.unet_input_size} × {result.image_info.unet_input_size}
                        </p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1">Resized</p>
                        <p className="text-sm font-semibold text-gray-900">
                          {result.image_info.will_resize_for_unet ? 'Yes' : 'No'}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Dot Extraction Stats */}
                {result.dot_extraction_stats && (
                  <div className="bg-white border border-gray-200 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Activity className="w-5 h-5 text-gray-500" />
                      <span className="text-sm font-medium text-gray-700">Dot Extraction</span>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      <div className="bg-gray-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1">Total Features</p>
                        <p className="text-lg font-semibold text-gray-900">
                          {result.dot_extraction_stats.total_features}
                        </p>
                      </div>
                      <div className="bg-green-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1">Dots Kept</p>
                        <p className="text-lg font-semibold text-green-600">
                          {result.dot_extraction_stats.kept_features}
                        </p>
                      </div>
                      <div className="bg-orange-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1">Rejected</p>
                        <p className="text-lg font-semibold text-orange-500">
                          {result.dot_extraction_stats.rejected_features}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Extra Outputs (Voronoi/ColorWheel images) */}
                {result.extra_outputs.length > 0 && (
                  <div className="grid grid-cols-2 gap-4">
                    {result.extra_outputs.map((output, idx) => (
                      <ImageViewer
                        key={idx}
                        src={output.image}
                        alt={output.title}
                        title={output.title}
                        subtitle={output.description}
                      />
                    ))}
                  </div>
                )}

                {/* Voronoi Stats (text) */}
                {result.voronoi_stats && Object.keys(result.voronoi_stats).length > 0 && (
                  <div className="bg-white border border-gray-200 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <BarChart3 className="w-5 h-5 text-gray-500" />
                      <span className="text-sm font-medium text-gray-700">Voronoi Statistics (Raw)</span>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-3 overflow-auto max-h-48">
                      <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                        {JSON.stringify(result.voronoi_stats, null, 2)}
                      </pre>
                    </div>
                  </div>
                )}

                {/* Analysis Details - Always show */}
                <div className="bg-white border border-gray-200 rounded-xl p-4">
                  <div className="flex items-center gap-2 mb-3">
                    <BarChart3 className="w-5 h-5 text-gray-500" />
                    <span className="text-sm font-medium text-gray-700">
                      {result.predicted_class === 'lines' ? 'Color Wheel Analysis Details' : 'Voronoi Analysis Details'}
                    </span>
                  </div>
                  {Object.keys(result.analysis_details).length > 0 ? (
                    <div className="space-y-3">
                      {/* Metrics Chart if numeric values exist */}
                      <AnalysisMetricsChart 
                        metrics={result.analysis_details}
                        title=""
                      />
                      {/* Text display */}
                      <AnalysisDetails 
                        details={result.analysis_details}
                        title=""
                      />
                    </div>
                  ) : (
                    <p className="text-sm text-gray-500">No analysis details available</p>
                  )}
                </div>

                {/* Color Wheel Stats */}
                {result.colorwheel_stats && (
                  <div className="bg-white border border-gray-200 rounded-xl p-4">
                    <div className="flex items-center gap-2 mb-3">
                      <Activity className="w-5 h-5 text-gray-500" />
                      <span className="text-sm font-medium text-gray-700">Color Wheel Analysis</span>
                    </div>
                    <div className="grid grid-cols-3 gap-3">
                      <div className="bg-gray-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1">Orientation Angle</p>
                        <p className="text-lg font-semibold text-gray-900">
                          {result.colorwheel_stats.orientation_angle.toFixed(2)}°
                        </p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1">GPU Accelerated</p>
                        <p className="text-lg font-semibold text-gray-900">
                          {result.colorwheel_stats.gpu_accelerated ? 'Yes' : 'No'}
                        </p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-3">
                        <p className="text-xs text-gray-500 mb-1">Grain Masks</p>
                        <p className="text-lg font-semibold text-gray-900">
                          {result.colorwheel_stats.grain_masks_count}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Raw JSON Response (Debug) */}
                <details className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                  <summary className="px-4 py-3 cursor-pointer hover:bg-gray-50 text-sm font-medium text-gray-700">
                    Raw API Response (Debug)
                  </summary>
                  <div className="px-4 pb-4">
                    <pre className="bg-gray-50 rounded-lg p-3 text-xs text-gray-700 overflow-auto max-h-96 font-mono">
                      {JSON.stringify({
                        job_id: result.job_id,
                        predicted_class: result.predicted_class,
                        confidence: result.confidence,
                        probabilities: result.probabilities,
                        image_info: result.image_info,
                        dot_extraction_stats: result.dot_extraction_stats,
                        voronoi_stats: result.voronoi_stats,
                        colorwheel_stats: result.colorwheel_stats,
                        analysis_details: result.analysis_details,
                        extra_outputs_count: result.extra_outputs.length,
                      }, null, 2)}
                    </pre>
                  </div>
                </details>
              </>
            ) : (
              <div className="bg-white border border-gray-200 rounded-xl p-12 text-center">
                <div className="mx-auto w-16 h-16 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                  <BarChart3 className="w-8 h-8 text-gray-400" />
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Upload an Image
                </h3>
                <p className="text-gray-500 text-sm max-w-md mx-auto">
                  Drop an AFM image to run the analysis pipeline.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
