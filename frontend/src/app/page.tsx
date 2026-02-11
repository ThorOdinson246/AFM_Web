'use client';

import { useState, useCallback, useEffect } from 'react';
import { 
  CheckCircle2,
  XCircle,
  RefreshCw,
  Cpu,
  Activity,
  BarChart3,
  Circle,
  Palette,
  Moon,
  Sun
} from 'lucide-react';

import { UploadZone } from '@/components/UploadZone';
import { ImageViewer } from '@/components/ImageViewer';
import { 
  PipelineStatus, 
  getDefaultPipelineSteps, 
  updateStepStatus 
} from '@/components/PipelineStatus';

import { 
  PlotlyClassificationBadge,
  PlotlyProbabilityChart,
  PlotlyConfidenceGauge,
  PlotlyStatsIndicators,
  PlotlyMetricsRadar,
  PlotlyDotExtractionPie,
  PlotlyDetailsBar
} from '@/components/plotly';

import { useTheme } from '@/contexts/ThemeContext';

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
  const { toggleTheme, isDark } = useTheme();
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
      } catch {
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
    } catch {
      setHealth(null);
    }
    setIsHealthChecking(false);
  };

  const cardClass = isDark 
    ? "bg-slate-800 border border-slate-700 rounded-xl p-4"
    : "bg-white border border-gray-200 rounded-xl p-4";
  
  const textMuted = isDark ? "text-slate-400" : "text-gray-500";
  const textPrimary = isDark ? "text-slate-100" : "text-gray-700";

  return (
    <main className={`min-h-screen py-6 px-4 sm:px-6 lg:px-8 ${isDark ? 'bg-slate-900' : 'bg-white'}`}>
      <div className="max-w-7xl mx-auto">
        <div className="mb-4 flex items-center justify-between">
          <h1 className={`text-xl font-semibold ${textPrimary}`}>AFM Analysis</h1>
          
          <div className="flex items-center gap-3">
            {isHealthChecking ? (
              <span className={`text-sm ${textMuted}`}>Checking backend...</span>
            ) : health ? (
              <div className="flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-green-500" />
                <span className="text-sm text-green-500">Connected</span>
                <span className={`text-xs ${textMuted}`}>| {health.unet_device.toUpperCase()}</span>
              </div>
            ) : (
              <div className="flex items-center gap-2">
                <XCircle className="w-4 h-4 text-red-500" />
                <span className="text-sm text-red-500">Disconnected</span>
              </div>
            )}
            
            <button
              onClick={refreshHealth}
              disabled={isHealthChecking}
              className={`p-1.5 rounded-lg transition-colors disabled:opacity-50 ${
                isDark ? 'bg-slate-800 hover:bg-slate-700' : 'bg-gray-100 hover:bg-gray-200'
              }`}
            >
              <RefreshCw className={`w-4 h-4 ${isHealthChecking ? 'animate-spin' : ''} ${textMuted}`} />
            </button>
            
            <button
              onClick={toggleTheme}
              className={`p-2 rounded-lg transition-colors ${
                isDark ? 'bg-slate-800 hover:bg-slate-700 text-blue-400' : 'bg-gray-100 hover:bg-gray-200 text-gray-600'
              }`}
            >
              {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1 space-y-4">
            <UploadZone onFileSelect={handleFileSelect} isLoading={isAnalyzing} />
            {(isAnalyzing || result) && <PipelineStatus steps={pipelineSteps} />}
            {error && (
              <div className={`p-4 rounded-lg border ${isDark ? 'border-red-900 bg-red-950/50' : 'border-red-200 bg-red-50'}`}>
                <p className="text-sm text-red-500">{error}</p>
              </div>
            )}
          </div>

          <div className="lg:col-span-2 space-y-4">
            {result ? (
              <>
                <div className={cardClass}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Cpu className={`w-5 h-5 ${textMuted}`} />
                      <span className={`text-sm font-medium ${textPrimary}`}>Classification</span>
                    </div>
                    <span className={`text-xs font-mono ${textMuted}`}>Job: {result.job_id}</span>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <PlotlyClassificationBadge predictedClass={result.predicted_class} confidence={result.confidence} />
                    <PlotlyProbabilityChart probabilities={result.probabilities} predictedClass={result.predicted_class} />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className={cardClass}>
                    <PlotlyConfidenceGauge confidence={result.confidence} predictedClass={result.predicted_class} />
                  </div>
                  {result.image_info && (
                    <div className={cardClass}>
                      <div className="flex items-center gap-2 mb-2">
                        <Activity className={`w-5 h-5 ${textMuted}`} />
                        <span className={`text-sm font-medium ${textPrimary}`}>Image Information</span>
                      </div>
                      <PlotlyStatsIndicators stats={[
                        { label: 'Width', value: result.image_info.original_width },
                        { label: 'Height', value: result.image_info.original_height },
                        { label: 'U-Net Size', value: result.image_info.unet_input_size },
                      ]} />
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <ImageViewer src={result.original_image} alt="Original AFM image" title="Original Image" />
                  <ImageViewer src={result.mask_image} alt="U-Net segmentation mask" title="U-Net Mask" />
                </div>

                {result.dot_extraction_stats && (
                  <div className={cardClass}>
                    <div className="flex items-center gap-2 mb-3">
                      <Circle className={`w-5 h-5 ${textMuted}`} />
                      <span className={`text-sm font-medium ${textPrimary}`}>Dot Extraction</span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <PlotlyDotExtractionPie kept={result.dot_extraction_stats.kept_features} rejected={result.dot_extraction_stats.rejected_features} />
                      <PlotlyStatsIndicators stats={[
                        { label: 'Total', value: result.dot_extraction_stats.total_features, color: isDark ? '#94a3b8' : '#374151' },
                        { label: 'Kept', value: result.dot_extraction_stats.kept_features, color: '#22c55e' },
                        { label: 'Rejected', value: result.dot_extraction_stats.rejected_features, color: '#f97316' },
                      ]} />
                    </div>
                  </div>
                )}

                {result.extra_outputs.length > 0 && (
                  <div className="grid grid-cols-2 gap-4">
                    {result.extra_outputs.map((output, idx) => (
                      <ImageViewer key={idx} src={output.image} alt={output.title} title={output.title} subtitle={output.description} />
                    ))}
                  </div>
                )}

                {Object.keys(result.analysis_details).length > 0 && (
                  <div className={cardClass}>
                    <div className="flex items-center gap-2 mb-3">
                      <BarChart3 className={`w-5 h-5 ${textMuted}`} />
                      <span className={`text-sm font-medium ${textPrimary}`}>
                        {result.predicted_class === 'lines' ? 'Color Wheel' : 'Voronoi'} Analysis
                      </span>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <PlotlyMetricsRadar metrics={result.analysis_details} title="Metrics Overview" />
                      <PlotlyDetailsBar details={result.analysis_details} />
                    </div>
                    <div className="mt-4 overflow-auto max-h-48">
                      <table className="w-full text-sm">
                        <tbody>
                          {Object.entries(result.analysis_details).map(([key, value]) => (
                            <tr key={key} className={`border-b ${isDark ? 'border-slate-700' : 'border-gray-100'}`}>
                              <td className={`py-2 pr-4 font-medium ${textMuted}`}>{key}</td>
                              <td className={`py-2 font-mono text-right ${textPrimary}`}>
                                {typeof value === 'number' ? value.toFixed(6) : String(value)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}

                {result.colorwheel_stats && (
                  <div className={cardClass}>
                    <div className="flex items-center gap-2 mb-3">
                      <Palette className={`w-5 h-5 ${textMuted}`} />
                      <span className={`text-sm font-medium ${textPrimary}`}>Color Wheel Stats</span>
                    </div>
                    <PlotlyStatsIndicators stats={[
                      { label: 'Orientation', value: Number(result.colorwheel_stats.orientation_angle.toFixed(2)) },
                      { label: 'Grain Masks', value: result.colorwheel_stats.grain_masks_count },
                    ]} />
                  </div>
                )}

                <details className={`${cardClass} overflow-hidden`}>
                  <summary className={`cursor-pointer hover:opacity-80 text-sm font-medium ${textPrimary}`}>
                    Raw API Response (Debug)
                  </summary>
                  <pre className={`mt-3 rounded-lg p-3 text-xs overflow-auto max-h-96 font-mono ${isDark ? 'bg-slate-900 text-slate-300' : 'bg-gray-50 text-gray-700'}`}>
                    {JSON.stringify({ job_id: result.job_id, predicted_class: result.predicted_class, confidence: result.confidence, probabilities: result.probabilities, image_info: result.image_info, dot_extraction_stats: result.dot_extraction_stats, voronoi_stats: result.voronoi_stats, colorwheel_stats: result.colorwheel_stats, analysis_details: result.analysis_details, extra_outputs_count: result.extra_outputs.length }, null, 2)}
                  </pre>
                </details>
              </>
            ) : (
              <div className={`${cardClass} p-12 text-center`}>
                <div className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center mb-4 ${isDark ? 'bg-slate-700' : 'bg-gray-100'}`}>
                  <BarChart3 className={`w-8 h-8 ${textMuted}`} />
                </div>
                <h3 className={`text-lg font-medium mb-2 ${textPrimary}`}>Upload an Image</h3>
                <p className={`text-sm max-w-md mx-auto ${textMuted}`}>Drop an AFM image to run the full analysis pipeline.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}
