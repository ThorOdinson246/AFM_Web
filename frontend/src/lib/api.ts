// API types for AFM Pipeline

export interface ImageInfo {
  original_width: number;
  original_height: number;
  cnn_input_size: number;
  unet_input_size: number;
  will_resize_for_unet: boolean;
  will_resize_for_cnn: boolean;
}

export interface DotExtractionStats {
  output_path: string;
  total_features: number;
  kept_features: number;
  rejected_features: number;
}

export interface AnalysisResult {
  success: boolean;
  job_id: string;
  original_image: string;
  mask_image: string;
  dots_mask_image?: string | null;
  predicted_class: 'dots' | 'lines' | 'mixed' | 'irregular';
  confidence: number;
  probabilities: Record<string, number>;
  extra_outputs: ExtraOutput[];
  analysis_details: Record<string, string | number>;
  voronoi_stats?: Record<string, string> | null;
  colorwheel_stats?: ColorWheelStats | null;
  dot_extraction_stats?: DotExtractionStats | null;
  image_info?: ImageInfo | null;
}

export interface ExtraOutput {
  image: string;
  title: string;
  description: string;
}

export interface ColorWheelStats {
  orientation_angle: number;
  gpu_accelerated: boolean;
  grain_masks_count: number;
}

export interface HealthStatus {
  status: string;
  models_loaded: boolean;
  cnn_device: string;
  unet_device: string;
}

export interface Job {
  job_id: string;
  created: string;
}

export interface JobListResponse {
  jobs: Job[];
}

// API client
const API_BASE = '/api';

export async function checkHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE}/`);
  if (!response.ok) {
    throw new Error('Backend is not available');
  }
  return response.json();
}

export async function analyzeImage(file: File): Promise<AnalysisResult> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Analysis failed' }));
    throw new Error(error.detail || 'Analysis failed');
  }

  return response.json();
}

export async function getJobs(): Promise<JobListResponse> {
  const response = await fetch(`${API_BASE}/jobs`);
  if (!response.ok) {
    throw new Error('Failed to fetch jobs');
  }
  return response.json();
}
