'use client';

import { useState } from 'react';
import { ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';
import clsx from 'clsx';

interface ImageViewerProps {
  src: string;
  alt: string;
  title: string;
  subtitle?: string;
}

export function ImageViewer({ src, alt, title, subtitle }: ImageViewerProps) {
  const [isFullscreen, setIsFullscreen] = useState(false);

  if (!src) {
    return (
      <div className="bg-white border border-gray-200 rounded-xl p-4">
        <h3 className="text-sm font-medium text-gray-700 mb-3">{title}</h3>
        <div className="aspect-square bg-gray-100 rounded-lg flex items-center justify-center">
          <p className="text-gray-400 text-sm">No image</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="bg-white border border-gray-200 rounded-xl p-4">
        <div className="flex items-center justify-between mb-3">
          <div>
            <h3 className="text-sm font-medium text-gray-900">{title}</h3>
            {subtitle && (
              <p className="text-xs text-gray-500 mt-0.5">{subtitle}</p>
            )}
          </div>
          <button
            onClick={() => setIsFullscreen(true)}
            className="p-1.5 rounded-lg bg-gray-100 hover:bg-gray-200 transition-colors"
            title="View fullscreen"
          >
            <Maximize2 className="w-4 h-4 text-gray-600" />
          </button>
        </div>
        
        <div className="image-container aspect-square">
          <img src={src} alt={alt} className="w-full h-full object-contain" />
        </div>
      </div>

      {/* Fullscreen Modal */}
      {isFullscreen && (
        <div 
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-8"
          onClick={() => setIsFullscreen(false)}
        >
          <div className="relative max-w-[90vw] max-h-[90vh]">
            <img 
              src={src} 
              alt={alt} 
              className="max-w-full max-h-[90vh] object-contain rounded-lg"
            />
            <button
              onClick={() => setIsFullscreen(false)}
              className="absolute top-4 right-4 p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
            >
              <span className="sr-only">Close</span>
              <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
            <div className="absolute bottom-4 left-4 text-white">
              <h3 className="font-medium">{title}</h3>
              {subtitle && <p className="text-sm text-white/70">{subtitle}</p>}
            </div>
          </div>
        </div>
      )}
    </>
  );
}
