'use client';

import { Info, ArrowRight, Scale } from 'lucide-react';
import type { ImageInfo } from '@/lib/api';

interface ImageInfoCardProps {
  imageInfo: ImageInfo;
}

export function ImageInfoCard({ imageInfo }: ImageInfoCardProps) {
  const {
    original_width,
    original_height,
    cnn_input_size,
    unet_input_size,
    will_resize_for_unet,
    will_resize_for_cnn,
  } = imageInfo;

  const needsResize = will_resize_for_unet || will_resize_for_cnn;

  return (
    <div className="card p-4">
      <div className="flex items-center gap-2 mb-4">
        <Scale className="w-4 h-4 text-[#6b7b8f]" />
        <h3 className="text-sm font-medium text-[#e8edf4]">Image Processing Info</h3>
      </div>

      <div className="space-y-3">
        {/* Original Size */}
        <div className="flex items-center justify-between py-2 px-3 rounded-lg bg-[#0f1419]/50">
          <span className="text-sm text-[#9ba9bc]">Original Size</span>
          <span className="text-sm font-mono text-[#e8edf4]">
            {original_width} × {original_height} px
          </span>
        </div>

        {/* Resize Info */}
        {needsResize && (
          <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
            <div className="flex items-start gap-2">
              <Info className="w-4 h-4 text-amber-400 mt-0.5 flex-shrink-0" />
              <div className="text-sm">
                <p className="text-amber-300 font-medium mb-2">Image will be resized for processing</p>
                <div className="space-y-1.5 text-[#9ba9bc]">
                  {will_resize_for_unet && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs">U-Net:</span>
                      <span className="font-mono text-xs">
                        {original_width}×{original_height}
                      </span>
                      <ArrowRight className="w-3 h-3" />
                      <span className="font-mono text-xs text-cyan-400">
                        {unet_input_size}×{unet_input_size}
                      </span>
                    </div>
                  )}
                  {will_resize_for_cnn && (
                    <div className="flex items-center gap-2">
                      <span className="text-xs">CNN:</span>
                      <span className="font-mono text-xs">
                        {original_width}×{original_height}
                      </span>
                      <ArrowRight className="w-3 h-3" />
                      <span className="font-mono text-xs text-purple-400">
                        {cnn_input_size}×{cnn_input_size}
                      </span>
                    </div>
                  )}
                </div>
                <p className="text-xs text-[#6b7b8f] mt-2">
                  Output mask is scaled back to original dimensions
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Model Input Sizes */}
        <div className="grid grid-cols-2 gap-2">
          <div className="py-2 px-3 rounded-lg bg-[#0f1419]/50">
            <p className="text-xs text-[#6b7b8f] mb-1">U-Net Input</p>
            <p className="text-sm font-mono text-cyan-400">{unet_input_size}×{unet_input_size}</p>
          </div>
          <div className="py-2 px-3 rounded-lg bg-[#0f1419]/50">
            <p className="text-xs text-[#6b7b8f] mb-1">CNN Input</p>
            <p className="text-sm font-mono text-purple-400">{cnn_input_size}×{cnn_input_size}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
