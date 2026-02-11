'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Image as ImageIcon } from 'lucide-react';
import clsx from 'clsx';

interface UploadZoneProps {
  onFileSelect: (file: File) => void;
  isLoading?: boolean;
  acceptedTypes?: string[];
}

export function UploadZone({ 
  onFileSelect, 
  isLoading = false,
  acceptedTypes = ['image/jpeg', 'image/png', 'image/tiff']
}: UploadZoneProps) {
  const [preview, setPreview] = useState<string | null>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      onFileSelect(file);
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(file);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.jpg', '.jpeg', '.png', '.tif', '.tiff']
    },
    maxFiles: 1,
    disabled: isLoading
  });

  return (
    <div
      {...getRootProps()}
      className={clsx(
        'upload-zone relative cursor-pointer p-8 text-center transition-all',
        isDragActive && 'active',
        isLoading && 'opacity-50 cursor-not-allowed'
      )}
    >
      <input {...getInputProps()} />
      
      {preview && !isLoading ? (
        <div className="space-y-4">
          <div className="mx-auto w-48 h-48 rounded-lg overflow-hidden bg-gray-100 border border-gray-200">
            <img 
              src={preview} 
              alt="Preview" 
              className="w-full h-full object-contain"
            />
          </div>
          <p className="text-sm text-gray-600">
            Click or drag to replace
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          <div className={clsx(
            'mx-auto w-16 h-16 rounded-full flex items-center justify-center',
            'bg-blue-50',
            'border border-blue-200'
          )}>
            {isLoading ? (
              <div className="w-6 h-6 border-2 border-blue-600 border-t-transparent rounded-full animate-spin" />
            ) : (
              <Upload className="w-7 h-7 text-blue-600" />
            )}
          </div>
          
          <div>
            <p className="text-lg font-medium text-gray-900">
              {isDragActive ? 'Drop image here' : 'Upload AFM Image'}
            </p>
            <p className="mt-1 text-sm text-gray-600">
              Drag & drop or click to browse
            </p>
            <p className="mt-2 text-xs text-gray-400">
              Supports JPG, PNG, TIFF
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
