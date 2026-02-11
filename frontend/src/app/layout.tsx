import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'AFM Analysis Pipeline',
  description: 'Advanced AFM image analysis with CNN classification, U-Net segmentation, Voronoi and Color Wheel analysis',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0f1419] text-[#e8edf4]">
        <div className="grid-pattern min-h-screen">
          {children}
        </div>
      </body>
    </html>
  );
}
