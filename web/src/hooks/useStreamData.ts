import { getMeshDataUrl, getFingeringDataUrl } from '../api';
import type { FrameData, FingeringItem } from '../types';
import { useAppStore } from '../stores/useAppStore';

// Module level singleton for fingering data
const fingeringData = new Map<number, FingeringItem[]>();
const originalFingeringData = new Map<number, FingeringItem[]>();
let isAiAnnotationData = false;

interface StreamDataOptions {
  onFrame?: (frameIndex: number, frameData: FrameData) => Promise<void>;
  onProgress?: (loaded: number, total: number) => void;
}

/**
 * Load mesh data (gzip-compressed bulk transfer)
 */
export async function streamMeshData(
  pieceId: number,
  _totalFrames: number,
  options: StreamDataOptions,
  setLoadedFrames?: (frames: number) => void
) {
  const url = getMeshDataUrl(pieceId);
  const startTime = Date.now();
  
  console.log(`Loading mesh data for piece ${pieceId}...`);
  
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`HTTP error! Status: ${response.status}`);
  }

  // Browser automatically decompresses gzip
  const frames: FrameData[] = await response.json();
  
  const fetchTime = Date.now() - startTime;
  console.log(`Fetched ${frames.length} frames in ${(fetchTime / 1000).toFixed(1)}s`);

  // Process frames
  for (let i = 0; i < frames.length; i++) {
    const frameData = frames[i];

        if (options.onFrame) {
      await options.onFrame(i, frameData);
        }

        if (setLoadedFrames) {
      setLoadedFrames(i + 1);
        }

        if (options.onProgress) {
      options.onProgress(i + 1, frames.length);
    }
  }

  const totalTime = (Date.now() - startTime) / 1000;
  console.log(`Processed ${frames.length} frames in ${totalTime.toFixed(1)}s`);

  return frames.length;
}

interface FingeringFrameData {
  frame_idx: number;
  fingering: FingeringItem[];
  original_fingering?: FingeringItem[];
  is_ai_annotation?: boolean;
}

/**
 * Load fingering data (gzip-compressed bulk transfer)
 */
export async function streamFingeringData(pieceId: number) {
  const url = getFingeringDataUrl(pieceId);
  const startTime = Date.now();
  
  console.log(`Loading fingering data for piece ${pieceId}...`);
  
  const response = await fetch(url);
  if (!response.ok) {
    console.warn(`Fingering data not available: ${response.status}`);
    return;
  }

  // Browser automatically decompresses gzip
  const frames: FingeringFrameData[] = await response.json();

  const fetchTime = Date.now() - startTime;
  console.log(`Fetched fingering data in ${(fetchTime / 1000).toFixed(1)}s`);

  // Clear existing data
  fingeringData.clear();
  originalFingeringData.clear();

  // Check if this is AI annotation (r0) - AI prior, editable by annotator
  if (frames.length > 0 && frames[0].is_ai_annotation) {
    isAiAnnotationData = true;
    console.log("🤖 AI Annotation (r0) detected - AI prior available for editing");
    // Update store
    useAppStore.getState().setIsAiAnnotation(true);
  } else {
    isAiAnnotationData = false;
    useAppStore.getState().setIsAiAnnotation(false);
  }

  // Process frames
  for (const frame of frames) {
    if (frame.frame_idx !== undefined && frame.fingering) {
      fingeringData.set(frame.frame_idx, frame.fingering);
      
      // Store original if edited version exists
      if (frame.original_fingering) {
        originalFingeringData.set(frame.frame_idx, frame.original_fingering);
      }
    }
  }

  console.log(`Loaded fingering data for ${fingeringData.size} frames`);
}

export function getFingering(frameIndex: number): FingeringItem[] {
  return fingeringData.get(frameIndex) || [];
}

export function getOriginalFingering(frameIndex: number): FingeringItem[] | undefined {
  return originalFingeringData.get(frameIndex);
}

export function setFingering(frameIndex: number, fingering: FingeringItem[]) {
  fingeringData.set(frameIndex, fingering);
}

export function clearFingeringData() {
  fingeringData.clear();
  originalFingeringData.clear();
  isAiAnnotationData = false;
  useAppStore.getState().setIsAiAnnotation(false);
}

export function isAiAnnotation(): boolean {
  return isAiAnnotationData;
}
