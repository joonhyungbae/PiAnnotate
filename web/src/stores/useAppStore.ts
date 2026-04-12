import { create } from 'zustand';
import type { AppSettings, FrameData, FingeringItem } from '../types';

interface AppState {
  // Current piece
  pieceId: number | null;
  setPieceId: (id: number | null) => void;

  // Loading state
  loadedFrames: number;
  totalFrames: number;
  setLoadedFrames: (frames: number) => void;
  setTotalFrames: (frames: number) => void;

  // Playback state
  currentFrame: number;
  isPlaying: boolean;
  setCurrentFrame: (frame: number) => void;
  setIsPlaying: (playing: boolean) => void;

  // Current frame data
  currentFrameData: FrameData | null;
  setCurrentFrameData: (data: FrameData | null) => void;

  // Current fingering
  currentFingering: FingeringItem[];
  setCurrentFingering: (fingering: FingeringItem[]) => void;
  
  // AI annotation flag (r0 = AI prior, editable by annotator)
  isAiAnnotation: boolean;
  setIsAiAnnotation: (isAi: boolean) => void;

  // Settings
  settings: AppSettings;
  updateSettings: (settings: Partial<AppSettings>) => void;

  // Metadata
  pieceName: string;
  pieceComposer: string;
  setPieceInfo: (name: string, composer: string) => void;
}

export const useAppStore = create<AppState>((set) => ({
  // Current piece
  pieceId: null,
  setPieceId: (id) => set({ pieceId: id }),

  // Loading state
  loadedFrames: 0,
  totalFrames: 0,
  setLoadedFrames: (frames) => set({ loadedFrames: frames }),
  setTotalFrames: (frames) => set({ totalFrames: frames }),

  // Playback state
  currentFrame: 0,
  isPlaying: false,
  setCurrentFrame: (frame) => set({ currentFrame: frame }),
  setIsPlaying: (playing) => set({ isPlaying: playing }),

  // Current frame data
  currentFrameData: null,
  setCurrentFrameData: (data) => set({ currentFrameData: data }),

  // Current fingering
  currentFingering: [],
  setCurrentFingering: (fingering) => set({ currentFingering: fingering }),
  
  // AI annotation flag (r0 = AI prior, editable by annotator)
  isAiAnnotation: false,
  setIsAiAnnotation: (isAi) => set({ isAiAnnotation: isAi }),

  // Settings
  settings: {
    showFingering: true,
    showHittingPoints: true,
    cameraPreset: 'top',
    handColor: 0xefceb9,
  },
  updateSettings: (newSettings) =>
    set((state) => ({
      settings: { ...state.settings, ...newSettings },
    })),

  // Metadata
  pieceName: '',
  pieceComposer: '',
  setPieceInfo: (name, composer) => set({ pieceName: name, pieceComposer: composer }),
}));
