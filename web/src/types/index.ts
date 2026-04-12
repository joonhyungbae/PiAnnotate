// Split type
export type SplitType = 'train' | 'valid' | 'test' | 'unknown';

// Piece metadata
export interface PieceMetadata {
  piece_id: number;
  name: string;
  composer: string;
  split?: SplitType;
  period?: string;
  difficulty?: number;
  dataset?: string;
  num_frames?: number;
  annotator?: number;  // 1-6 or undefined (unassigned)
}

// Detailed metadata
export interface PieceDetailMetadata {
  name: string;
  composer: string;
  num_frames: number;
}

// MANO faces data
export interface ManoFacesData {
  left_faces: number[][];
  right_faces: number[][];
}

// Frame data
export interface FrameData {
  left_vertices: string[][];
  right_vertices: string[][];
  left_joints: string[];
  right_joints: string[];
  pressed_keys: number[];
}

// Fingering data
export interface FingeringItem {
  hand: 'left' | 'right';
  finger: number; // 1-5
  finger_name: 'thumb' | 'index' | 'middle' | 'ring' | 'pinky';
  key_idx?: number;
  ambiguous?: boolean; // true if the fingering is ambiguous/uncertain
  was_corrected?: boolean; // true if AI corrected this fingering (r0 only)
  correction_prob?: number; // AI correction confidence (0-1)
}

export interface FingeringFrameData {
  frame_idx: number;
  fingering: FingeringItem[];
  original_fingering?: FingeringItem[];
  is_ai_annotation?: boolean;  // r0 (AI annotation) - AI prior, editable by annotator
}

// Hitting points data
export interface HittingPointsData {
  hitting_points: number[][];
  keyboard_bounds?: {
    min: number[];
    max: number[];
    center: number[];
    size: number[];
  };
  key_features?: {
    key_idx: number;
    midi_note: number;
    pitch_name: string;
    is_black_key: boolean;
    hitting_point: number[];
  }[];
}

// Camera preset
export interface CameraPreset {
  position: { x: number; y: number; z: number };
  target: { x: number; y: number; z: number };
}

// App settings
export interface AppSettings {
  showFingering: boolean;
  showHittingPoints: boolean;
  cameraPreset: 'top' | 'front' | 'back' | 'left' | 'right';
  handColor: number;
}

// Annotation status (2-stage review system)
// Stage 1: Annotator review
// Stage 2: Professional pianist review
export interface AnnotationStatusDetails {
  completed_at: string;
  notes: string;
}

export interface AnnotationStatus {
  review1: Record<string, AnnotationStatusDetails>;
  review2: Record<string, AnnotationStatusDetails>;
  review3: Record<string, AnnotationStatusDetails>;
  fingering_completed: Record<string, AnnotationStatusDetails>;
  post_completed: Record<string, AnnotationStatusDetails>;
}

export interface PieceAnnotationStatus {
  piece_id: number;
  review1_completed: boolean;
  review1_details: AnnotationStatusDetails | null;
  review2_completed: boolean;
  review2_details: AnnotationStatusDetails | null;
  review3_completed: boolean;
  review3_details: AnnotationStatusDetails | null;
  fingering_completed: boolean;
  fingering_details: AnnotationStatusDetails | null;
  post_completed: boolean;
  post_details: AnnotationStatusDetails | null;
}

// Motion issues (problematic motion segments)
export interface MotionIssue {
  id: string;
  start_time: number;  // seconds
  end_time: number;    // seconds
  note: string;
  created_at: string;
}

// Post-playing segment (after piano playing ends)
export interface PostPlayingSegment {
  start_time: number;  // seconds
  end_time: number;    // seconds
}

// Test segment (for user evaluation, TEST dataset only)
export interface TestSegment {
  id: string;
  start_time: number;  // seconds
  end_time: number;    // seconds
  note: string;
  created_at: string;
}
