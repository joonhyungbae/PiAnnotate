import type {
  PieceMetadata,
  PieceDetailMetadata,
  ManoFacesData,
  HittingPointsData,
  AnnotationStatus,
  PieceAnnotationStatus,
  MotionIssue,
  PostPlayingSegment,
  TestSegment,
} from '../types';

const BASE_URL = '';

export async function fetchPiecesMetadata(): Promise<PieceMetadata[]> {
  const response = await fetch(`${BASE_URL}/pieces_metadata`);
  if (!response.ok) {
    throw new Error(`Failed to fetch pieces metadata: ${response.status}`);
  }
  const data = await response.json();
  return Array.isArray(data) ? data : [];
}

export async function fetchPieceMetadata(pieceId: number): Promise<PieceDetailMetadata> {
  const response = await fetch(`${BASE_URL}/metadata/${pieceId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch piece metadata: ${response.status}`);
  }
  return response.json();
}

export async function fetchManoFacesData(): Promise<ManoFacesData> {
  const response = await fetch(`${BASE_URL}/mano_faces_data`);
  if (!response.ok) {
    throw new Error(`Failed to fetch MANO faces data: ${response.status}`);
  }
  return response.json();
}

export async function fetchHittingPoints(): Promise<HittingPointsData | null> {
  const response = await fetch(`${BASE_URL}/hitting_points`);
  if (!response.ok) {
    console.warn(`Hitting points not available: ${response.status}`);
    return null;
  }
  return response.json();
}

export function getAudioUrl(pieceId: number): string {
  return `${BASE_URL}/audio/${pieceId}`;
}

export function getPianoMeshUrl(keyIndex: number, ext: 'obj' | 'mtl'): string {
  return `${BASE_URL}/piano_mesh/${keyIndex}.${ext}`;
}

export function getResourceUrl(filename: string): string {
  return `${BASE_URL}/resources/${filename}`;
}

// Streaming data URL
export function getMeshDataUrl(pieceId: number): string {
  return `${BASE_URL}/mano_vertices_data/${pieceId}`;
}

export function getFingeringDataUrl(pieceId: number): string {
  return `${BASE_URL}/fingering_data/${pieceId}`;
}

// Annotation status API
export async function fetchAnnotationStatus(): Promise<AnnotationStatus> {
  const response = await fetch(`${BASE_URL}/annotation_status`);
  if (!response.ok) {
    throw new Error(`Failed to fetch annotation status: ${response.status}`);
  }
  return response.json();
}

// Annotation sources API (r0: AI, r1: human, original: rule-based)
export type AnnotationSource = 'r0' | 'r1' | 'original';
export type AnnotationSources = Record<string, AnnotationSource>;

export async function fetchAnnotationSources(): Promise<AnnotationSources> {
  const response = await fetch(`${BASE_URL}/annotation_sources`);
  if (!response.ok) {
    throw new Error(`Failed to fetch annotation sources: ${response.status}`);
  }
  return response.json();
}

export async function fetchPieceAnnotationStatus(pieceId: number): Promise<PieceAnnotationStatus> {
  const response = await fetch(`${BASE_URL}/annotation_status/${pieceId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch piece annotation status: ${response.status}`);
  }
  return response.json();
}

export async function updateAnnotationStatus(
  pieceId: number, 
  stage: 1 | 2 | 3,
  completed: boolean, 
  notes?: string
): Promise<{ success: boolean; piece_id: number; stage: number; completed: boolean; auto_review_completed: number | null }> {
  const response = await fetch(`${BASE_URL}/annotation_status/${pieceId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ stage, completed, notes: notes || '' }),
  });
  if (!response.ok) {
    throw new Error(`Failed to update annotation status: ${response.status}`);
  }
  return response.json();
}

// Update fingering/post completion status
export async function updateSubtaskStatus(
  pieceId: number,
  subtask: 'fingering' | 'post',
  completed: boolean
): Promise<{ success: boolean; piece_id: number; subtask: string; completed: boolean }> {
  const response = await fetch(`${BASE_URL}/annotation_status/${pieceId}/subtask`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ subtask, completed }),
  });
  if (!response.ok) {
    throw new Error(`Failed to update subtask status: ${response.status}`);
  }
  return response.json();
}

// Motion Issues API
export async function fetchMotionIssues(pieceId: number): Promise<MotionIssue[]> {
  const response = await fetch(`${BASE_URL}/motion_issues/${pieceId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch motion issues: ${response.status}`);
  }
  return response.json();
}

export async function addMotionIssue(
  pieceId: number,
  startTime: number,
  endTime: number,
  note?: string
): Promise<{ success: boolean; issue: MotionIssue }> {
  const response = await fetch(`${BASE_URL}/motion_issues/${pieceId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      start_time: startTime,
      end_time: endTime,
      note: note || '',
    }),
  });
  if (!response.ok) {
    throw new Error(`Failed to add motion issue: ${response.status}`);
  }
  return response.json();
}

export async function deleteMotionIssue(
  pieceId: number,
  issueId: string
): Promise<{ success: boolean }> {
  const response = await fetch(`${BASE_URL}/motion_issues/${pieceId}/${issueId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(`Failed to delete motion issue: ${response.status}`);
  }
  return response.json();
}

// Post-Playing Segment API
export async function fetchPostPlaying(pieceId: number): Promise<PostPlayingSegment | null> {
  const response = await fetch(`${BASE_URL}/post_playing/${pieceId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch post-playing segment: ${response.status}`);
  }
  return response.json();
}

export async function savePostPlaying(
  pieceId: number,
  startTime: number,
  endTime: number
): Promise<{ success: boolean; segment: PostPlayingSegment }> {
  const response = await fetch(`${BASE_URL}/post_playing/${pieceId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      start_time: startTime,
      end_time: endTime,
    }),
  });
  if (!response.ok) {
    throw new Error(`Failed to save post-playing segment: ${response.status}`);
  }
  return response.json();
}

export async function deletePostPlaying(pieceId: number): Promise<{ success: boolean }> {
  const response = await fetch(`${BASE_URL}/post_playing/${pieceId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(`Failed to delete post-playing segment: ${response.status}`);
  }
  return response.json();
}

// Test Segments API (for user evaluation, TEST dataset only)
export async function fetchTestSegments(pieceId: number): Promise<TestSegment[]> {
  const response = await fetch(`${BASE_URL}/test_segments/${pieceId}`);
  if (!response.ok) {
    // 404 means no segments exist yet, return empty array
    if (response.status === 404) {
      return [];
    }
    throw new Error(`Failed to fetch test segments: ${response.status}`);
  }
  
  // Check if response is JSON
  const contentType = response.headers.get('content-type');
  if (!contentType || !contentType.includes('application/json')) {
    // Silently return empty array if response is not JSON (e.g., HTML error page)
    return [];
  }
  
  try {
    return await response.json();
  } catch (err) {
    // Silently return empty array on parse error
    return [];
  }
}

export async function addTestSegment(
  pieceId: number,
  startTime: number,
  endTime: number,
  note?: string
): Promise<{ success: boolean; segment: TestSegment }> {
  const response = await fetch(`${BASE_URL}/test_segments/${pieceId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      start_time: startTime,
      end_time: endTime,
      note: note || '',
    }),
  });
  if (!response.ok) {
    throw new Error(`Failed to add test segment: ${response.status}`);
  }
  return response.json();
}

export async function deleteTestSegment(
  pieceId: number,
  segmentId: string
): Promise<{ success: boolean }> {
  const response = await fetch(`${BASE_URL}/test_segments/${pieceId}/${segmentId}`, {
    method: 'DELETE',
  });
  if (!response.ok) {
    throw new Error(`Failed to delete test segment: ${response.status}`);
  }
  return response.json();
}

// Annotation Progress API (track where user left off)
export interface AnnotationProgress {
  last_frame: number;
  last_time_seconds: number;
  updated_at: string;
}

export async function fetchAnnotationProgress(pieceId: number): Promise<AnnotationProgress | null> {
  const response = await fetch(`${BASE_URL}/annotation_progress/${pieceId}`);
  if (!response.ok) {
    if (response.status === 404) {
      return null;
    }
    throw new Error(`Failed to fetch annotation progress: ${response.status}`);
  }
  return response.json();
}

export async function saveAnnotationProgress(
  pieceId: number,
  lastFrame: number,
  lastTimeSeconds: number
): Promise<{ success: boolean; progress: AnnotationProgress }> {
  const response = await fetch(`${BASE_URL}/annotation_progress/${pieceId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      last_frame: lastFrame,
      last_time_seconds: lastTimeSeconds,
    }),
  });
  if (!response.ok) {
    throw new Error(`Failed to save annotation progress: ${response.status}`);
  }
  return response.json();
}

// Fingering Data API
export interface FingeringEntry {
  finger: number;  // 1-10 (1-5: left, 6-10: right)
  key_id: number;  // 0-87 (normalized from key_index)
  hand: 'left' | 'right';
  ambiguous?: boolean;
  was_corrected?: boolean;  // AI가 수정한 핑거링
  is_missing?: boolean;     // 누락된 노트에 추가된 핑거링
}

export interface FingeringFrame {
  frame_idx: number;
  fingering: FingeringEntry[];
  original_fingering?: FingeringEntry[];
}

// Raw API response types
interface RawFingeringEntry {
  finger: number;
  key_index?: number;
  key_id?: number;
  hand: string;
  ambiguous?: boolean;
  was_corrected?: boolean;
  is_missing?: boolean;
}

interface RawFingeringFrame {
  frame_idx: number;
  fingering: RawFingeringEntry[];
  original_fingering?: RawFingeringEntry[];
}

// Normalize entry to use key_id consistently
function normalizeEntry(entry: RawFingeringEntry): FingeringEntry {
  return {
    finger: entry.finger,
    key_id: entry.key_index ?? entry.key_id ?? 0,
    hand: entry.hand as 'left' | 'right',
    ambiguous: entry.ambiguous,
    was_corrected: entry.was_corrected,
    is_missing: entry.is_missing,
  };
}

export async function fetchFingeringData(pieceId: number): Promise<FingeringFrame[]> {
  const response = await fetch(`${BASE_URL}/fingering_data/${pieceId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch fingering data: ${response.status}`);
  }
  const rawData: RawFingeringFrame[] = await response.json();
  
  // Normalize key_index to key_id
  return rawData.map(frame => ({
    frame_idx: frame.frame_idx,
    fingering: (frame.fingering || []).map(normalizeEntry),
    original_fingering: frame.original_fingering 
      ? frame.original_fingering.map(normalizeEntry)
      : undefined,
  }));
}

