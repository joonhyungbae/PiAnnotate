import { useEffect, useRef, useCallback, useState } from 'react';
import type { FingeringFrame, FingeringEntry } from '../../api';

interface PianoRollProps {
  fingeringData: FingeringFrame[];
  currentFrame: number;
  totalFrames: number;
  zoom: number;
  scrollX: number;
  onScrollChange: (x: number) => void;
  onSeek: (frame: number) => void;
  isPlaying?: boolean; // 재생 중일 때 playhead 고정 위치에 그리기
}

// Piano key configuration
const TOTAL_KEYS = 88;
const HEADER_WIDTH = 40;
const FPS = 60;
const MIN_KEY_HEIGHT = 4;

// Colors for comparison view
const COLORS = {
  unchanged: '#4CAF50',      // Green - same as original
  changed: '#F44336',        // Red - modified fingering
  missing: '#FFCC00',        // Yellow - missing note added
  originalGhost: '#666666',  // Gray - original (when different)
  background: '#1a1a2e',
  grid: '#2d2d44',
  gridMajor: '#3d3d5c',
  playhead: '#FFEB3B',
};

// Check if key is black
function isBlackKey(keyId: number): boolean {
  const noteInOctave = keyId % 12;
  return [1, 3, 6, 8, 10].includes(noteInOctave);
}

// Get note name
function getNoteName(keyId: number): string {
  const noteNames = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#'];
  const noteInOctave = keyId % 12;
  const octave = Math.floor((keyId + 9) / 12);
  return `${noteNames[noteInOctave]}${octave}`;
}

export function PianoRoll({
  fingeringData,
  currentFrame,
  totalFrames,
  zoom,
  scrollX,
  onScrollChange,
  onSeek,
  isPlaying = false,
}: PianoRollProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoveredInfo, setHoveredInfo] = useState<{
    frame: number;
    key: number;
    finger: number;
    originalFinger: number | null;
    note: string;
    status: 'unchanged' | 'changed' | 'added' | 'removed' | 'missing';
  } | null>(null);

  const [canvasWidth, setCanvasWidth] = useState(800);
  const [canvasHeight, setCanvasHeight] = useState(400);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartScrollX, setDragStartScrollX] = useState(0);

  // Calculate dimensions dynamically
  const keyHeight = Math.max(MIN_KEY_HEIGHT, canvasHeight / TOTAL_KEYS);
  const frameWidth = 2 * zoom;
  const contentWidth = totalFrames * frameWidth;

  // Resize observer
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setCanvasWidth(entry.contentRect.width);
        setCanvasHeight(entry.contentRect.height);
      }
    });

    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Draw piano roll
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width;
    const height = canvas.height;

    // Clear
    ctx.fillStyle = COLORS.background;
    ctx.fillRect(0, 0, width, height);

    // Calculate visible frame range
    const visibleWidth = width - HEADER_WIDTH;
    const scrollFrameOffset = scrollX / frameWidth;
    const startFrame = Math.max(0, Math.floor(scrollFrameOffset));
    const endFrame = Math.min(fingeringData.length, Math.ceil(scrollFrameOffset + visibleWidth / frameWidth) + 1);

    // Draw piano keys (header)
    for (let i = 0; i < TOTAL_KEYS; i++) {
      const y = (TOTAL_KEYS - 1 - i) * keyHeight;
      const isBlack = isBlackKey(i);
      
      ctx.fillStyle = isBlack ? '#252538' : '#35354d';
      ctx.fillRect(0, y, HEADER_WIDTH - 1, keyHeight - 1);
      
      // Key label (only for C notes, when tall enough)
      if (i % 12 === 3 && keyHeight >= 8) {
        ctx.fillStyle = '#888';
        ctx.font = `${Math.min(10, keyHeight - 2)}px monospace`;
        ctx.fillText(getNoteName(i), 2, y + keyHeight - 2);
      }
    }

    // Draw horizontal grid lines (keys)
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= TOTAL_KEYS; i++) {
      const y = i * keyHeight;
      ctx.beginPath();
      ctx.moveTo(HEADER_WIDTH, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw vertical lines (time markers)
    for (let frame = startFrame; frame <= endFrame; frame++) {
      if (frame % FPS === 0) {
        const x = HEADER_WIDTH + (frame - scrollFrameOffset) * frameWidth;
        ctx.strokeStyle = frame % (FPS * 5) === 0 ? COLORS.gridMajor : COLORS.grid;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
      }
    }

    // Draw fingering data with comparison
    for (let frameIdx = startFrame; frameIdx < endFrame; frameIdx++) {
      const frameData = fingeringData[frameIdx];
      if (!frameData) continue;

      const editedFingering = frameData.fingering || [];
      const originalFingering = frameData.original_fingering || editedFingering;

      // Create lookup maps
      const editedMap = new Map<number, FingeringEntry>();
      const originalMap = new Map<number, FingeringEntry>();
      
      for (const e of editedFingering) {
        editedMap.set(e.key_id, e);
      }
      for (const o of originalFingering) {
        originalMap.set(o.key_id, o);
      }

      // Get all unique key_ids
      const allKeys = new Set([...editedMap.keys(), ...originalMap.keys()]);

      for (const keyId of allKeys) {
        const edited = editedMap.get(keyId);
        const original = originalMap.get(keyId);
        
        const x = HEADER_WIDTH + (frameIdx - scrollFrameOffset) * frameWidth;
        const y = (TOTAL_KEYS - 1 - keyId) * keyHeight;
        const w = Math.max(frameWidth - 1, 2);
        const h = keyHeight - 1;

        // Skip if outside visible area
        if (x < HEADER_WIDTH - w || x > width) continue;

        if (edited && original) {
          if (edited.finger === original.finger) {
            // UNCHANGED - Green
            ctx.fillStyle = COLORS.unchanged;
            ctx.fillRect(x, y, w, h);
            drawFingerNumber(ctx, edited.finger, x, y, w, h, '#fff');
          } else {
            // CHANGED - Show original as outline, edited as filled
            // Draw original as gray outline
            ctx.strokeStyle = COLORS.originalGhost;
            ctx.lineWidth = 1;
            ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
            
            // Use yellow for missing notes, red for regular changes
            ctx.fillStyle = edited.is_missing ? COLORS.missing : COLORS.changed;
            ctx.fillRect(x + 1, y + 1, w - 2, h - 2);
            drawFingerNumber(ctx, edited.finger, x, y, w, h, edited.is_missing ? '#000' : '#fff');
          }
        } else if (edited && !original) {
          // ADDED (only in edited) - Yellow for missing, Red for regular
          ctx.fillStyle = edited.is_missing ? COLORS.missing : COLORS.changed;
          ctx.fillRect(x, y, w, h);
          drawFingerNumber(ctx, edited.finger, x, y, w, h, edited.is_missing ? '#000' : '#fff');
        } else if (!edited && original) {
          // REMOVED (only in original) - Gray outline only
          ctx.strokeStyle = COLORS.originalGhost;
          ctx.lineWidth = 1;
          ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
          
          // Draw X mark
          ctx.strokeStyle = COLORS.originalGhost;
          ctx.beginPath();
          ctx.moveTo(x + 2, y + 2);
          ctx.lineTo(x + w - 2, y + h - 2);
          ctx.moveTo(x + w - 2, y + 2);
          ctx.lineTo(x + 2, y + h - 2);
          ctx.stroke();
        }
      }
    }

    // Draw playhead (일시정지 중에만 표시, 재생 중에는 맨 왼쪽이 기준이므로 표시 안함)
    if (!isPlaying) {
      const playheadX = HEADER_WIDTH + (currentFrame - scrollFrameOffset) * frameWidth;
      
      if (playheadX >= HEADER_WIDTH && playheadX <= width) {
        ctx.strokeStyle = COLORS.playhead;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(playheadX, 0);
        ctx.lineTo(playheadX, height);
        ctx.stroke();
      }
    }

  }, [fingeringData, currentFrame, totalFrames, zoom, scrollX, canvasWidth, canvasHeight, keyHeight, frameWidth, isPlaying]);

  // Helper to draw finger number
  function drawFingerNumber(
    ctx: CanvasRenderingContext2D, 
    finger: number, 
    x: number, 
    y: number, 
    w: number, 
    h: number,
    color: string
  ) {
    if (frameWidth >= 6 && keyHeight >= 6) {
      ctx.fillStyle = color;
      ctx.font = `bold ${Math.min(10, keyHeight - 2)}px monospace`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      const fingerNum = finger > 5 ? finger - 5 : finger;
      ctx.fillText(String(fingerNum), x + w / 2, y + h / 2);
    }
  }

  // Redraw on changes
  useEffect(() => {
    draw();
  }, [draw]);

  // Mouse handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStartX(e.clientX);
    setDragStartScrollX(scrollX);
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isDragging) {
      const deltaX = dragStartX - e.clientX;
      const newScrollX = Math.max(0, Math.min(contentWidth - (canvasWidth - HEADER_WIDTH), dragStartScrollX + deltaX));
      onScrollChange(newScrollX);
    } else {
      // Hover info
      if (x > HEADER_WIDTH) {
        const frame = Math.floor((x - HEADER_WIDTH + scrollX) / frameWidth);
        const keyId = TOTAL_KEYS - 1 - Math.floor(y / keyHeight);
        
        if (frame >= 0 && frame < fingeringData.length && keyId >= 0 && keyId < TOTAL_KEYS) {
          const frameData = fingeringData[frame];
          const edited = frameData?.fingering?.find(f => f.key_id === keyId);
          const original = frameData?.original_fingering?.find(f => f.key_id === keyId);
          
          if (edited || original) {
            let status: 'unchanged' | 'changed' | 'added' | 'removed' | 'missing';
            if (edited && original) {
              if (edited.finger === original.finger) {
                status = 'unchanged';
              } else if (edited.is_missing) {
                status = 'missing';
              } else {
                status = 'changed';
              }
            } else if (edited) {
              status = edited.is_missing ? 'missing' : 'added';
            } else {
              status = 'removed';
            }
            
            setHoveredInfo({
              frame,
              key: keyId,
              finger: edited?.finger || original?.finger || 0,
              originalFinger: original?.finger || null,
              note: getNoteName(keyId),
              status,
            });
          } else {
            setHoveredInfo(null);
          }
        }
      }
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleMouseLeave = () => {
    setIsDragging(false);
    setHoveredInfo(null);
  };

  const handleClick = (e: React.MouseEvent) => {
    if (isDragging) return;
    
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;

    if (x > HEADER_WIDTH) {
      const frame = Math.floor((x - HEADER_WIDTH + scrollX) / frameWidth);
      if (frame >= 0 && frame < totalFrames) {
        onSeek(frame);
      }
    }
  };

  // Native wheel event listener
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleWheel = (e: WheelEvent) => {
      e.preventDefault();
      const delta = e.deltaX || e.deltaY;
      const maxScroll = Math.max(0, contentWidth - (canvasWidth - HEADER_WIDTH));
      const newScrollX = Math.max(0, Math.min(maxScroll, scrollX + delta));
      onScrollChange(newScrollX);
    };

    canvas.addEventListener('wheel', handleWheel, { passive: false });
    return () => {
      canvas.removeEventListener('wheel', handleWheel);
    };
  }, [contentWidth, canvasWidth, scrollX, onScrollChange]);

  // Format status text
  const getStatusText = (status: string, finger: number, originalFinger: number | null) => {
    const fingerNum = finger > 5 ? finger - 5 : finger;
    const hand = finger > 5 ? 'R' : 'L';
    
    switch (status) {
      case 'unchanged':
        return `${hand}${fingerNum} (unchanged)`;
      case 'changed':
        const origNum = originalFinger ? (originalFinger > 5 ? originalFinger - 5 : originalFinger) : '?';
        const origHand = originalFinger && originalFinger > 5 ? 'R' : 'L';
        return `${origHand}${origNum} → ${hand}${fingerNum}`;
      case 'added':
        return `${hand}${fingerNum} (added)`;
      case 'removed':
        return `${hand}${fingerNum} (removed)`;
      case 'missing':
        return `${hand}${fingerNum} (missing note)`;
      default:
        return `${hand}${fingerNum}`;
    }
  };

  return (
    <div className="piano-roll-wrapper" ref={containerRef}>
      <canvas
        ref={canvasRef}
        width={canvasWidth}
        height={canvasHeight}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseLeave}
        onClick={handleClick}
        style={{ cursor: isDragging ? 'grabbing' : 'crosshair' }}
      />
      {hoveredInfo && (
        <div className={`hover-tooltip ${hoveredInfo.status}`}>
          <span className="tooltip-frame">F{hoveredInfo.frame}</span>
          <span className="tooltip-note">{hoveredInfo.note}</span>
          <span className="tooltip-status">
            {getStatusText(hoveredInfo.status, hoveredInfo.finger, hoveredInfo.originalFinger)}
          </span>
        </div>
      )}
    </div>
  );
}
