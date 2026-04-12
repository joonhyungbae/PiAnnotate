import { useEffect, useState, useRef, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { fetchFingeringData, fetchPieceMetadata, getAudioUrl } from '../../api';
import type { FingeringFrame, FingeringEntry } from '../../api';
import { PianoRoll } from './PianoRoll';
import './FingeringCompare.css';

interface PieceInfo {
  name: string;
  composer: string;
  num_frames: number;
}

export function FingeringCompare() {
  const { pieceId } = useParams<{ pieceId: string }>();
  const navigate = useNavigate();
  const pieceIdNum = parseInt(pieceId || '0', 10);

  // Data state
  const [pieceInfo, setPieceInfo] = useState<PieceInfo | null>(null);
  const [fingeringData, setFingeringData] = useState<FingeringFrame[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingProgress, setLoadingProgress] = useState(0);
  const [loadingStatus, setLoadingStatus] = useState('Initializing...');
  const [error, setError] = useState<string | null>(null);

  // Playback state
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [duration, setDuration] = useState(0);
  const audioRef = useRef<HTMLAudioElement>(null);

  // View state
  const [zoom, setZoom] = useState(2);
  const [scrollX, setScrollX] = useState(0);

  // Stats
  const [stats, setStats] = useState({ total: 0, changed: 0, changeRate: 0 });
  const [changedFrameIndices, setChangedFrameIndices] = useState<number[]>([]);

  const FPS = 60;

  // Load data
  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setLoadingProgress(0);
        setLoadingStatus('Loading metadata...');
        
        // Load metadata first
        const meta = await fetchPieceMetadata(pieceIdNum);
        setLoadingProgress(20);
        
        setPieceInfo({
          name: meta.name,
          composer: meta.composer,
          num_frames: meta.num_frames,
        });
        
        setLoadingStatus('Loading fingering data...');
        setLoadingProgress(40);
        
        // Load fingering data
        const fingering = await fetchFingeringData(pieceIdNum);
        setLoadingProgress(70);
        
        setFingeringData(fingering);
        setLoadingStatus('Processing data...');
        setLoadingProgress(85);

        // Calculate stats and collect changed frame indices
        let changedFramesCount = 0;
        let totalWithFingering = 0;
        const changedIndices: number[] = [];
        
        for (let i = 0; i < fingering.length; i++) {
          const frame = fingering[i];
          if (frame.fingering && frame.fingering.length > 0) {
            totalWithFingering++;
            if (frame.original_fingering) {
              // Compare fingering
              const currentStr = JSON.stringify([...frame.fingering].sort((a, b) => a.key_id - b.key_id));
              const originalStr = JSON.stringify([...frame.original_fingering].sort((a, b) => a.key_id - b.key_id));
              if (currentStr !== originalStr) {
                changedFramesCount++;
                changedIndices.push(i);
              }
            }
          }
        }
        setChangedFrameIndices(changedIndices);
        setStats({
          total: totalWithFingering,
          changed: changedFramesCount,
          changeRate: totalWithFingering > 0 ? (changedFramesCount / totalWithFingering) * 100 : 0,
        });

        setLoadingProgress(100);
        setLoadingStatus('Complete!');
        
        // Small delay to show 100%
        await new Promise(resolve => setTimeout(resolve, 200));
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
        setLoading(false);
      }
    }
    loadData();
  }, [pieceIdNum]);

  // Audio 메타데이터 및 종료 이벤트 리스너
  useEffect(() => {
    if (loading) return;
    
    const audio = audioRef.current;
    if (!audio) return;

    const handleLoadedMetadata = () => {
      setDuration(audio.duration);
    };

    const handleEnded = () => {
      setIsPlaying(false);
    };

    // 일시정지 상태에서 seek 시 프레임 업데이트
    const handleTimeUpdate = () => {
      if (!isPlaying) {
        const frame = Math.floor(audio.currentTime * FPS);
        setCurrentFrame(frame);
      }
    };

    audio.addEventListener('loadedmetadata', handleLoadedMetadata);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('timeupdate', handleTimeUpdate);

    return () => {
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('timeupdate', handleTimeUpdate);
    };
  }, [loading, isPlaying]);

  // 재생 중 부드러운 스크롤 - requestAnimationFrame 사용
  useEffect(() => {
    if (!isPlaying) return;
    
    const audio = audioRef.current;
    if (!audio) return;
    
    let animationFrameId: number;
    
    const updateScroll = () => {
      const currentTime = audio.currentTime;
      const frame = Math.floor(currentTime * FPS);
      const frameWidth = 2 * zoom;
      
      // 현재 프레임 업데이트
      setCurrentFrame(frame);
      
      // 부드러운 스크롤: 정확한 시간 기반으로 계산
      const exactScrollX = currentTime * FPS * frameWidth;
      setScrollX(Math.max(0, exactScrollX));
      
      // 다음 프레임 요청
      animationFrameId = requestAnimationFrame(updateScroll);
    };
    
    // 애니메이션 시작
    animationFrameId = requestAnimationFrame(updateScroll);
    
    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isPlaying, zoom]);

  // Play/Pause
  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
    setIsPlaying(!isPlaying);
  }, [isPlaying]);

  // Seek (수동 탐색 시 스크롤도 업데이트)
  const handleSeek = useCallback((frame: number) => {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = frame / FPS;
    setCurrentFrame(frame);
    
    // 수동 seek 시에도 자동 스크롤
    const frameWidth = 2 * zoom;
    const viewportWidth = 1200;
    const targetScrollX = Math.max(0, frame * frameWidth - viewportWidth / 3);
    setScrollX(targetScrollX);
  }, [zoom]);

  // Zoom
  const handleZoomIn = () => setZoom((z) => Math.min(z * 1.5, 10));
  const handleZoomOut = () => setZoom((z) => Math.max(z / 1.5, 0.1));

  // Navigate to next/prev change
  const findNextChange = useCallback((fromFrame: number, direction: 1 | -1) => {
    const step = direction;
    for (let i = fromFrame + step; i >= 0 && i < fingeringData.length; i += step) {
      const frame = fingeringData[i];
      if (frame.original_fingering && frame.fingering) {
        const currentStr = JSON.stringify(frame.fingering.sort((a, b) => a.key_id - b.key_id));
        const originalStr = JSON.stringify(frame.original_fingering.sort((a, b) => a.key_id - b.key_id));
        if (currentStr !== originalStr) {
          return i;
        }
      }
    }
    return null;
  }, [fingeringData]);

  const goToNextChange = () => {
    const next = findNextChange(currentFrame, 1);
    if (next !== null) handleSeek(next);
  };

  const goPrevChange = () => {
    const prev = findNextChange(currentFrame, -1);
    if (prev !== null) handleSeek(prev);
  };

  // Format time
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  if (loading) {
    return (
      <div className="fingering-compare loading">
        <div className="loading-spinner"></div>
        <span className="loading-text">{loadingStatus}</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="fingering-compare error">
        <div className="error-container">
          <div className="error-icon">⚠️</div>
          <div className="error-message">{error}</div>
          <button className="error-btn" onClick={() => navigate('/')}>
            ← Back to List
          </button>
        </div>
      </div>
    );
  }

  const totalFrames = pieceInfo?.num_frames || fingeringData.length;
  const hasEdits = stats.changed > 0;

  return (
    <div className="fingering-compare">
      {/* Header */}
      <header className="fc-header">
        <button className="back-btn" onClick={() => navigate('/')}>
          ← Back
        </button>
        <div className="piece-info">
          <span className="piece-id">#{pieceIdNum}</span>
          <span className="piece-name">{pieceInfo?.name}</span>
          <span className="piece-composer">{pieceInfo?.composer}</span>
        </div>
        {hasEdits && (
          <div className="stats-badge">
            <span className="changed">{stats.changed.toLocaleString()}</span>
            <span className="separator">/</span>
            <span className="total">{stats.total.toLocaleString()}</span>
            <span className="rate">({stats.changeRate.toFixed(1)}%)</span>
          </div>
        )}
      </header>

      {/* Legend */}
      <div className="legend-bar">
        <div className="legend-item unchanged">
          <span className="legend-dot"></span>
          <span>Unchanged</span>
        </div>
        <div className="legend-item changed">
          <span className="legend-dot"></span>
          <span>Modified</span>
        </div>
        <div className="legend-item missing">
          <span className="legend-dot"></span>
          <span>Missing Note Added</span>
        </div>
        <div className="legend-item removed">
          <span className="legend-dot"></span>
          <span>Original (changed)</span>
        </div>
      </div>

      {/* Single Piano Roll with Comparison View */}
      <div className="piano-roll-container">
        <PianoRoll
          fingeringData={fingeringData}
          currentFrame={currentFrame}
          totalFrames={totalFrames}
          zoom={zoom}
          scrollX={scrollX}
          onScrollChange={setScrollX}
          onSeek={handleSeek}
          isPlaying={isPlaying}
        />
      </div>

      {/* Controls */}
      <div className="fc-controls">
        <div className="playback-controls">
          <button className="control-btn" onClick={goPrevChange} title="Previous Change">
            ⏮ Prev
          </button>
          <button className="control-btn play-btn" onClick={togglePlay}>
            {isPlaying ? '⏸ Pause' : '▶ Play'}
          </button>
          <button className="control-btn" onClick={goToNextChange} title="Next Change">
            Next ⏭
          </button>
        </div>

        <div className="timeline">
          <span className="time current">{formatTime(currentFrame / FPS)}</span>
          <div className="timeline-container">
            {/* 수정된 부분 마커들 */}
            <div className="change-markers">
              {changedFrameIndices.map((frameIdx) => (
                <div
                  key={frameIdx}
                  className="change-marker"
                  style={{ left: `${(frameIdx / totalFrames) * 100}%` }}
                  onClick={() => handleSeek(frameIdx)}
                  title={`Changed at ${formatTime(frameIdx / FPS)}`}
                />
              ))}
            </div>
            <input
              type="range"
              min={0}
              max={totalFrames - 1}
              value={currentFrame}
              onChange={(e) => handleSeek(parseInt(e.target.value, 10))}
              className="timeline-slider"
            />
          </div>
          <span className="time duration">{formatTime(duration || totalFrames / FPS)}</span>
        </div>

        <div className="zoom-controls">
          <span className="zoom-label">Zoom:</span>
          <button className="control-btn" onClick={handleZoomOut}>−</button>
          <span className="zoom-value">{zoom.toFixed(1)}x</span>
          <button className="control-btn" onClick={handleZoomIn}>+</button>
        </div>
      </div>

      {/* Hidden Audio */}
      <audio ref={audioRef} src={getAudioUrl(pieceIdNum)} preload="metadata" />
    </div>
  );
}

