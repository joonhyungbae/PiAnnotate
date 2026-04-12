import { useState, useEffect } from 'react';
import { useAppStore } from '../../stores/useAppStore';
import './AudioPlayer.css';

const FPS = 60000 / 1001;

interface AudioPlayerProps {
  audioRef: React.RefObject<HTMLAudioElement | null>;
}

export function AudioPlayer({ audioRef }: AudioPlayerProps) {
  const { totalFrames, loadedFrames, currentFrame, isPlaying, setIsPlaying, pieceName, pieceComposer } =
    useAppStore();

  const [volume, setVolume] = useState(1);
  const [scrubberValue, setScrubberValue] = useState(0);

  const totalDuration = totalFrames / FPS;

  // Calculate current time
  const currentTime = totalDuration > 0 ? (currentFrame / totalFrames) * totalDuration : 0;
  const progress = totalFrames > 0 ? (loadedFrames / totalFrames) * 100 : 0;

  // Update scrubber value
  useEffect(() => {
    if (totalDuration > 0) {
      setScrubberValue((currentTime / totalDuration) * 100);
    }
  }, [currentTime, totalDuration]);

  // Audio event listeners
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioRef, setIsPlaying]);

  const togglePlay = () => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
    } else {
      audio.play();
    }
  };

  const handleScrubberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio || totalDuration <= 0) return;

    const value = parseFloat(e.target.value);
    setScrubberValue(value);
    audio.currentTime = (value / 100) * totalDuration;
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const audio = audioRef.current;
    if (!audio) return;

    const newVolume = parseFloat(e.target.value);
    setVolume(newVolume);
    audio.volume = newVolume;
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  };

  const getVolumeIcon = () => {
    if (volume === 0) return '🔇';
    if (volume <= 0.5) return '🔉';
    return '🔊';
  };

  // Go to previous frame
  const goToPrevFrame = () => {
    const audio = audioRef.current;
    if (!audio || totalDuration <= 0) return;

    const frameDuration = 1 / FPS;
    const newTime = Math.max(0, audio.currentTime - frameDuration);
    audio.currentTime = newTime;
    setScrubberValue((newTime / totalDuration) * 100);
  };

  // Go to next frame
  const goToNextFrame = () => {
    const audio = audioRef.current;
    if (!audio || totalDuration <= 0) return;

    const frameDuration = 1 / FPS;
    const newTime = Math.min(totalDuration, audio.currentTime + frameDuration);
    audio.currentTime = newTime;
    setScrubberValue((newTime / totalDuration) * 100);
  };

  return (
    <>
      {/* Info box */}
      <div className="info-box">
        <p className="piece-info">
          {pieceName}
          <br />
          <span className="composer">by {pieceComposer}</span>
        </p>
        {progress < 100 && (
          <p className="loading-text">Loading... {Math.floor(progress)}%</p>
        )}
      </div>

      {/* Audio player */}
      <div className="player">
        <button className="play-btn" onClick={togglePlay}>
          {isPlaying ? '⏸' : '▶'}
        </button>

        <div className="time-display">{formatTime(currentTime)}</div>

        <div className="frame-display" title="Current frame / Total frames">
          <span className="frame-current">{currentFrame}</span>
          <span className="frame-separator">/</span>
          <span className="frame-total">{totalFrames}</span>
        </div>

        <div className="progress-bar-container">
          <div className="buffered-bar" style={{ width: `${progress}%` }} />
          <div className="played-bar" style={{ width: `${scrubberValue}%` }} />
          <input
            type="range"
            className="scrubber"
            value={scrubberValue}
            min="0"
            max="100"
            step="0.1"
            onChange={handleScrubberChange}
          />
        </div>

        <div className="time-display">{formatTime(totalDuration)}</div>

        <div className="frame-nav-container">
          <button className="frame-nav-btn" onClick={goToPrevFrame} title="Previous frame">
            ⏮
          </button>
          <button className="frame-nav-btn" onClick={goToNextFrame} title="Next frame">
            ⏭
          </button>
        </div>

        <div className="volume-container">
          <span className="volume-icon">{getVolumeIcon()}</span>
          <input
            type="range"
            className="volume-bar"
            min="0"
            max="1"
            step="0.1"
            value={volume}
            onChange={handleVolumeChange}
          />
        </div>
      </div>
    </>
  );
}
