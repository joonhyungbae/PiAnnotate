import { useCallback, useEffect, useRef } from 'react';
import { useAppStore } from '../stores/useAppStore';

const FPS = 60000 / 1001; // ~59.94 fps

export function useAudioSync(audioRef: React.RefObject<HTMLAudioElement | null>) {
  const {
    totalFrames,
    setCurrentFrame,
    isPlaying,
    setIsPlaying,
  } = useAppStore();

  const totalDuration = totalFrames / FPS;
  const lastPreloadRef = useRef<number>(0);

  // Calculate current frame
  const getCurrentFrameIndex = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return 0;
    return Math.floor(audio.currentTime * FPS);
  }, [audioRef]);

  // Toggle play/pause
  const togglePlay = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;

    if (isPlaying) {
      audio.pause();
      setIsPlaying(false);
    } else {
      audio.play();
      setIsPlaying(true);
    }
  }, [audioRef, isPlaying, setIsPlaying]);

  // Seek to time
  const seek = useCallback(
    (percentage: number) => {
      const audio = audioRef.current;
      if (!audio || totalDuration <= 0) return;

      const newTime = (percentage / 100) * totalDuration;
      audio.currentTime = newTime;
      setCurrentFrame(Math.floor(newTime * FPS));
    },
    [audioRef, totalDuration, setCurrentFrame]
  );

  // Set volume
  const setVolume = useCallback(
    (volume: number) => {
      const audio = audioRef.current;
      if (!audio) return;
      audio.volume = Math.max(0, Math.min(1, volume));
    },
    [audioRef]
  );

  // Format time
  const formatTime = useCallback((seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60)
      .toString()
      .padStart(2, '0');
    return `${mins}:${secs}`;
  }, []);

  // Set up audio event handlers
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const handleTimeUpdate = () => {
      const frameIndex = getCurrentFrameIndex();
      setCurrentFrame(frameIndex);
    };

    const handlePlay = () => setIsPlaying(true);
    const handlePause = () => setIsPlaying(false);
    const handleEnded = () => setIsPlaying(false);

    audio.addEventListener('timeupdate', handleTimeUpdate);
    audio.addEventListener('play', handlePlay);
    audio.addEventListener('pause', handlePause);
    audio.addEventListener('ended', handleEnded);

    return () => {
      audio.removeEventListener('timeupdate', handleTimeUpdate);
      audio.removeEventListener('play', handlePlay);
      audio.removeEventListener('pause', handlePause);
      audio.removeEventListener('ended', handleEnded);
    };
  }, [audioRef, getCurrentFrameIndex, setCurrentFrame, setIsPlaying]);

  return {
    fps: FPS,
    totalDuration,
    getCurrentFrameIndex,
    togglePlay,
    seek,
    setVolume,
    formatTime,
    lastPreloadRef,
  };
}

