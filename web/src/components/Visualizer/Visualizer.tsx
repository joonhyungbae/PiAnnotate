import { useEffect, useRef, useState, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { fetchPieceAnnotationStatus, updateAnnotationStatus, updateSubtaskStatus, fetchMotionIssues, addMotionIssue, deleteMotionIssue, fetchPostPlaying, savePostPlaying, deletePostPlaying, fetchTestSegments, addTestSegment, deleteTestSegment, fetchPiecesMetadata, getFingeringDataUrl, fetchFingeringData, fetchAnnotationProgress } from '../../api';
import { getFingering } from '../../hooks/useStreamData';
import { useAppStore } from '../../stores/useAppStore';
import type { MotionIssue, PostPlayingSegment, TestSegment, SplitType, FingeringFrameData } from '../../types';
import './Visualizer.css';

// Frame rate (NTSC standard)
const FPS = 60000 / 1001; // ~59.94 fps

// Format seconds to mm:ss.ms
function formatTime(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins}:${secs.toFixed(2).padStart(5, '0')}`;
}

// Parse time string (mm:ss.ms or ss.ms) to seconds
function parseTime(timeStr: string): number | null {
  const parts = timeStr.split(':');
  if (parts.length === 2) {
    const mins = parseInt(parts[0], 10);
    const secs = parseFloat(parts[1]);
    if (!isNaN(mins) && !isNaN(secs)) {
      return mins * 60 + secs;
    }
  } else if (parts.length === 1) {
    const secs = parseFloat(parts[0]);
    if (!isNaN(secs)) {
      return secs;
    }
  }
  return null;
}

export function Visualizer() {
  const [searchParams] = useSearchParams();
  const pieceId = parseInt(searchParams.get('id') || '0', 10);
  const scriptLoadedRef = useRef(false);
  const [showFingering, setShowFingering] = useState(true);
  const [showSkeleton, setShowSkeleton] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [showSpeedDropdown, setShowSpeedDropdown] = useState(false);
  const [isReview1Completed, setIsReview1Completed] = useState(false);
  const [isReview2Completed, setIsReview2Completed] = useState(false);
  const [isReview3Completed, setIsReview3Completed] = useState(false);
  const [isFingeringCompleted, setIsFingeringCompleted] = useState(false);
  
  // AI annotation (r0) state from store
  const isAiAnnotation = useAppStore((state) => state.isAiAnnotation);
  const [isPostCompleted, setIsPostCompleted] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);
  
  // Motion issues state
  const [motionIssues, setMotionIssues] = useState<MotionIssue[]>([]);
  const [showIssuePopup, setShowIssuePopup] = useState(false);
  const [showIssueList, setShowIssueList] = useState(false);
  const [issueStartTime, setIssueStartTime] = useState('');
  const [issueEndTime, setIssueEndTime] = useState('');
  const [isAddingIssue, setIsAddingIssue] = useState(false);
  const [audioDuration, setAudioDuration] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);

  // Volume/Mute state
  const [isMuted, setIsMuted] = useState(false);
  const [previousVolume, setPreviousVolume] = useState(1);

  // Post-playing state
  const [postPlaying, setPostPlaying] = useState<PostPlayingSegment | null>(null);
  const [showPostPopup, setShowPostPopup] = useState(false);
  const [postStartTime, setPostStartTime] = useState('');
  const [postEndTime, setPostEndTime] = useState('');
  const [isSavingPost, setIsSavingPost] = useState(false);

  // Test segments state (for TEST dataset only)
  const [pieceSplit, setPieceSplit] = useState<SplitType>('unknown');
  const [testSegments, setTestSegments] = useState<TestSegment[]>([]);
  const [showTestPopup, setShowTestPopup] = useState(false);
  const [showTestList, setShowTestList] = useState(false);
  const [testStartTime, setTestStartTime] = useState('');
  const [testEndTime, setTestEndTime] = useState('');
  const [testNote, setTestNote] = useState('');
  const [isAddingTest, setIsAddingTest] = useState(false);
  
  // Tooltip state for ambiguous fingering
  const [tooltipContent, setTooltipContent] = useState<string | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const tooltipRef = useRef<HTMLDivElement>(null);
  
  // Frame tooltip state (hover on progress bar)
  const [frameTooltip, setFrameTooltip] = useState<{ frame: number; x: number } | null>(null);
  
  // Ambiguous fingering markers
  const [ambiguousFrames, setAmbiguousFrames] = useState<number[]>([]);
  
  // Missing fingering segments (pressed keys without fingering)
  const [missingFingeringSegments, setMissingFingeringSegments] = useState<Array<{start_time: number, end_time: number, keyIndex: number}>>([]);
  
  // Edited fingering frames (frames where fingering was modified)
  const [editedFingeringFrames, setEditedFingeringFrames] = useState<number[]>([]);
  
  // Resume annotation popup state
  const [showResumePopup, setShowResumePopup] = useState(false);
  const [resumeProgress, setResumeProgress] = useState<{ lastFrame: number; lastTimeSeconds: number } | null>(null);
  const [isDataLoaded, setIsDataLoaded] = useState(false);
  
  // Update Show Fingering button state
  // Update current frame display based on audio time and navigation events
  useEffect(() => {
    const audio = document.getElementById('audio') as HTMLAudioElement;
    if (!audio) return;
    
    const updateFrame = () => {
      const frame = Math.floor(audio.currentTime * FPS);
      setCurrentFrame(frame);
    };
    
    // vis.js에서 발생시키는 frameChange 이벤트 처리 (핑거링 네비게이션용)
    const handleFrameChange = (e: CustomEvent<{ frame: number }>) => {
      setCurrentFrame(e.detail.frame);
    };
    
    audio.addEventListener('timeupdate', updateFrame);
    audio.addEventListener('seeked', updateFrame);
    window.addEventListener('frameChange', handleFrameChange as EventListener);
    
    return () => {
      audio.removeEventListener('timeupdate', updateFrame);
      audio.removeEventListener('seeked', updateFrame);
      window.removeEventListener('frameChange', handleFrameChange as EventListener);
    };
  }, []);

  // Listen for fingering toggle from keyboard shortcuts in vis.js
  useEffect(() => {
    const handleFingeringToggle = (e: CustomEvent<boolean>) => {
      console.log('Fingering toggle event received:', e.detail);
      setShowFingering(e.detail);
    };
    
    window.addEventListener('fingeringToggle', handleFingeringToggle as EventListener);
    return () => {
      window.removeEventListener('fingeringToggle', handleFingeringToggle as EventListener);
    };
  }, []);

  // Sync state when playback rate changes via keyboard shortcuts
  useEffect(() => {
    const handlePlaybackRateChange = (e: CustomEvent) => {
      setPlaybackRate(e.detail);
    };
    
    window.addEventListener('playbackRateChange', handlePlaybackRateChange as EventListener);
    return () => {
      window.removeEventListener('playbackRateChange', handlePlaybackRateChange as EventListener);
    };
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      const speedContainer = document.querySelector('.speed-container');
      if (speedContainer && !speedContainer.contains(e.target as Node)) {
        setShowSpeedDropdown(false);
      }
    };

    if (showSpeedDropdown) {
      document.addEventListener('click', handleClickOutside);
    }
    
    return () => {
      document.removeEventListener('click', handleClickOutside);
    };
  }, [showSpeedDropdown]);

  // Load annotation status (2-stage + subtasks)
  useEffect(() => {
    if (pieceId >= 0) {
      fetchPieceAnnotationStatus(pieceId)
        .then((status) => {
          setIsReview1Completed(status.review1_completed);
          setIsReview2Completed(status.review2_completed);
          setIsReview3Completed(status.review3_completed);
          setIsFingeringCompleted(status.fingering_completed);
          setIsPostCompleted(status.post_completed);
        })
        .catch((err) => {
          console.error('Failed to fetch annotation status:', err);
        });
    }
  }, [pieceId]);

  // Listen for visualizer data loaded event
  useEffect(() => {
    const handleDataLoaded = (e: CustomEvent) => {
      console.log('Visualizer data loaded:', e.detail);
      setIsDataLoaded(true);
    };
    
    window.addEventListener('visualizerDataLoaded', handleDataLoaded as EventListener);
    return () => {
      window.removeEventListener('visualizerDataLoaded', handleDataLoaded as EventListener);
    };
  }, []);

  // Load annotation progress (resume from where left off)
  // Wait until all data is loaded before showing popup
  // Don't show popup if R1 is already completed (annotation is done)
  useEffect(() => {
    if (pieceId >= 0 && isDataLoaded && !isReview1Completed) {
      fetchAnnotationProgress(pieceId)
        .then((progress) => {
          if (progress && progress.last_frame > 0) {
            setResumeProgress({
              lastFrame: progress.last_frame,
              lastTimeSeconds: progress.last_time_seconds,
            });
            setShowResumePopup(true);
          }
        })
        .catch((err) => {
          console.error('Failed to fetch annotation progress:', err);
        });
    }
  }, [pieceId, isDataLoaded, isReview1Completed]);

  // Load motion issues
  useEffect(() => {
    if (pieceId >= 0) {
      fetchMotionIssues(pieceId)
        .then((issues) => {
          setMotionIssues(issues);
        })
        .catch((err) => {
          console.error('Failed to fetch motion issues:', err);
        });
    }
  }, [pieceId]);

  // Load post-playing segment
  useEffect(() => {
    if (pieceId >= 0) {
      fetchPostPlaying(pieceId)
        .then((segment) => {
          setPostPlaying(segment);
        })
        .catch((err) => {
          console.error('Failed to fetch post-playing segment:', err);
        });
    }
  }, [pieceId]);

  // Load ambiguous fingering frames (grouped by continuous segments)
  useEffect(() => {
    if (pieceId >= 0 && audioDuration > 0) {
      fetchFingeringData(pieceId)
        .then((frames) => {
          const ambiguousFrameIndices: number[] = [];
          frames.forEach((frame) => {
            if (frame.fingering && frame.fingering.some(f => f.ambiguous === true)) {
              ambiguousFrameIndices.push(frame.frame_idx);
            }
          });
          
          // Group continuous frames and keep only the first frame of each segment
          const segmentStarts: number[] = [];
          for (let i = 0; i < ambiguousFrameIndices.length; i++) {
            const current = ambiguousFrameIndices[i];
            const prev = ambiguousFrameIndices[i - 1];
            // Start of a new segment if it's the first or not continuous
            if (i === 0 || current !== prev + 1) {
              segmentStarts.push(current);
            }
          }
          
          setAmbiguousFrames(segmentStarts);
        })
        .catch((err) => {
          console.error('Failed to fetch fingering data for ambiguous markers:', err);
        });
    }
  }, [pieceId, audioDuration]);

  // Listen for ambiguous changes from vis.js (startFrame of segment is sent)
  useEffect(() => {
    const handleAmbiguousChanged = (e: CustomEvent) => {
      const { frameIndex, isAmbiguous } = e.detail;
      console.log('Ambiguous changed event:', frameIndex, isAmbiguous);
      if (isAmbiguous) {
        setAmbiguousFrames(prev => {
          if (!prev.includes(frameIndex)) {
            const newFrames = [...prev, frameIndex].sort((a, b) => a - b);
            console.log('Updated ambiguous frames:', newFrames);
            return newFrames;
          }
          return prev;
        });
      } else {
        setAmbiguousFrames(prev => prev.filter(f => f !== frameIndex));
      }
    };

    window.addEventListener('ambiguousChanged', handleAmbiguousChanged as EventListener);
    return () => {
      window.removeEventListener('ambiguousChanged', handleAmbiguousChanged as EventListener);
    };
  }, []);

  // Load missing fingering segments (pressed keys without fingering)
  // 프레임 데이터가 로드된 후 계산
  useEffect(() => {
    if (pieceId >= 0 && audioDuration > 0) {
      // 프레임 데이터 로드를 기다린 후 계산
      const checkAndUpdate = () => {
        if ((window as any).visualizerControls) {
          // Missing fingering segments
          if ((window as any).visualizerControls.getMissingFingeringSegments) {
            try {
              const missingSegments = (window as any).visualizerControls.getMissingFingeringSegments();
              setMissingFingeringSegments(missingSegments);
            } catch (err) {
              console.warn('Failed to get missing fingering segments:', err);
            }
          }
          
          // Edited fingering frames
          if ((window as any).visualizerControls.getEditedFingeringFrames) {
            try {
              const editedFrames = (window as any).visualizerControls.getEditedFingeringFrames();
              setEditedFingeringFrames(editedFrames);
            } catch (err) {
              console.warn('Failed to get edited fingering frames:', err);
            }
          }
        }
      };
      
      // 즉시 시도
      checkAndUpdate();
      
      // 프레임 로딩 후 다시 시도 (5초 후)
      const timeout = setTimeout(checkAndUpdate, 5000);
      
      // 주기적으로 업데이트 (핑거링 수정 시 반영)
      const interval = setInterval(checkAndUpdate, 2000);
      
      // 핑거링 수정 이벤트 리스너
      const handleFingeringEdited = () => {
        checkAndUpdate();
      };
      window.addEventListener('fingeringEdited', handleFingeringEdited);
      
      return () => {
        clearTimeout(timeout);
        clearInterval(interval);
        window.removeEventListener('fingeringEdited', handleFingeringEdited);
      };
    }
  }, [pieceId, audioDuration]);

  // Load piece split info and test segments (if TEST dataset)
  useEffect(() => {
    if (pieceId >= 0) {
      fetchPiecesMetadata()
        .then((pieces) => {
          const piece = pieces.find(p => p.piece_id === pieceId);
          if (piece && piece.split) {
            setPieceSplit(piece.split);
            // Load test segments only for test dataset
            if (piece.split === 'test') {
              fetchTestSegments(pieceId)
                .then((segments) => {
                  setTestSegments(segments);
                })
                .catch((err) => {
                  console.error('Failed to fetch test segments:', err);
                });
            }
          }
        })
        .catch((err) => {
          console.error('Failed to fetch piece metadata:', err);
        });
    }
  }, [pieceId]);

  // Track audio duration
  useEffect(() => {
    const checkAudioDuration = () => {
      const audio = document.getElementById('audio') as HTMLAudioElement;
      if (audio && audio.duration && !isNaN(audio.duration)) {
        setAudioDuration(audio.duration);
      }
    };
    
    const interval = setInterval(checkAudioDuration, 500);
    return () => clearInterval(interval);
  }, []);

  // Toggle review 1 (annotator) completion handler
  // Special rules: A3 → auto R2, A4 → auto R3
  const handleToggleReview1 = async () => {
    if (isUpdating) return;
    
    setIsUpdating(true);
    try {
      const newStatus = !isReview1Completed;
      const result = await updateAnnotationStatus(pieceId, 1, newStatus);
      setIsReview1Completed(newStatus);
      
      // Handle auto-completed reviews (A3 → R2, A4 → R3)
      if (result.auto_review_completed === 2) {
        setIsReview2Completed(true);
      } else if (result.auto_review_completed === 3) {
        setIsReview3Completed(true);
      }
      
      // If uncompleting R1, also uncomplete auto reviews
      if (!newStatus) {
        // Server handles this, but we need to refresh the state
        // The server will uncomplete R2 for A3, R3 for A4
        const status = await fetchPieceAnnotationStatus(pieceId);
        setIsReview2Completed(status.review2_completed);
        setIsReview3Completed(status.review3_completed);
      }
    } catch (err) {
      console.error('Failed to update review 1 status:', err);
    } finally {
      setIsUpdating(false);
    }
  };

  // Toggle review 2 (pianist) completion handler
  const handleToggleReview2 = async () => {
    if (isUpdating) return;
    
    setIsUpdating(true);
    try {
      const newStatus = !isReview2Completed;
      await updateAnnotationStatus(pieceId, 2, newStatus);
      setIsReview2Completed(newStatus);
    } catch (err) {
      console.error('Failed to update review 2 status:', err);
    } finally {
      setIsUpdating(false);
    }
  };

  // Toggle review 3 completion handler
  const handleToggleReview3 = async () => {
    if (isUpdating) return;
    
    setIsUpdating(true);
    try {
      const newStatus = !isReview3Completed;
      await updateAnnotationStatus(pieceId, 3, newStatus);
      setIsReview3Completed(newStatus);
    } catch (err) {
      console.error('Failed to update review 3 status:', err);
    } finally {
      setIsUpdating(false);
    }
  };

  // Toggle fingering completion handler
  const handleToggleFingering = async () => {
    if (isUpdating) return;
    
    setIsUpdating(true);
    try {
      const newStatus = !isFingeringCompleted;
      const result = await updateSubtaskStatus(pieceId, 'fingering', newStatus);
      setIsFingeringCompleted(newStatus);
      
      // If both are now complete, R1 is auto-completed
      if (result.auto_r1_completed) {
        setIsReview1Completed(true);
      }
    } catch (err) {
      console.error('Failed to update fingering status:', err);
    } finally {
      setIsUpdating(false);
    }
  };

  // Toggle post completion handler
  const handleTogglePost = async () => {
    if (isUpdating) return;
    
    setIsUpdating(true);
    try {
      const newStatus = !isPostCompleted;
      const result = await updateSubtaskStatus(pieceId, 'post', newStatus);
      setIsPostCompleted(newStatus);
      
      // If both are now complete, R1 is auto-completed
      if (result.auto_r1_completed) {
        setIsReview1Completed(true);
      }
    } catch (err) {
      console.error('Failed to update post status:', err);
    } finally {
      setIsUpdating(false);
    }
  };

  // Get current audio time
  const getCurrentTime = useCallback(() => {
    const audio = document.getElementById('audio') as HTMLAudioElement;
    return audio ? audio.currentTime : 0;
  }, []);

  // Open issue popup with current time as start
  const handleOpenIssuePopup = () => {
    const currentTime = getCurrentTime();
    setIssueStartTime(formatTime(currentTime));
    setIssueEndTime('');
    setShowIssuePopup(true);
  };

  // Set current time to issue field
  const handleSetIssueTime = (field: 'start' | 'end') => {
    const currentTime = getCurrentTime();
    if (field === 'start') {
      setIssueStartTime(formatTime(currentTime));
    } else {
      setIssueEndTime(formatTime(currentTime));
    }
  };

  // Submit new issue
  const handleSubmitIssue = async () => {
    const startTime = parseTime(issueStartTime);
    const endTime = parseTime(issueEndTime);
    
    if (startTime === null || endTime === null) {
      alert('Please enter valid time format (mm:ss.ms or ss.ms)');
      return;
    }
    
    if (startTime >= endTime) {
      alert('Start time must be less than end time');
      return;
    }
    
    setIsAddingIssue(true);
    try {
      const result = await addMotionIssue(pieceId, startTime, endTime);
      setMotionIssues([...motionIssues, result.issue]);
      setShowIssuePopup(false);
    } catch (err) {
      console.error('Failed to add motion issue:', err);
      alert('Failed to add issue');
    } finally {
      setIsAddingIssue(false);
    }
  };

  // Delete issue
  const handleDeleteIssue = async (issueId: string) => {
    if (!confirm('Delete this issue?')) return;
    
    try {
      await deleteMotionIssue(pieceId, issueId);
      setMotionIssues(motionIssues.filter(i => i.id !== issueId));
    } catch (err) {
      console.error('Failed to delete motion issue:', err);
    }
  };

  // Seek to issue start time
  const handleSeekToIssue = (issue: MotionIssue) => {
    const audio = document.getElementById('audio') as HTMLAudioElement;
    if (audio) {
      audio.currentTime = issue.start_time;
    }
  };

  // Open post-playing popup
  const handleOpenPostPopup = () => {
    if (postPlaying) {
      setPostStartTime(formatTime(postPlaying.start_time));
      setPostEndTime(formatTime(postPlaying.end_time));
    } else {
      const currentTime = getCurrentTime();
      setPostStartTime(formatTime(currentTime));
      // Default end_time to audio duration (end of the piece)
      // This represents when the last hand finishes playing
      const audio = document.getElementById('audio') as HTMLAudioElement;
      const defaultEndTime = audio && audio.duration && !isNaN(audio.duration) 
        ? audio.duration 
        : currentTime + 1; // Fallback: 1 second after start if duration not available
      setPostEndTime(formatTime(defaultEndTime));
    }
    setShowPostPopup(true);
  };

  // Set current time to post field
  const handleSetPostTime = (field: 'start' | 'end') => {
    const currentTime = getCurrentTime();
    if (field === 'start') {
      setPostStartTime(formatTime(currentTime));
    } else {
      setPostEndTime(formatTime(currentTime));
    }
  };

  // Submit post-playing segment
  const handleSubmitPost = async () => {
    const startTime = parseTime(postStartTime);
    const endTime = parseTime(postEndTime);
    
    if (startTime === null || endTime === null) {
      alert('Please enter valid time format (mm:ss.ms or ss.ms)');
      return;
    }
    
    if (startTime >= endTime) {
      alert('Start time must be less than end time');
      return;
    }
    
    setIsSavingPost(true);
    try {
      const result = await savePostPlaying(pieceId, startTime, endTime);
      setPostPlaying(result.segment);
      setShowPostPopup(false);
    } catch (err) {
      console.error('Failed to save post-playing segment:', err);
      alert('Failed to save');
    } finally {
      setIsSavingPost(false);
    }
  };

  // Delete post-playing segment
  const handleDeletePost = async () => {
    if (!confirm('Delete post-playing segment?')) return;
    
    try {
      await deletePostPlaying(pieceId);
      setPostPlaying(null);
      setShowPostPopup(false);
    } catch (err) {
      console.error('Failed to delete post-playing segment:', err);
    }
  };

  // Seek to post-playing start
  const handleSeekToPost = () => {
    if (postPlaying) {
      const audio = document.getElementById('audio') as HTMLAudioElement;
      if (audio) {
        audio.currentTime = postPlaying.start_time;
      }
    }
  };

  // Open test segment popup with current time as start
  const handleOpenTestPopup = () => {
    const currentTime = getCurrentTime();
    setTestStartTime(formatTime(currentTime));
    setTestEndTime('');
    setTestNote('');
    setShowTestPopup(true);
  };

  // Set current time to test field
  const handleSetTestTime = (field: 'start' | 'end') => {
    const currentTime = getCurrentTime();
    if (field === 'start') {
      setTestStartTime(formatTime(currentTime));
    } else {
      setTestEndTime(formatTime(currentTime));
    }
  };

  // Submit new test segment
  const handleSubmitTestSegment = async () => {
    const startTime = parseTime(testStartTime);
    const endTime = parseTime(testEndTime);
    
    if (startTime === null || endTime === null) {
      alert('Please enter valid time format (mm:ss.ms or ss.ms)');
      return;
    }
    
    if (startTime >= endTime) {
      alert('Start time must be less than end time');
      return;
    }
    
    setIsAddingTest(true);
    try {
      const result = await addTestSegment(pieceId, startTime, endTime, testNote);
      setTestSegments([...testSegments, result.segment]);
      setShowTestPopup(false);
    } catch (err) {
      console.error('Failed to add test segment:', err);
      alert('Failed to add test segment');
    } finally {
      setIsAddingTest(false);
    }
  };

  // Delete test segment
  const handleDeleteTestSegment = async (segmentId: string) => {
    if (!confirm('Delete this test segment?')) return;
    
    try {
      await deleteTestSegment(pieceId, segmentId);
      setTestSegments(testSegments.filter(s => s.id !== segmentId));
    } catch (err) {
      console.error('Failed to delete test segment:', err);
    }
  };

  // Seek to test segment start time
  const handleSeekToTestSegment = (segment: TestSegment) => {
    const audio = document.getElementById('audio') as HTMLAudioElement;
    if (audio) {
      audio.currentTime = segment.start_time;
    }
  };

  // Go to previous frame
  const goToPrevFrame = async () => {
    const audio = document.getElementById('audio') as HTMLAudioElement;
    if (!audio) return;
    
    // Pause if playing
    if (!audio.paused) {
      audio.pause();
    }
    
    // 현재 프레임 계산 후 이전 프레임으로 이동
    const currentFrameIdx = Math.floor(audio.currentTime * FPS);
    const newFrame = Math.max(0, currentFrameIdx - 1);
    
    // Reset vis.js navigation state
    (window as any).currentNavigationIndex = -1;
    (window as any).selectedFingeringKey = null;
    (window as any).selectedFingeringFrameIndex = -1;
    
    // goToFrame을 사용하여 정확한 프레임으로 이동
    if ((window as any).visualizerControls?.goToFrame) {
      (window as any).visualizerControls.goToFrame(newFrame);
    }
  };

  // Go to next frame
  const goToNextFrame = async () => {
    const audio = document.getElementById('audio') as HTMLAudioElement;
    if (!audio) return;
    
    // Pause if playing
    if (!audio.paused) {
      audio.pause();
    }
    
    // 현재 프레임 계산 후 다음 프레임으로 이동
    const currentFrameIdx = Math.floor(audio.currentTime * FPS);
    const totalFrames = Math.floor((audio.duration || 0) * FPS);
    const newFrame = Math.min(totalFrames - 1, currentFrameIdx + 1);
    
    // Reset vis.js navigation state
    (window as any).currentNavigationIndex = -1;
    (window as any).selectedFingeringKey = null;
    (window as any).selectedFingeringFrameIndex = -1;
    
    // goToFrame을 사용하여 정확한 프레임으로 이동
    if ((window as any).visualizerControls?.goToFrame) {
      (window as any).visualizerControls.goToFrame(newFrame);
    }
  };

  // Handle resume to last annotation position
  const handleResume = () => {
    if (resumeProgress) {
      const audio = document.getElementById('audio') as HTMLAudioElement;
      if (audio) {
        audio.currentTime = resumeProgress.lastTimeSeconds;
        // Trigger sync
        if ((window as any).visualizerControls?.syncAnimationAndAudio) {
          (window as any).visualizerControls.syncAnimationAndAudio();
        }
      }
    }
    setShowResumePopup(false);
  };

  // Handle start from beginning
  const handleStartFromBeginning = () => {
    setShowResumePopup(false);
  };

  // Toggle mute/unmute
  const toggleMute = () => {
    const audio = document.getElementById('audio') as HTMLAudioElement;
    const volumeControl = document.getElementById('volumeControl') as HTMLInputElement;
    
    if (!audio) return;
    
    if (isMuted) {
      // Unmute: restore previous volume
      audio.volume = previousVolume;
      audio.muted = false;
      if (volumeControl) {
        volumeControl.value = previousVolume.toString();
      }
      setIsMuted(false);
    } else {
      // Mute: save current volume and set to 0
      setPreviousVolume(audio.volume);
      audio.volume = 0;
      audio.muted = true;
      if (volumeControl) {
        volumeControl.value = '0';
      }
      setIsMuted(true);
    }
  };

  // Sync mute state when volume changes
  useEffect(() => {
    const volumeControl = document.getElementById('volumeControl') as HTMLInputElement;
    const audio = document.getElementById('audio') as HTMLAudioElement;
    
    if (!volumeControl || !audio) return;
    
    const handleVolumeChange = () => {
      const volume = parseFloat(volumeControl.value);
      audio.volume = volume;
      
      if (volume === 0) {
        setIsMuted(true);
      } else {
        setIsMuted(false);
        setPreviousVolume(volume);
      }
    };
    
    volumeControl.addEventListener('input', handleVolumeChange);
    return () => {
      volumeControl.removeEventListener('input', handleVolumeChange);
    };
  }, []);

  // Sync mute state when M key is pressed (from vis.js)
  useEffect(() => {
    const handleMuteChange = (e: CustomEvent<boolean>) => {
      setIsMuted(e.detail);
      if (!e.detail) {
        // When unmuting, restore previous volume
        const audio = document.getElementById('audio') as HTMLAudioElement;
        if (audio && audio.volume > 0) {
          setPreviousVolume(audio.volume);
        }
      }
    };
    
    window.addEventListener('muteChange', handleMuteChange as EventListener);
    return () => {
      window.removeEventListener('muteChange', handleMuteChange as EventListener);
    };
  }, []);

  useEffect(() => {
    if (scriptLoadedRef.current) return;
    scriptLoadedRef.current = true;

    // Update URL params so vis.js reads the correct pieceId
    const currentUrl = new URL(window.location.href);
    currentUrl.searchParams.set('id', pieceId.toString());
    window.history.replaceState({}, '', currentUrl.toString());

    // Wait until DOM is fully ready
    const loadScript = () => {
      // Check if all required DOM elements exist
      const canvas = document.getElementById('threeCanvas');
      const audio = document.getElementById('audio');
      const playBtn = document.getElementById('playPauseBtn');
      
      if (!canvas || !audio || !playBtn) {
        // Retry if DOM not ready yet
        requestAnimationFrame(loadScript);
        return;
      }

      // Check if script is already loaded (match any version)
      const existingScript = document.querySelector('script[src^="/js/vis.js"]');
      if (existingScript) {
        console.log('vis.js already loaded, checking if initialization needed...');
        // If script already loaded, just try to initialize
        if ((window as any).initVisualizer) {
          (window as any).initVisualizer();
        }
        return;
      }

      console.log('DOM ready, loading vis.js...');
      
      // Load existing vis.js file with cache busting
      const script = document.createElement('script');
      script.type = 'module';
      script.src = `/js/vis.js?v=${Date.now()}`;
      script.onerror = (err) => {
        console.error('Failed to load vis.js:', err);
        const loadingText = document.getElementById('loadingText');
        if (loadingText) {
          loadingText.innerHTML = 'Error loading visualizer. Please refresh.';
        }
      };
      
      script.onload = () => {
        console.log('vis.js loaded successfully');
        // Call initializer after script loads
        setTimeout(() => {
          if ((window as any).initVisualizer) {
            // Check if already initialized (managed in vis.js)
            console.log('Calling initVisualizer...');
            (window as any).initVisualizer();
          } else {
            console.warn('initVisualizer not found, waiting...');
            // Retry if initVisualizer not ready yet
            setTimeout(() => {
              if ((window as any).initVisualizer) {
                (window as any).initVisualizer();
              } else {
                console.error('initVisualizer still not found after delay');
              }
            }, 500);
          }
        }, 200);
      };
      
      document.body.appendChild(script);
    };

    // Add slight delay to ensure DOM is fully rendered
    setTimeout(() => {
      requestAnimationFrame(loadScript);
    }, 100);

    return () => {
      // Cleanup: remove script (match any version)
      const scripts = document.querySelectorAll('script[src^="/js/vis.js"]');
      scripts.forEach(s => s.remove());
      
      // Clean up canvas
      const canvas = document.getElementById('threeCanvas') as HTMLCanvasElement;
      if (canvas) {
        const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
        if (gl) {
          gl.getExtension('WEBGL_lose_context')?.loseContext();
        }
      }
    };
  }, [pieceId]);

  return (
    <div id="visualizer-container">
      <canvas id="threeCanvas"></canvas>
      
      {/* Three.js Loading Overlay */}
      <div className="threejs-loading-overlay" id="threejsLoadingOverlay">
        <div className="threejs-spinner"></div>
        <p className="threejs-loading-title">Loading 3D Scene</p>
        <div className="threejs-progress-container">
          <div className="threejs-progress-bar" id="threejsProgressBar"></div>
        </div>
        <p className="threejs-loading-status" id="threejsLoadingStatus">Initializing...</p>
        <p className="threejs-loading-detail" id="threejsLoadingDetail"></p>
      </div>
      
      {/* AppBar */}
      <div className="app-bar">
        {/* Back Button */}
        <button className="app-bar-button" onClick={() => window.location.href = '/'} title="Back to list">
          <i className="fas fa-arrow-left"></i>
        </button>
        
        {/* Piece Info */}
        <div className="app-bar-info">
          <div className="info-content loading" id="infoContent">
            <h2 className="piece-title">
              <span className="piece-id-badge">[{pieceId}]</span>
              <span id="pieceTitle"></span>
{/* r0 배지 표시하지 않음 - AI prior, 수정 가능 */}
            </h2>
            <p className="piece-composer" id="pieceComposer"></p>
          </div>
          <div className="loading-container" id="loadingContainer">
            <div className="spinner" id="spinner"></div>
            <p className="loading-text" id="loadingText">Initializing...</p>
          </div>
        </div>
        
        {/* Controls - Grouped */}
        <div className="app-bar-controls">
          
          {/* Group 1: View Settings */}
          <div className="control-group view-group">
            <span className="group-label">View</span>
            <button 
              className={`app-bar-control-button ${showFingering ? 'active' : ''}`}
              id="showFingeringBtn"
              onClick={() => {
                const newValue = !showFingering;
                console.log('Fingering button clicked, newValue:', newValue);
                setShowFingering(newValue);
                if ((window as any).visualizerControls) {
                  (window as any).visualizerControls.setShowFingering(newValue);
                }
              }}
              title="Show/Hide Fingering Labels"
            >
              <i className="fas fa-eye"></i>
              <span>Fingering</span>
            </button>
            
            <button 
              className={`app-bar-control-button ${showSkeleton ? 'active' : ''}`}
              id="showSkeletonBtn"
              onClick={() => {
                const newValue = !showSkeleton;
                console.log('Skeleton button clicked, newValue:', newValue);
                setShowSkeleton(newValue);
                if ((window as any).visualizerControls) {
                  (window as any).visualizerControls.setShowSkeleton(newValue);
                }
              }}
              title="Show/Hide Hand Skeleton"
            >
              <i className="fas fa-bone"></i>
              <span>Skeleton</span>
            </button>
            
            <select 
              id="cameraPresetSelect"
              className="app-bar-select"
              onChange={(e) => {
                if ((window as any).visualizerControls) {
                  (window as any).visualizerControls.updateCamera(e.target.value);
                }
              }}
              title="Camera Preset"
            >
              <option value="top">Top</option>
              <option value="left">Left</option>
              <option value="right">Right</option>
            </select>
          </div>

          <div className="control-divider"></div>
          
          {/* Group 2: Annotation Tools */}
          <div className="control-group tools-group">
            <span className="group-label">Tools</span>
            <button 
              className="app-bar-control-button assign-btn"
              onClick={() => {
                if ((window as any).visualizerControls) {
                  (window as any).visualizerControls.showAssignFingeringPopup();
                }
              }}
              title="Assign Fingering"
            >
              <i className="fas fa-plus"></i>
              <span>Assign</span>
            </button>

            <button 
              className={`app-bar-control-button issue-btn ${motionIssues.length > 0 ? 'has-issues' : ''}`}
              onClick={handleOpenIssuePopup}
              title="Report Issue"
            >
              <i className="fas fa-flag"></i>
              <span>Issue</span>
              {motionIssues.length > 0 && (
                <span className="badge">{motionIssues.length}</span>
              )}
            </button>

            {motionIssues.length > 0 && (
              <button 
                className="app-bar-control-button icon-only"
                onClick={() => setShowIssueList(!showIssueList)}
                title="View Issues"
              >
                <i className="fas fa-list"></i>
              </button>
            )}

            <button 
              className={`app-bar-control-button post-btn ${postPlaying ? 'has-post' : ''}`}
              onClick={handleOpenPostPopup}
              title="Mark Post-Playing"
            >
              <i className="fas fa-hand-paper"></i>
              <span>Post</span>
            </button>

            {pieceSplit === 'test' && (
              <>
                <button 
                  className={`app-bar-control-button test-btn ${testSegments.length > 0 ? 'has-tests' : ''}`}
                  onClick={handleOpenTestPopup}
                  title="Mark Test Segment"
                >
                  <i className="fas fa-clipboard-check"></i>
                  <span>Test</span>
                  {testSegments.length > 0 && (
                    <span className="badge">{testSegments.length}</span>
                  )}
                </button>

                {testSegments.length > 0 && (
                  <button 
                    className="app-bar-control-button icon-only"
                    onClick={() => setShowTestList(!showTestList)}
                    title="View Test Segments"
                  >
                    <i className="fas fa-list-check"></i>
                  </button>
                )}
              </>
            )}
            
            <button 
              className="app-bar-control-button"
              id="helpBtn"
              onClick={() => {
                const popup = document.getElementById('keyboardShortcutsPopup');
                if (popup) {
                  popup.classList.toggle('hidden');
                }
              }}
              title="Keyboard Shortcuts"
            >
              <i className="fas fa-question-circle"></i>
              <span>Help</span>
            </button>
          </div>

          <div className="control-divider"></div>

          {/* Group 3: Completion Status */}
          <div className="control-group status-group">
            <span className="group-label">Done</span>
            <button
              className={`status-check-btn ${isFingeringCompleted ? 'completed' : ''} ${isUpdating ? 'updating' : ''}`}
              onClick={handleToggleFingering}
              disabled={isUpdating}
              title={isFingeringCompleted ? 'Fingering ✓' : 'Mark Fingering Done'}
            >
              <i className={`fas ${isFingeringCompleted ? 'fa-check-square' : 'fa-square'}`}></i>
              <span>Fingering</span>
            </button>
            <button
              className={`status-check-btn ${isPostCompleted ? 'completed' : ''} ${isUpdating ? 'updating' : ''}`}
              onClick={handleTogglePost}
              disabled={isUpdating}
              title={isPostCompleted ? 'Post ✓' : 'Mark Post Done'}
            >
              <i className={`fas ${isPostCompleted ? 'fa-check-square' : 'fa-square'}`}></i>
              <span>Post</span>
            </button>
            <span className="status-arrow">→</span>
            <button 
              className={`status-review-btn ${isReview1Completed ? 'completed' : ''} ${isUpdating ? 'updating' : ''}`}
              onClick={handleToggleReview1}
              disabled={isUpdating}
              title={isReview1Completed ? 'R1 ✓' : 'Complete F+P for R1'}
            >
              R1
            </button>
            <button 
              className={`status-review-btn r2 ${isReview2Completed ? 'completed' : ''} ${isUpdating ? 'updating' : ''}`}
              onClick={handleToggleReview2}
              disabled={isUpdating}
              title={isReview2Completed ? 'R2 ✓' : 'Mark R2 Done'}
            >
              R2
            </button>
            <button 
              className={`status-review-btn r3 ${isReview3Completed ? 'completed' : ''} ${isUpdating ? 'updating' : ''}`}
              onClick={handleToggleReview3}
              disabled={isUpdating}
              title={isReview3Completed ? 'R3 ✓' : 'Mark R3 Done'}
            >
              R3
            </button>
          </div>
        </div>
        
        {/* Camera Info Toggle Button */}
        <button className="app-bar-button" onClick={() => {
          if (window.visualizerControls?.toggleCameraInfo) {
            const isVisible = window.visualizerControls.toggleCameraInfo();
            // 버튼 활성화 상태 업데이트
            const btn = document.getElementById('cameraInfoToggleBtn');
            if (btn) {
              btn.classList.toggle('active', isVisible);
            }
          }
        }} id="cameraInfoToggleBtn" title="Toggle Camera Info">
          <i className="fas fa-crosshairs"></i>
        </button>
        
        {/* Fullscreen Toggle Button */}
        <button className="app-bar-button" onClick={() => {
          if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
          } else {
            document.exitFullscreen();
          }
        }} title="Toggle Fullscreen">
          <i className="fas fa-expand"></i>
        </button>
      </div>
      
      {/* Player Controls */}
      <div className="player">
        <button id="playPauseBtn" className="play-pause-btn" aria-label="Play/Pause">
          <i id="playPauseIcon" className="fas fa-play"></i>
        </button>
        
        <div className="time-display" id="timeDisplay">0:00</div>
        
        <div 
          className="progress-bar-container"
          onMouseMove={(e) => {
            if (!audioDuration) return;
            
            const rect = e.currentTarget.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const percent = Math.max(0, Math.min(100, (x / rect.width) * 100));
            const time = (percent / 100) * audioDuration;
            const frameIndex = Math.floor(time * FPS);
            
            // 프레임 번호 툴팁 표시
            setFrameTooltip({ frame: frameIndex, x: e.clientX });
            
            // 현재 프레임의 ambiguous 핑거링 찾기
            const fingering = getFingering(frameIndex);
            const ambiguousFingerings = fingering.filter(f => f.ambiguous === true);
            
            if (ambiguousFingerings.length > 0) {
              const fingerNames: Record<string, string> = {
                'thumb': 'Thumb',
                'index': 'Index',
                'middle': 'Middle',
                'ring': 'Ring',
                'pinky': 'Pinky'
              };
              
              const ambiguousList = ambiguousFingerings.map(f => 
                `${f.hand === 'left' ? 'L' : 'R'}${f.finger} (${fingerNames[f.finger_name]})`
              ).join(', ');
              
              setTooltipContent(`Ambiguous: ${ambiguousList}`);
              setTooltipPosition({ x: e.clientX, y: e.clientY - 30 });
            } else {
              setTooltipContent(null);
            }
          }}
          onMouseLeave={() => {
            setTooltipContent(null);
            setFrameTooltip(null);
          }}
        >
          {/* Frame number tooltip */}
          {frameTooltip && (
            <div 
              className="frame-tooltip"
              style={{ left: frameTooltip.x }}
            >
              {frameTooltip.frame}
            </div>
          )}
          <div id="bufferedBar" className="buffered-bar"></div>
          <div id="playedBar" className="played-bar"></div>
          {/* Issue markers on progress bar */}
          {audioDuration > 0 && motionIssues.map((issue) => (
            <div
              key={issue.id}
              className="issue-marker"
              style={{
                left: `${(issue.start_time / audioDuration) * 100}%`,
                width: `${((issue.end_time - issue.start_time) / audioDuration) * 100}%`,
              }}
              onClick={(e) => {
                e.stopPropagation();
                handleSeekToIssue(issue);
              }}
              title={`Issue: ${formatTime(issue.start_time)} ~ ${formatTime(issue.end_time)}`}
            />
          ))}
          {/* Post-playing marker on progress bar */}
          {audioDuration > 0 && postPlaying && (
            <div
              className="post-marker"
              style={{
                left: `${(postPlaying.start_time / audioDuration) * 100}%`,
                width: `${((postPlaying.end_time - postPlaying.start_time) / audioDuration) * 100}%`,
              }}
              onClick={(e) => {
                e.stopPropagation();
                handleSeekToPost();
              }}
              title={`Post-playing: ${formatTime(postPlaying.start_time)} ~ ${formatTime(postPlaying.end_time)}`}
            />
          )}
          {/* Test segment markers on progress bar */}
          {audioDuration > 0 && testSegments.map((segment) => (
            <div
              key={segment.id}
              className="test-marker"
              style={{
                left: `${(segment.start_time / audioDuration) * 100}%`,
                width: `${((segment.end_time - segment.start_time) / audioDuration) * 100}%`,
              }}
              onClick={(e) => {
                e.stopPropagation();
                handleSeekToTestSegment(segment);
              }}
              title={`Test: ${formatTime(segment.start_time)} ~ ${formatTime(segment.end_time)}${segment.note ? ` (${segment.note})` : ''}`}
            />
          ))}
          {/* Missing fingering markers on progress bar (pressed keys without fingering) */}
          {audioDuration > 0 && missingFingeringSegments.map((segment, idx) => (
            <div
              key={`missing-${segment.keyIndex}-${idx}`}
              className="missing-fingering-marker"
              style={{
                left: `${(segment.start_time / audioDuration) * 100}%`,
                width: `${((segment.end_time - segment.start_time) / audioDuration) * 100}%`,
              }}
              onClick={(e) => {
                e.stopPropagation();
                const audio = document.getElementById('audio') as HTMLAudioElement;
                if (audio) {
                  audio.currentTime = segment.start_time;
                }
              }}
              title={`Missing fingering: Key ${segment.keyIndex} (${formatTime(segment.start_time)} ~ ${formatTime(segment.end_time)})`}
            />
          ))}
          
          {/* Edited fingering markers on progress bar (red lines) */}
          {audioDuration > 0 && editedFingeringFrames.map((frameIdx) => {
            const time = frameIdx / FPS;
            return (
              <div
                key={`edited-${frameIdx}`}
                className="edited-fingering-marker"
                style={{
                  left: `${(time / audioDuration) * 100}%`,
                }}
                onClick={(e) => {
                  e.stopPropagation();
                  const audio = document.getElementById('audio') as HTMLAudioElement;
                  if (audio) {
                    audio.currentTime = time;
                  }
                }}
                title={`Edited fingering at ${formatTime(time)}`}
              />
            );
          })}
          
          {/* Ambiguous fingering markers on progress bar */}
          {audioDuration > 0 && ambiguousFrames.map((frameIdx) => {
            const time = frameIdx / FPS;
            const leftPercent = (time / audioDuration) * 100;
            
            return [
              <div
                key={`ambiguous-marker-${frameIdx}`}
                className="ambiguous-marker"
                style={{ left: `${leftPercent}%` }}
                onClick={(e) => {
                  e.stopPropagation();
                  const audio = document.getElementById('audio') as HTMLAudioElement;
                  if (audio) {
                    audio.currentTime = time;
                  }
                }}
              />,
              <svg
                key={`ambiguous-arrow-${frameIdx}`}
                className="ambiguous-arrow"
                style={{ left: `${leftPercent}%` }}
                width="10"
                height="7"
                viewBox="0 0 10 7"
              >
                <polygon points="5,7 0,0 10,0" fill="#FF9800" />
              </svg>
            ];
          })}
          <input type="range" id="scrubber" className="progress-bar" defaultValue="0" min="0" max="100" step="0.1" aria-label="Progress" />
        </div>
        
        <div className="time-display" id="totalDuration">0:00</div>
        
        <div className="frame-nav-container">
          <button className="frame-nav-btn" onClick={goToPrevFrame} title="Previous frame">
            ⏮
          </button>
          <button className="frame-nav-btn" onClick={goToNextFrame} title="Next frame">
            ⏭
          </button>
        </div>
        
        <div className="frame-display" title="Current frame / Total frames">
          <span className="frame-current">{currentFrame}</span>
          <span className="frame-separator">/</span>
          <span className="frame-total">{Math.floor(audioDuration * FPS)}</span>
        </div>
        
        <div className="volume-container">
          <button 
            className="volume-toggle" 
            id="volumeToggle" 
            aria-label={isMuted ? "Unmute" : "Mute"}
            onClick={toggleMute}
            title={isMuted ? "음소거 해제 (M)" : "음소거 (M)"}
          >
            <i className={`fas ${isMuted ? 'fa-volume-mute' : 'fa-volume-up'} volume-icon`} id="volumeIcon"></i>
          </button>
          <div className="volume-slider-wrapper">
            <input type="range" id="volumeControl" className="volume-bar" min="0" max="1" step="0.1" defaultValue="1" aria-label="Volume" />
          </div>
        </div>
        
        {/* Playback Speed Control */}
        <div className="speed-container">
          <button
            className={`speed-toggle ${showSpeedDropdown ? 'active' : ''}`}
            id="speedToggle"
            aria-label="Playback Speed"
            title="Playback Speed"
            onClick={() => setShowSpeedDropdown(!showSpeedDropdown)}
          >
            <span className="speed-value">{playbackRate}x</span>
          </button>
          {showSpeedDropdown && (
            <div className="speed-dropdown show" id="speedDropdown">
              {[0.1, 0.25, 0.5, 1, 1.5, 2].map((speed) => (
                <button
                  key={speed}
                  className={`speed-option ${playbackRate === speed ? 'active' : ''}`}
                  onClick={() => {
                    setPlaybackRate(speed);
                    setShowSpeedDropdown(false);
                    const audio = document.getElementById('audio') as HTMLAudioElement;
                    if (audio) {
                      audio.playbackRate = speed;
                    }
                    if ((window as any).visualizerControls) {
                      (window as any).visualizerControls.setPlaybackRate(speed);
                    }
                  }}
                >
                  {speed}x
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
      
      <audio id="audio"></audio>
      
      {/* Key Overlay */}
      <div id="keyOverlay" className="key-overlay hidden"></div>
      
      {/* Motion Issue Popup (Floating - no overlay) */}
      {showIssuePopup && (
        <div className="issue-popup floating">
          <div className="issue-popup-header">
            <h3>Report Motion Issue</h3>
            <button className="close-popup-btn" onClick={() => setShowIssuePopup(false)}>
              <i className="fas fa-times"></i>
            </button>
          </div>
          <div className="issue-popup-content">
            <div className="issue-time-row">
              <label>Start:</label>
              <input
                type="text"
                className="issue-time-input"
                value={issueStartTime}
                onChange={(e) => setIssueStartTime(e.target.value)}
                placeholder="0:00.00"
              />
              <button 
                className="issue-now-btn"
                onClick={() => handleSetIssueTime('start')}
                title="Use current time"
              >
                <i className="fas fa-crosshairs"></i> Now
              </button>
            </div>
            <div className="issue-time-row">
              <label>End:</label>
              <input
                type="text"
                className="issue-time-input"
                value={issueEndTime}
                onChange={(e) => setIssueEndTime(e.target.value)}
                placeholder="0:00.00"
              />
              <button 
                className="issue-now-btn"
                onClick={() => handleSetIssueTime('end')}
                title="Use current time"
              >
                <i className="fas fa-crosshairs"></i> Now
              </button>
            </div>
          </div>
          <div className="issue-popup-actions">
            <button className="issue-cancel-btn" onClick={() => setShowIssuePopup(false)}>
              Cancel
            </button>
            <button 
              className="issue-save-btn" 
              onClick={handleSubmitIssue}
              disabled={isAddingIssue}
            >
              {isAddingIssue ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
      )}

      {/* Post-Playing Popup (Floating - no overlay) */}
      {showPostPopup && (
        <div className="post-popup floating">
          <div className="issue-popup-header">
            <h3>Post-Playing Segment</h3>
            <button className="close-popup-btn" onClick={() => setShowPostPopup(false)}>
              <i className="fas fa-times"></i>
            </button>
          </div>
          <div className="issue-popup-content">
            <div className="issue-time-row">
              <label>Start:</label>
              <input
                type="text"
                className="issue-time-input"
                value={postStartTime}
                onChange={(e) => setPostStartTime(e.target.value)}
                placeholder="0:00.00"
              />
              <button 
                className="issue-now-btn"
                onClick={() => handleSetPostTime('start')}
                title="Use current time"
              >
                <i className="fas fa-crosshairs"></i> Now
              </button>
            </div>
            <div className="issue-time-row">
              <label>End:</label>
              <input
                type="text"
                className="issue-time-input"
                value={postEndTime}
                onChange={(e) => setPostEndTime(e.target.value)}
                placeholder="0:00.00"
              />
              <button 
                className="issue-now-btn"
                onClick={() => handleSetPostTime('end')}
                title="Use current time"
              >
                <i className="fas fa-crosshairs"></i> Now
              </button>
            </div>
          </div>
          <div className="issue-popup-actions">
            {postPlaying && (
              <button className="issue-delete-btn" onClick={handleDeletePost}>
                <i className="fas fa-trash"></i> Delete
              </button>
            )}
            <button className="issue-cancel-btn" onClick={() => setShowPostPopup(false)}>
              Cancel
            </button>
            <button 
              className="issue-save-btn post-save-btn" 
              onClick={handleSubmitPost}
              disabled={isSavingPost}
            >
              {isSavingPost ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
      )}

      {/* Issue List Popup */}
      {showIssueList && (
        <div className="issue-list-popup">
          <div className="issue-list-header">
            <h3>Motion Issues ({motionIssues.length})</h3>
            <button className="close-popup-btn" onClick={() => setShowIssueList(false)}>
              <i className="fas fa-times"></i>
            </button>
          </div>
          <div className="issue-list-content">
            {motionIssues.length === 0 ? (
              <p className="no-issues">No issues reported</p>
            ) : (
              motionIssues.map((issue) => (
                <div key={issue.id} className="issue-list-item">
                  <div className="issue-list-item-time" onClick={() => handleSeekToIssue(issue)}>
                    <i className="fas fa-play-circle"></i>
                    {formatTime(issue.start_time)} ~ {formatTime(issue.end_time)}
                  </div>
                  <button 
                    className="issue-delete-btn"
                    onClick={() => handleDeleteIssue(issue.id)}
                    title="Delete issue"
                  >
                    <i className="fas fa-trash"></i>
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Test Segment Popup (Floating - no overlay) */}
      {showTestPopup && (
        <div className="test-popup floating">
          <div className="issue-popup-header">
            <h3>Mark Test Segment</h3>
            <button className="close-popup-btn" onClick={() => setShowTestPopup(false)}>
              <i className="fas fa-times"></i>
            </button>
          </div>
          <div className="issue-popup-content">
            <div className="issue-time-row">
              <label>Start:</label>
              <input
                type="text"
                className="issue-time-input"
                value={testStartTime}
                onChange={(e) => setTestStartTime(e.target.value)}
                placeholder="0:00.00"
              />
              <button 
                className="issue-now-btn"
                onClick={() => handleSetTestTime('start')}
                title="Use current time"
              >
                <i className="fas fa-crosshairs"></i> Now
              </button>
            </div>
            <div className="issue-time-row">
              <label>End:</label>
              <input
                type="text"
                className="issue-time-input"
                value={testEndTime}
                onChange={(e) => setTestEndTime(e.target.value)}
                placeholder="0:00.00"
              />
              <button 
                className="issue-now-btn"
                onClick={() => handleSetTestTime('end')}
                title="Use current time"
              >
                <i className="fas fa-crosshairs"></i> Now
              </button>
            </div>
            <div className="issue-note-row">
              <label>Note:</label>
              <input
                type="text"
                className="issue-note-input"
                value={testNote}
                onChange={(e) => setTestNote(e.target.value)}
                placeholder="Optional description..."
              />
            </div>
          </div>
          <div className="issue-popup-actions">
            <button className="issue-cancel-btn" onClick={() => setShowTestPopup(false)}>
              Cancel
            </button>
            <button 
              className="issue-save-btn test-save-btn" 
              onClick={handleSubmitTestSegment}
              disabled={isAddingTest}
            >
              {isAddingTest ? 'Saving...' : 'Save'}
            </button>
          </div>
        </div>
      )}

      {/* Test Segment List Popup */}
      {showTestList && (
        <div className="test-list-popup">
          <div className="issue-list-header">
            <h3>Test Segments ({testSegments.length})</h3>
            <button className="close-popup-btn" onClick={() => setShowTestList(false)}>
              <i className="fas fa-times"></i>
            </button>
          </div>
          <div className="issue-list-content">
            {testSegments.length === 0 ? (
              <p className="no-issues">No test segments</p>
            ) : (
              testSegments.map((segment) => (
                <div key={segment.id} className="test-list-item">
                  <div className="test-list-item-time" onClick={() => handleSeekToTestSegment(segment)}>
                    <i className="fas fa-play-circle"></i>
                    {formatTime(segment.start_time)} ~ {formatTime(segment.end_time)}
                  </div>
                  {segment.note && (
                    <div className="test-list-item-note" title={segment.note}>
                      {segment.note}
                    </div>
                  )}
                  <button 
                    className="issue-delete-btn"
                    onClick={() => handleDeleteTestSegment(segment.id)}
                    title="Delete segment"
                  >
                    <i className="fas fa-trash"></i>
                  </button>
                </div>
              ))
            )}
          </div>
        </div>
      )}
      
      {/* Resume Annotation Popup */}
      {showResumePopup && resumeProgress && (
        <div className="resume-popup-overlay">
          <div className="resume-popup">
            <div className="resume-popup-header">
              <h3>Resume from last position?</h3>
            </div>
            <div className="resume-popup-content">
              <p>
                Last edited position: <strong>{formatTime(resumeProgress.lastTimeSeconds)}</strong>
                <br />
                <span className="resume-popup-detail">
                  (Frame {resumeProgress.lastFrame})
                </span>
              </p>
            </div>
            <div className="resume-popup-actions">
              <button className="resume-btn-secondary" onClick={handleStartFromBeginning}>
                No, start from beginning
              </button>
              <button className="resume-btn-primary" onClick={handleResume}>
                Yes, resume
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Keyboard Shortcuts Popup */}
      <div id="keyboardShortcutsPopup" className="keyboard-shortcuts-popup hidden">
        <div className="shortcuts-header">
          <h3>Keyboard Shortcuts</h3>
          <button 
            className="close-shortcuts-btn"
            onClick={() => {
              const popup = document.getElementById('keyboardShortcutsPopup');
              if (popup) {
                popup.classList.add('hidden');
              }
            }}
          >
            <i className="fas fa-times"></i>
          </button>
        </div>
        <div className="shortcuts-content">
          <div className="shortcut-item">
            <span className="shortcut-key">Space</span>
            <span className="shortcut-desc">Play / Pause</span>
          </div>
          <div className="shortcut-item">
            <span className="shortcut-key">← / A</span>
            <span className="shortcut-desc">Previous Fingering</span>
          </div>
          <div className="shortcut-item">
            <span className="shortcut-key">→ / D</span>
            <span className="shortcut-desc">Next Fingering</span>
          </div>
          <div className="shortcut-item">
            <span className="shortcut-key">Q</span>
            <span className="shortcut-desc">Previous Frame</span>
          </div>
          <div className="shortcut-item">
            <span className="shortcut-key">E</span>
            <span className="shortcut-desc">Next Frame</span>
          </div>
          <div className="shortcut-item">
            <span className="shortcut-key">F</span>
            <span className="shortcut-desc">Toggle Fingering</span>
          </div>
          <div className="shortcut-item">
            <span className="shortcut-key">ESC</span>
            <span className="shortcut-desc">Deselect Fingering</span>
          </div>
          <div className="shortcut-item">
            <span className="shortcut-key">?</span>
            <span className="shortcut-desc">Show / Hide Help</span>
          </div>
          <div className="shortcut-item">
            <span className="shortcut-key">[ / ]</span>
            <span className="shortcut-desc">Decrease / Increase Speed</span>
          </div>
          <div className="shortcut-item">
            <span className="shortcut-key">M</span>
            <span className="shortcut-desc">Mute / Unmute</span>
          </div>
        </div>
      </div>
      
      {/* Ambiguous Fingering Tooltip - outside player container */}
      {tooltipContent && (
        <div
          ref={tooltipRef}
          className="ambiguous-tooltip"
          style={{
            position: 'fixed',
            left: `${tooltipPosition.x}px`,
            top: `${tooltipPosition.y}px`,
            pointerEvents: 'none',
            zIndex: 10000,
          }}
        >
          {tooltipContent}
        </div>
      )}
    </div>
  );
}
