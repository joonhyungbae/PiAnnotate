import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.153.0/build/three.module.js';
import {OrbitControls} from 'https://cdn.jsdelivr.net/npm/three@0.153.0/examples/jsm/controls/OrbitControls.js';
import {TrackballControls} from 'https://cdn.jsdelivr.net/npm/three@0.153.0/examples/jsm/controls/TrackballControls.js';
import {OBJLoader} from 'https://cdn.jsdelivr.net/npm/three@0.153.0/examples/jsm/loaders/OBJLoader.js';
import {MTLLoader} from 'https://cdn.jsdelivr.net/npm/three@0.153.0/examples/jsm/loaders/MTLLoader.js';
import {RGBELoader} from 'three/addons/loaders/RGBELoader.js';
import {GUI} from 'three/addons/libs/lil-gui.module.min.js';

// 상수 정의
const FPS = 60000 / 1001;
const NUM_PIANO_KEYS = 88;
const NUM_VERTICES = 778;
const BLACK_KEY_PATTERN = [1, 3, 6, 8, 10];
const FINGER_TIP_INDICES = {
    thumb: 4,
    index: 8,
    middle: 12,
    ring: 16,
    pinky: 20
};

// 전역 상태 변수
const urlParams = new URLSearchParams(window.location.search);
const pieceId = parseInt(urlParams.get('id')) || 0;
const audioElement = document.getElementById("audio");

let totalFrames = 0;
let totalDuration = 0;
let loadedFrames = 0;
let keyObjs = {};
let preloadedFrames = {};  // 메모리 캐시 (유일한 프레임 저장소)
let fingeringData = {};
let originalFingeringData = {}; // 서버에서 받은 원본 핑거링 데이터 (편집 여부 판단용)
let editedFingeringData = {}; // 편집된 핑거링 데이터
let isAiAnnotation = false; // AI 어노테이션 (r0) 여부 - AI가 보정한 prior, 어노테이터가 수정 가능
let midiNotes = []; // MIDI 노트 정보: [{key_idx, onset_frame, offset_frame}, ...]
let isEditMode = false; // 편집 모드 상태
let raycaster = null; // Raycaster 인스턴스
let mouse = new THREE.Vector2(); // 마우스 좌표
let currentCamera = null; // 현재 카메라 참조
let currentRenderer = null; // 현재 렌더러 참조
let editingFingeringFile = null; // 편집 중인 파일명 (fingering_edited.pkl)
let fingerTipSprites = {}; // 손가락 끝별 스프라이트 (전역으로 이동)
let spriteToFingeringMap = new Map(); // 스프라이트와 핑거링 정보 매핑
let editListenersSetup = false; // 이벤트 리스너 설정 여부
let currentFrameIndex = 0; // 현재 프레임 인덱스 (전역 변수로 선언)
let selectedFingeringKey = null; // 현재 선택된 핑거링 (hand_fingerName 형식)
let selectedFingeringFrameIndex = -1; // 선택된 핑거링의 프레임 인덱스
let currentNavigationIndex = -1; // 전역 핑거링 네비게이션 인덱스
let isNavigatingToFrame = false; // 핑거링 네비게이션 중인지 여부 (syncAnimationAndAudio 무시용)
let keyboardListenerSetup = false; // 키보드 이벤트 리스너 설정 여부
let keyboardListener = null; // 키보드 이벤트 리스너 참조

// Assign 팝업 선택 상태 (전역)
let assignPopupState = {
    selectedHand: null,
    selectedFingerNumber: null,
    selectedKeyIndex: null,
    selectedOnsetFrame: null,
    selectedOffsetFrame: null,
    ambiguousState: false,
    applyFromCurrentFrame: false
};

// 프레임 fetch 동시성 제어
const pendingFetches = new Map(); // frameIndex -> Promise (진행 중인 fetch)
const MAX_CONCURRENT_FETCHES = 24; // localhost에서 높은 동시성 지원
const MAX_RETRIES = 3; // 재시도 횟수
let activeFetchCount = 0;
const fetchQueue = []; // 대기 중인 fetch 요청

/**
 * 피아노 키가 검은 건반인지 확인합니다.
 */
function isBlackKey(keyIndex) {
    return BLACK_KEY_PATTERN.includes((keyIndex + 8) % 12);
}

/**
 * 단일 프레임을 서버에서 가져옵니다 (range 요청 사용, 동시성 제어 + 중복 방지).
 * 단일 프레임 요청 대신 작은 range를 요청하여 효율성을 높입니다.
 */
async function fetchSingleFrame(frameIndex) {
    // 이미 캐시에 있으면 바로 반환
    if (preloadedFrames[frameIndex]) {
        return preloadedFrames[frameIndex];
    }
    
    // 이미 진행 중인 fetch가 있으면 그 Promise 반환
    if (pendingFetches.has(frameIndex)) {
        return pendingFetches.get(frameIndex);
    }
    
    // 작은 range로 요청 (단일 프레임보다 효율적)
    const rangeStart = Math.max(0, frameIndex - 10);
    const rangeEnd = Math.min(totalFrames - 1, frameIndex + 10);
    
    // fetch 실행 함수
    const doFetch = async () => {
        const frames = await fetchFrameRange(rangeStart, rangeEnd);
        
        // 모든 프레임을 메모리 캐시에 저장
        for (const frame of frames) {
            preloadedFrames[frame.frame_idx] = frame;
        }
        
        // 요청한 프레임 반환
        return preloadedFrames[frameIndex];
    };
    
    // 동시 fetch 수 제한
    const fetchPromise = new Promise(async (resolve, reject) => {
        // 동시 fetch 수가 너무 많으면 대기
        while (activeFetchCount >= MAX_CONCURRENT_FETCHES) {
            await new Promise(r => setTimeout(r, 50));
        }
        
        activeFetchCount++;
        try {
            const frameData = await doFetch();
            if (!frameData) {
                throw new Error(`Frame ${frameIndex} not found in range ${rangeStart}-${rangeEnd}`);
            }
            resolve(frameData);
        } catch (e) {
            reject(e);
        } finally {
            activeFetchCount--;
            pendingFetches.delete(frameIndex);
        }
    });
    
    pendingFetches.set(frameIndex, fetchPromise);
    return fetchPromise;
}

// 볼륨 컨트롤 초기화
function initializeVolumeControl() {
    const volumeControl = document.getElementById('volumeControl');
    const volumeIcon = document.getElementById('volumeIcon');
    const audio = document.getElementById('audio');
    
    if (!volumeControl || !volumeIcon || !audio) return;
    
    volumeControl.addEventListener('input', () => {
        audio.volume = volumeControl.value;
        if (audio.volume === 0) {
            volumeIcon.className = 'fas fa-volume-mute';
        } else if (audio.volume <= 0.5) {
            volumeIcon.className = 'fas fa-volume-down';
        } else {
            volumeIcon.className = 'fas fa-volume-up';
        }
    });
}

initializeVolumeControl();
if (audioElement) {
    audioElement.src = `/audio/${pieceId}`;
}
console.log("Piece ID:", pieceId);

/**
 * 메타데이터를 서버에서 가져옵니다.
 */
async function fetchMetaData() {
    const response = await fetch(`/metadata/${pieceId}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch metadata: ${response.status}`);
    }
    return await response.json();
}

/**
 * MANO 모델의 면 데이터를 가져옵니다.
 */
async function fetchFacesData() {
    const response = await fetch('/mano_faces_data');
    if (!response.ok) {
        throw new Error(`Failed to fetch faces data: ${response.status}`);
    }
    return await response.json();
}

/**
 * 메모리 캐시에서 프레임을 가져옵니다.
 */
function getPreloadedFrame(frameIndex) {
    return preloadedFrames[frameIndex];
}

/**
 * 프레임이 메모리에 로드되었는지 확인합니다.
 */
function hasFrame(frameIndex) {
    return preloadedFrames[frameIndex] !== undefined;
}

/**
 * 시간을 분:초 형식으로 포맷합니다.
 */
function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}m ${secs}s`;
}

/**
 * 중앙 로딩 오버레이의 상태를 업데이트합니다.
 */
function updateLoadingOverlay(status, detail = '', progress = -1) {
    const statusEl = document.getElementById('threejsLoadingStatus');
    const detailEl = document.getElementById('threejsLoadingDetail');
    const progressBar = document.getElementById('threejsProgressBar');
    const bufferedBar = document.getElementById('bufferedBar');
    
    if (statusEl) {
        statusEl.textContent = status;
    }
    
    if (detailEl) {
        detailEl.textContent = detail;
    }
    
    if (progressBar && progress >= 0) {
        progressBar.style.width = `${Math.min(100, progress)}%`;
    }
    
    // 플레이어 버퍼 바도 업데이트
    if (bufferedBar && progress >= 0) {
        bufferedBar.style.width = `${Math.min(100, progress)}%`;
        bufferedBar.style.backgroundColor = '#4CAF50';
    }
}

/**
 * AppBar 메시지 업데이트 (레거시 호환)
 */
function updateAppBarMessage(message) {
    updateLoadingOverlay(message);
}

/**
 * 로딩 진행 상태를 업데이트합니다.
 * 프레임 로딩은 전체 진행률의 20%~95% 구간을 차지합니다.
 */
function updateLoadingProgress(current, total, startTime) {
    const framePercentage = (current / total) * 100;
    // 프레임 로딩: 20% ~ 95% 구간 (75% 범위)
    const overallProgress = 20 + (framePercentage * 0.75);
    const elapsedTime = (Date.now() - startTime) / 1000;
    
    let status = '';
    let detail = '';
    
    if (current > 0 && elapsedTime > 0) {
        const loadRate = current / elapsedTime;
        const framesRemaining = total - current;
        const timeRemaining = loadRate > 0 ? framesRemaining / loadRate : 0;
        
        status = `Downloading frames: ${current.toLocaleString()} / ${total.toLocaleString()} (${Math.floor(framePercentage)}%)`;
        detail = `ETA: ${formatTime(timeRemaining)} • ${Math.floor(loadRate)} frames/s`;
    } else {
        status = `Downloading frames: 0 / ${total.toLocaleString()}`;
        detail = 'Starting parallel download...';
    }
    
    updateLoadingOverlay(status, detail, overallProgress);
}

// Lazy loading configuration (최적화)
const INITIAL_LOAD_FRAMES = 180;  // Load first 3 seconds (at 60fps)
const CHUNK_SIZE = 30;            // Load 0.5 seconds at a time (~3MB per chunk, more stable)
const PRELOAD_AHEAD = 300;        // Preload 5 seconds ahead of current position

// 프레임 로딩 상태
let isLoadingChunk = false;
let backgroundLoadingActive = false;

/**
 * 프레임이 메모리에 로드되었는지 확인합니다.
 */
function isFrameLoaded(frameIdx) {
    return preloadedFrames[frameIdx] !== undefined;
}

// Backend URL - Vite 프록시를 우회하여 Flask 서버에 직접 연결
const BACKEND_URL = window.location.port === '3000' ? 'http://localhost:8080' : '';

async function fetchFrameRange(start, end) {
    const response = await fetch(`${BACKEND_URL}/mano_vertices_data/${pieceId}/frames/${start}-${end}`);
    if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    const frames = await response.json();
    
    return frames;
}

/**
 * 메쉬 데이터를 Lazy loading 방식으로 가져옵니다 (최적화된 Range 요청 + gzip).
 */
async function fetchMeshData() {
    console.log("Starting mesh data fetch (optimized range requests + gzip)...");
    updateLoadingOverlay("Fetching scene information...", "Connecting to server");
    const startTime = Date.now();
    
    try {
        // Get mesh info
        const infoResponse = await fetch(`${BACKEND_URL}/mano_vertices_data/${pieceId}/info`);
        if (!infoResponse.ok) {
            throw new Error(`Failed to get mesh info: ${infoResponse.status}`);
        }
        const info = await infoResponse.json();
        totalFrames = info.n_frames;
        console.log(`Total frames: ${totalFrames}`);
        
        updateLoadingOverlay(`Found ${totalFrames.toLocaleString()} frames`, "Starting parallel download...", 0);
        await fetchMeshDataLazy(startTime);
        
    } catch (error) {
        console.error("Error fetching mesh data:", error);
        updateLoadingOverlay(`Error: ${error.message}`, "Failed to load data");
        throw error;
    }
}

/**
 * 메쉬 데이터를 병렬로 로드합니다 (모든 프레임 로드 완료 후 재생 가능).
 */
async function fetchMeshDataLazy(startTime) {
    console.log(`Loading all ${totalFrames} frames in parallel...`);
    updateLoadingProgress(0, totalFrames, startTime);
    
    // 모든 청크 범위 계산
    const chunks = [];
    for (let start = 0; start < totalFrames; start += CHUNK_SIZE) {
        const end = Math.min(start + CHUNK_SIZE - 1, totalFrames - 1);
        chunks.push({ start, end });
    }
    
    const totalChunks = chunks.length;
    console.log(`Total chunks: ${totalChunks}, parallel: ${MAX_CONCURRENT_FETCHES}, chunk size: ${CHUNK_SIZE}`);
    
    // 병렬로 청크들을 로드 (동시성 제어 + 재시도)
    let completedChunks = 0;
    const failedChunks = [];
    
    const loadChunk = async (chunk, retryCount = 0) => {
        try {
            const frames = await fetchFrameRange(chunk.start, chunk.end);
            
            // 메모리 캐시에 저장
            for (const frame of frames) {
                const frameIdx = frame.frame_idx;
                if (!preloadedFrames[frameIdx]) {
                    loadedFrames++;
                }
                preloadedFrames[frameIdx] = frame;
            }
            
            completedChunks++;
            
            // 진행 상황 업데이트 (프로그레스 바 + ETA)
            updateLoadingProgress(loadedFrames, totalFrames, startTime);
            
        } catch (e) {
            if (retryCount < MAX_RETRIES) {
                console.warn(`Retrying frames ${chunk.start}-${chunk.end} (attempt ${retryCount + 1}/${MAX_RETRIES})...`);
                await new Promise(r => setTimeout(r, 500 * (retryCount + 1))); // 점진적 대기
                await loadChunk(chunk, retryCount + 1);
            } else {
                console.error(`Failed to load frames ${chunk.start}-${chunk.end} after ${MAX_RETRIES} retries:`, e);
                failedChunks.push(chunk);
                completedChunks++;
            }
        }
    };
    
    // 동시성 제어: MAX_CONCURRENT_FETCHES 개씩 병렬 처리
    for (let i = 0; i < chunks.length; i += MAX_CONCURRENT_FETCHES) {
        const batch = chunks.slice(i, i + MAX_CONCURRENT_FETCHES);
        await Promise.all(batch.map(c => loadChunk(c, 0)));
    }
    
    // 실패한 청크가 있으면 마지막으로 한 번 더 재시도
    if (failedChunks.length > 0) {
        console.warn(`${failedChunks.length} chunks failed. Final retry...`);
        updateLoadingOverlay(
            `Retrying ${failedChunks.length} failed chunks...`,
            'Please wait...',
            (loadedFrames / totalFrames) * 100
        );
        
        // 하나씩 순차적으로 재시도
        for (const chunk of failedChunks) {
            try {
                await new Promise(r => setTimeout(r, 1000)); // 1초 대기
                const frames = await fetchFrameRange(chunk.start, chunk.end);
                for (const frame of frames) {
                    const frameIdx = frame.frame_idx;
                    if (!preloadedFrames[frameIdx]) {
                        loadedFrames++;
                    }
                    preloadedFrames[frameIdx] = frame;
                }
                console.log(`Successfully recovered frames ${chunk.start}-${chunk.end}`);
            } catch (e) {
                console.error(`Final retry failed for frames ${chunk.start}-${chunk.end}:`, e);
            }
        }
    }
    
    const fetchTime = Date.now() - startTime;
    const successRate = Math.round((loadedFrames / totalFrames) * 100);
    const avgSpeed = Math.round(loadedFrames / (fetchTime / 1000));
    console.log(`Loaded ${loadedFrames}/${totalFrames} frames (${successRate}%) in ${(fetchTime / 1000).toFixed(1)}s (${avgSpeed} fps, parallel: ${MAX_CONCURRENT_FETCHES})`);
    
    // 로딩 완료 상태 표시
    if (loadedFrames >= totalFrames * 0.95) {
        updateLoadingOverlay(
            `Complete! ${loadedFrames.toLocaleString()} frames`,
            `Loaded in ${(fetchTime / 1000).toFixed(1)}s • ${avgSpeed} fps`,
            100
        );
    } else {
        updateLoadingOverlay(
            `${successRate}% loaded (${loadedFrames.toLocaleString()} / ${totalFrames.toLocaleString()})`,
            'Some frames may be missing',
            successRate
        );
    }
    
    // 버퍼 바 색상 변경
    const bufferedBar = document.getElementById('bufferedBar');
    if (bufferedBar) {
        bufferedBar.style.backgroundColor = '#2196F3';
    }
}


/**
 * 특정 프레임 주변을 우선 로드합니다 (시간 이동 시 사용).
 */
async function ensureFramesLoaded(centerFrame) {
    const start = Math.max(0, centerFrame - PRELOAD_AHEAD);
    const end = Math.min(centerFrame + PRELOAD_AHEAD, totalFrames - 1);
    
    // Check if already loaded
    let allLoaded = true;
    for (let i = start; i <= end; i++) {
        if (!isFrameLoaded(i)) {
            allLoaded = false;
            break;
        }
    }
    
    if (allLoaded) return;
    
    // Find unloaded ranges
    let unloadedStart = null;
    for (let i = start; i <= end; i++) {
        if (!isFrameLoaded(i) && unloadedStart === null) {
            unloadedStart = i;
        } else if (isFrameLoaded(i) && unloadedStart !== null) {
            // Load this range
            await loadFrameRangeWithPriority(unloadedStart, i - 1);
            unloadedStart = null;
        }
    }
    
    if (unloadedStart !== null) {
        await loadFrameRangeWithPriority(unloadedStart, end);
    }
}

async function loadFrameRangeWithPriority(start, end) {
    console.log(`Priority loading frames ${start}-${end}...`);
    
    try {
        const frames = await fetchFrameRange(start, end);
        
        for (const frame of frames) {
            const frameIdx = frame.frame_idx;
            if (!preloadedFrames[frameIdx]) {
                    loadedFrames++;
            }
            preloadedFrames[frameIdx] = frame;
        }
        
                } catch (e) {
        console.error(`Error priority loading frames ${start}-${end}:`, e);
    }
}

/**
 * MIDI 노트 정보를 서버에서 가져옵니다.
 */
async function fetchMidiNotes() {
    const url = `${BACKEND_URL}/midi_notes/${pieceId}`;
    
    try {
        const response = await fetch(url);
        if (!response.ok) {
            console.warn(`MIDI notes not available for piece ${pieceId}: ${response.status}`);
            midiNotes = [];
            return;
        }
        
        const data = await response.json();
        midiNotes = data.notes || [];
        console.log(`Loaded ${midiNotes.length} MIDI notes`);
    } catch (e) {
        console.error(`Failed to fetch MIDI notes for piece ${pieceId}:`, e);
        midiNotes = [];
    }
}

/**
 * 특정 프레임과 키에 대해 해당하는 MIDI 노트를 찾습니다.
 * @param {number} frameIndex - 프레임 인덱스
 * @param {number} keyIndex - 키 인덱스
 * @returns {{key_idx: number, onset_frame: number, offset_frame: number} | null}
 */
function findMidiNoteForFrame(frameIndex, keyIndex) {
    // 해당 키와 프레임이 포함된 MIDI 노트 찾기
    for (const note of midiNotes) {
        if (note.key_idx === keyIndex && 
            note.onset_frame <= frameIndex && 
            frameIndex <= note.offset_frame) {
            return note;
        }
    }
    return null;
}

/**
 * pressed_keys에서 현재 프레임의 노트 범위를 찾습니다.
 * 키가 눌리기 시작한 지점(onset)과 떼어지는 지점(offset)을 찾습니다.
 * midiNotes가 로드되지 않아도 동작합니다.
 * @param {number} frameIndex - 현재 프레임 인덱스
 * @param {number} keyIndex - 키 인덱스 (0-87)
 * @returns {{onset: number, offset: number, keyIndex: number} | null}
 */
function findNoteRangeFromPressedKeys(frameIndex, keyIndex) {
    const frameData = preloadedFrames[frameIndex];
    if (!frameData || !frameData.pressed_keys) {
        console.log(`No pressed_keys data for frame ${frameIndex}`);
        return null;
    }
    
    // 현재 프레임에서 해당 키가 눌려있는지 확인
    if (frameData.pressed_keys[keyIndex] <= 0.5) {
        console.log(`Key ${keyIndex} is not pressed at frame ${frameIndex}`);
        return null;
    }
    
    // onset 찾기: 현재 프레임에서 뒤로 가면서 키가 눌리기 시작한 지점
    let onset = frameIndex;
    for (let i = frameIndex - 1; i >= 0; i--) {
        const fd = preloadedFrames[i];
        if (!fd || !fd.pressed_keys || fd.pressed_keys[keyIndex] <= 0.5) {
            onset = i + 1;
            break;
        }
        if (i === 0) {
            onset = 0;
        }
    }
    
    // offset 찾기: 현재 프레임에서 앞으로 가면서 키가 떼어지는 지점
    let offset = frameIndex;
    const validFrames = Object.keys(preloadedFrames).map(Number).filter(n => !isNaN(n));
    const maxFrame = validFrames.length > 0 ? Math.max(...validFrames) : frameIndex;
    
    for (let i = frameIndex + 1; i <= maxFrame; i++) {
        const fd = preloadedFrames[i];
        if (!fd || !fd.pressed_keys || fd.pressed_keys[keyIndex] <= 0.5) {
            offset = i - 1;
            break;
        }
        if (i === maxFrame) {
            offset = maxFrame;
        }
    }
    
    console.log(`Found note range from pressed_keys: key ${keyIndex}, frames ${onset}~${offset}`);
    return { onset, offset, keyIndex };
}

/**
 * 핑거링 데이터를 gzip 압축된 JSON으로 가져와 메모리에 저장합니다.
 */
async function fetchFingeringData() {
    const url = `${BACKEND_URL}/fingering_data/${pieceId}`;
    const startTime = Date.now();
    
    console.log("Loading fingering data...");
    
    let response;
    try {
        response = await fetch(url);
    } catch (e) {
        console.error(`Failed to fetch fingering data for piece ${pieceId}:`, e);
        return;
    }

    if (!response.ok) {
        console.warn(`Fingering data not available for piece ${pieceId}: ${response.status}`);
        return;
    }

    try {
        // Browser automatically decompresses gzip (if compressed)
        const frames = await response.json();
        const fetchTime = Date.now() - startTime;
        console.log(`Fetched fingering data in ${(fetchTime / 1000).toFixed(1)}s`);

        // Check if this is AI annotation (r0) - AI prior, editable by annotator
        if (frames.length > 0 && frames[0].is_ai_annotation) {
            isAiAnnotation = true;
            console.log("🤖 AI Annotation (r0) detected - AI prior available for editing");
        } else {
            isAiAnnotation = false;
        }

        // Process frames
        for (const frameData of frames) {
                if (frameData.error) {
                    console.warn("Error in fingering data:", frameData.error);
                    continue;
                }
                
                // 프레임별로 핑거링 데이터 저장
                const frameIdx = frameData.frame_idx;
                fingeringData[frameIdx] = frameData.fingering || [];
                
                // 원본 핑거링 데이터 저장 (서버에서 original_fingering 필드가 있으면 사용)
                if (frameData.original_fingering !== undefined) {
                    originalFingeringData[frameIdx] = frameData.original_fingering;
                } else {
                    // original_fingering이 없으면 현재 fingering이 원본
                    originalFingeringData[frameIdx] = frameData.fingering || [];
        }
    }

    console.log(`Loaded fingering data for ${Object.keys(fingeringData).length} frames`);
    const hasEdits = Object.keys(originalFingeringData).some(idx => 
        JSON.stringify(fingeringData[idx]) !== JSON.stringify(originalFingeringData[idx])
    );
    console.log(`Has edited fingering data: ${hasEdits}`);
    
    const frameCount = Object.keys(fingeringData).length;
    if (frameCount > 0) {
        console.log(`Fingering data loaded: ${frameCount} frames`);
        }
    } catch (e) {
        console.error("Error parsing fingering data:", e);
        // 네트워크 에러나 파싱 에러 상세 정보 출력
        if (e instanceof TypeError && e.message.includes('Failed to fetch')) {
            console.error("Network error - check server connection and Content-Length headers");
        } else if (e instanceof SyntaxError) {
            console.error("JSON parsing error - response may be incomplete or corrupted");
        }
    }
}

/**
 * 렌더링을 위해 프레임을 가져옵니다.
 */
function fetchFrameForRendering(frameIndex) {
    return preloadedFrames[frameIndex] || null;
}


async function main() {
    console.log("Starting application");
    updateLoadingOverlay("Initializing...", "Setting up 3D environment", 0);
    // Setup the scene, camera, and renderer
    const scene = new THREE.Scene();
    const gui = new GUI();
    const camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 0.1, 1000);
    const canvas = document.getElementById('threeCanvas');
    if (!canvas) {
        console.error("Canvas element not found!");
        return;
    }
    const renderer = new THREE.WebGLRenderer({canvas: canvas, antialias: true});
    renderer.setSize(window.innerWidth, window.innerHeight);
    // React 컴포넌트에서 이미 canvas가 있으므로 appendChild 불필요


    const pmremGenerator = new THREE.PMREMGenerator(renderer);

    const hdriLoader = new RGBELoader()
    hdriLoader.load("resources/envmap.hdr", function (texture) {
        const envMap = pmremGenerator.fromEquirectangular(texture).texture;
        texture.dispose();
        scene.environment = envMap;
        scene.background = envMap;
        scene.backgroundBlurriness = 0.3;
    }, undefined, function (error) {
        console.warn("Failed to load HDR environment map:", error);
        console.warn("Continuing without environment map...");
        // Set a simple background color instead
        scene.background = new THREE.Color(0x222222);
    });

    const renderingParams = {
        handColor: 0xefceb9,
        showFingering: true,
        showSkeleton: false,
        skeletonGrayness: 0.6,
    }
    
    // MANO hand joint connections (bone structure)
    const JOINT_CONNECTIONS = [
        // Wrist to thumb
        [0, 1], [1, 2], [2, 3], [3, 4],
        // Wrist to index
        [0, 5], [5, 6], [6, 7], [7, 8],
        // Wrist to middle
        [0, 9], [9, 10], [10, 11], [11, 12],
        // Wrist to ring
        [0, 13], [13, 14], [14, 15], [15, 16],
        // Wrist to pinky
        [0, 17], [17, 18], [18, 19], [19, 20]
    ];
    
    // Fingertip joint indices
    const FINGERTIP_JOINTS = [4, 8, 12, 16, 20];
    const WRIST_JOINT = 0;
    
    // 스켈레톤 시각화를 위한 그룹
    const leftSkeletonGroup = new THREE.Group();
    const rightSkeletonGroup = new THREE.Group();
    scene.add(leftSkeletonGroup);
    scene.add(rightSkeletonGroup);
    
    // 스켈레톤 뼈(선) 및 관절(구) 객체 배열
    const leftSkeletonBones = [];
    const rightSkeletonBones = [];
    const leftSkeletonJoints = [];
    const rightSkeletonJoints = [];
    
    // 스켈레톤 재질 생성
    function createSkeletonMaterials() {
        const grayValue = Math.round(renderingParams.skeletonGrayness * 255);
        const grayColor = (grayValue << 16) | (grayValue << 8) | grayValue;
        
        return {
            bone: new THREE.LineBasicMaterial({
                color: grayColor,
                transparent: true,
                opacity: 0.8,
                depthTest: false
            }),
            joint: new THREE.MeshBasicMaterial({
                color: grayColor,
                transparent: true,
                opacity: 0.8,
                depthTest: false
            }),
            fingertip: new THREE.MeshStandardMaterial({
                color: 0x64ffda, // Cyan glow for fingertips
                emissive: 0x64ffda,
                emissiveIntensity: 2.5,
                transparent: true,
                opacity: 1.0,
                depthTest: false
            }),
            wrist: new THREE.MeshBasicMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0.8,
                depthTest: false
            })
        };
    }
    
    // 스켈레톤 객체 초기화
    function initSkeletonObjects() {
        const materials = createSkeletonMaterials();
        const jointSize = 0.003;
        const jointGeometry = new THREE.SphereGeometry(jointSize, 12, 12);
        
        // 양손에 대해 스켈레톤 객체 생성
        [{ group: leftSkeletonGroup, bones: leftSkeletonBones, joints: leftSkeletonJoints },
         { group: rightSkeletonGroup, bones: rightSkeletonBones, joints: rightSkeletonJoints }].forEach(({ group, bones, joints }) => {
            // 뼈(선) 객체 생성
            for (let i = 0; i < JOINT_CONNECTIONS.length; i++) {
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(6); // 2 points * 3 coords
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                const line = new THREE.Line(geometry, materials.bone.clone());
                line.renderOrder = 100;
                line.visible = false;
                group.add(line);
                bones.push(line);
            }
            
            // 관절(구) 객체 생성
            for (let i = 0; i < 21; i++) {
                const isFingertip = FINGERTIP_JOINTS.includes(i);
                const isWrist = i === WRIST_JOINT;
                
                let material;
                if (isFingertip) {
                    material = materials.fingertip.clone();
                } else if (isWrist) {
                    material = materials.wrist.clone();
                } else {
                    material = materials.joint.clone();
                }
                
                const mesh = new THREE.Mesh(jointGeometry, material);
                mesh.renderOrder = 100;
                mesh.visible = false;
                group.add(mesh);
                joints.push({ mesh, isFingertip, isWrist });
            }
        });
    }
    
    // 스켈레톤 초기화 호출
    initSkeletonObjects();
    
    /**
     * 스켈레톤을 업데이트합니다.
     */
    function updateSkeletonForFrame(frameData) {
        // 스켈레톤 숨기기
        if (!renderingParams.showSkeleton || !frameData.left_joints || !frameData.right_joints) {
            leftSkeletonBones.forEach(bone => bone.visible = false);
            rightSkeletonBones.forEach(bone => bone.visible = false);
            leftSkeletonJoints.forEach(j => j.mesh.visible = false);
            rightSkeletonJoints.forEach(j => j.mesh.visible = false);
            return;
        }
        
        const leftJoints = frameData.left_joints.map(v => parseFloat(v));
        const rightJoints = frameData.right_joints.map(v => parseFloat(v));
        
        // 관절 위치 파싱 (21 joints × 3 coords = 63 values)
        function parseJointPositions(joints) {
            const positions = [];
            for (let i = 0; i < 21; i++) {
                positions.push([
                    joints[i * 3],
                    joints[i * 3 + 1],
                    joints[i * 3 + 2]
                ]);
            }
            return positions;
        }
        
        // 양손에 대해 스켈레톤 업데이트
        [{ joints: leftJoints, bones: leftSkeletonBones, jointMeshes: leftSkeletonJoints },
         { joints: rightJoints, bones: rightSkeletonBones, jointMeshes: rightSkeletonJoints }].forEach(({ joints, bones, jointMeshes }) => {
            const positions = parseJointPositions(joints);
            
            // 뼈(선) 업데이트
            JOINT_CONNECTIONS.forEach((connection, boneIdx) => {
                const [startIdx, endIdx] = connection;
                const start = positions[startIdx];
                const end = positions[endIdx];
                
                if (!start || !end) {
                    bones[boneIdx].visible = false;
                    return;
                }
                
                const positionAttr = bones[boneIdx].geometry.attributes.position;
                positionAttr.array[0] = start[0];
                positionAttr.array[1] = start[1];
                positionAttr.array[2] = start[2];
                positionAttr.array[3] = end[0];
                positionAttr.array[4] = end[1];
                positionAttr.array[5] = end[2];
                positionAttr.needsUpdate = true;
                bones[boneIdx].visible = true;
            });
            
            // 관절(구) 업데이트
            positions.forEach((pos, jointIdx) => {
                if (!pos) {
                    jointMeshes[jointIdx].mesh.visible = false;
                    return;
                }
                
                jointMeshes[jointIdx].mesh.position.set(pos[0], pos[1], pos[2]);
                jointMeshes[jointIdx].mesh.visible = true;
            });
        });
    }
    // gui.addColor(renderingParams, 'handColor').onChange((value) => {
    //     leftMesh.material.color.setHex(value);
    //     rightMesh.material.color.setHex(value);
    //     leftMaterial.material.needsUpdate = true;
    //     rightMaterial.material.needsUpdate = true;
    // });
    
    // GUI 숨기기
    gui.hide();
    
    // 전역으로 노출하여 AppBar에서 접근 가능하도록
    window.visualizerControls = {
        renderingParams: renderingParams,
        setShowFingering: (value) => {
            renderingParams.showFingering = value;
            // 현재 프레임의 핑거링 스프라이트 즉시 업데이트
            const frameData = preloadedFrames[currentFrameIndex];
            if (frameData) {
                updateFingeringSprites(frameData, currentFrameIndex);
            }
        },
        getShowFingering: () => renderingParams.showFingering,
        setShowSkeleton: (value) => {
            renderingParams.showSkeleton = value;
            // 현재 프레임의 스켈레톤 즉시 업데이트
            const frameData = preloadedFrames[currentFrameIndex];
            if (frameData) {
                updateSkeletonForFrame(frameData);
            }
        },
        getShowSkeleton: () => renderingParams.showSkeleton,
        getCurrentFrameIndex: () => currentFrameIndex,
        syncAnimationAndAudio: syncAnimationAndAudio,
        moveFrame: (direction) => {
            const currentTime = audio.currentTime || 0;
            const currentFrame = Math.floor(currentTime * FPS);
            const newFrame = Math.max(0, Math.min(totalFrames - 1, currentFrame + direction));
            const newTime = newFrame / FPS;
            audio.currentTime = newTime;
            syncAnimationAndAudio();
        },
        goToTime: async (timeValue) => {
            if (timeValue >= 0 && timeValue <= totalDuration) {
                const targetFrame = Math.floor(timeValue * FPS);
                // Ensure frames around target are loaded
                await ensureFramesLoaded(targetFrame);
                audio.currentTime = timeValue;
                syncAnimationAndAudio();
            }
        },
        // 특정 프레임으로 직접 이동 (audio.currentTime 의존 없이 정확한 프레임 표시)
        goToFrame: (frameIndex) => {
            console.log(`[goToFrame] Called with frameIndex=${frameIndex}, totalFrames=${totalFrames}`);
            if (frameIndex >= 0 && frameIndex < totalFrames) {
                // 네비게이션 플래그 설정 (syncAnimationAndAudio가 덮어쓰지 않도록)
                isNavigatingToFrame = true;
                
                // 현재 프레임 인덱스를 먼저 업데이트 (중요!)
                currentFrameIndex = frameIndex;
                
                // 오디오 시간 설정
                audio.currentTime = frameIndex / FPS;
                
                const frameData = preloadedFrames[frameIndex];
                console.log(`[goToFrame] frameData exists: ${!!frameData}`);
                if (frameData) {
                    console.log(`[goToFrame] Updating to frame ${frameIndex}, currentFrameIndex now: ${currentFrameIndex}`);
                    updateMeshForFrame(frameData, frameIndex);
                    // 강제 렌더링
                    renderer.render(scene, camera);
                } else {
                    console.warn(`[goToFrame] No frame data for frame ${frameIndex}`);
                }
                
                // 시간 표시 업데이트
                updateTimeDisplay();
                
                // React 컴포넌트에 프레임 변경 알림 (프레임 표시 UI 업데이트용)
                window.dispatchEvent(new CustomEvent('frameChange', { detail: { frame: frameIndex } }));
                
                // 약간의 지연 후 플래그 해제 (timeupdate 이벤트가 처리될 시간)
                setTimeout(() => {
                    isNavigatingToFrame = false;
                }, 100);
            } else {
                console.warn(`[goToFrame] Invalid frameIndex: ${frameIndex}, totalFrames: ${totalFrames}`);
            }
        },
        getTotalDuration: () => totalDuration,
        getFPS: () => FPS,
        setPlaybackRate: (rate) => {
            const audioEl = document.getElementById('audio');
            if (audioEl) {
                audioEl.playbackRate = rate;
                console.log(`Playback rate set to ${rate}x`);
            }
        },
        getPlaybackRate: () => {
            const audioEl = document.getElementById('audio');
            return audioEl ? audioEl.playbackRate : 1;
        },
        getMissingFingeringSegments: () => {
            return findMissingFingeringSegments();
        },
        // 현재 프레임 다시 렌더링 (핑거링 수정 후 즉시 반영용)
        refreshCurrentFrame: () => {
            const currentFrame = selectedFingeringFrameIndex >= 0 ? selectedFingeringFrameIndex : Math.floor((audio?.currentTime || 0) * FPS);
            const frameData = preloadedFrames[currentFrame];
            if (frameData) {
                console.log(`[refreshCurrentFrame] Updating mesh for frame ${currentFrame}`);
                updateMeshForFrame(frameData, currentFrame);
            }
        },
        getEditedFingeringFrames: () => {
            return findEditedFingeringFrames();
        },
        showAssignFingeringPopup: () => {
            showAssignFingeringFromAppBar();
        }
    };
    
    // 편집 기능 초기화 (항상 활성화)
    if (!raycaster) {
        raycaster = new THREE.Raycaster();
    }
    currentCamera = camera;
    currentRenderer = renderer;
    setupFingeringEditListeners();


    const hemiLight = new THREE.HemisphereLight(0xffffff, 0x8d8d8d, 3);
    hemiLight.position.set(0, 20, 0);
    // scene.add(hemiLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 3);
    dirLight.position.set(3, 10, 10);
    dirLight.castShadow = false;
    // scene.add(dirLight);

    // Setup OrbitControls
    const pianoCenter = new THREE.Vector3(0.61431422, -0.074, -0.0055);
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.rotateSpeed = 0.5;
    // const controls = new TrackballControls(camera, renderer.domElement);
    // controls.rotateSpeed = 2.0;
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.screenSpacePanning = false;
    controls.minDistance = 0;
    controls.maxDistance = Infinity;
    // Define preset viewpoints
    const presets = {
        top: {
            position: {
                "x": 0.10415014734151176,
                "y": 1.2303206276916157,
                "z": -0.5999895298700443
            }, target: {
                "x": 0.1,
                "y": 0,
                "z": -0.6
            }
        },
        front: {
            position: {
                "x": -1.1,
                "y": 0,
                "z": -0.6
            },
            target: {
                "x": 0.1,
                "y": 0,
                "z": -0.6
            }
        },
        back: {
            position: {
                "x": 1.153,
                "y": 0,
                "z": -0.6
            },
            target: {
                "x": 0.1,
                "y": 0,
                "z": -0.6
            }
        },
        left: {
            position: {
                "x": 0.6998938717565077,
                "y": 1.0171497080001928,
                "z": 0.5578656721849795
            },
            target: {
                "x": 0.13180072175419472,
                "y": 1.5691260771514555e-19,
                "z": -0.5241889198249452
            },
        },
        right: {
            position: {
                "x": 0.6027349580836255,
                "y": 0.818198927359689,
                "z": -1.8632773667524538
            }, target: {
                "x": 0.06154058571770953,
                "y": 1.6977216676322782e-18,
                "z": -0.8159637729174926
            }
        }
    };

// Function to update camera and controls
    function updateCameraAndControls(preset) {
        camera.position.set(preset.position.x, preset.position.y, preset.position.z);
        controls.target.set(preset.target.x, preset.target.y, preset.target.z);
        controls.update();
    }

    const cameraSettings = {
        preset: 'top'
    };

    // 전역으로 노출하여 AppBar에서 접근 가능하도록
    window.visualizerControls.cameraSettings = cameraSettings;
    window.visualizerControls.presets = presets;
    window.visualizerControls.updateCamera = (presetName) => {
        if (presets[presetName]) {
            cameraSettings.preset = presetName;
            updateCameraAndControls(presets[presetName]);
        }
    };
    
    updateCameraAndControls(presets.top);
    // Load all meshes
    // Create materials
    const whiteMaterial = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        roughness: 0.3,
        metalness: 0.2
    });

    const blackMaterial = new THREE.MeshStandardMaterial({
        color: 0x000000,
        roughness: 0.4,
        metalness: 0.1
    });
    /**
     * 피아노 키 메쉬를 로드합니다.
     */
    async function loadPianoKeys() {
        console.log("Loading piano meshes...");
        updateLoadingOverlay("Loading piano model...", `0 / ${NUM_PIANO_KEYS} keys`, 5);
        let loadedKeys = 0;
        
        const loadKey = (index) => {
            return new Promise((resolve, reject) => {
                const objLoader = new OBJLoader();
                const mtlLoader = new MTLLoader();
                const meshUrl = `/piano_mesh/${index}`;
                
                mtlLoader.load(
                    `${meshUrl}.mtl`,
                    (materials) => {
                        materials.preload();
                        const mat = Object.values(materials.materials)[0];
                        mat.color.setHex(isBlackKey(index + 1) ? 0x000000 : 0xffffff);
                        objLoader.setMaterials(materials);
                        
                        objLoader.load(
                            `${meshUrl}.obj`,
                            (object) => {
                                object.traverse((child) => {
                                    if (child instanceof THREE.Mesh) {
                                        child.material = isBlackKey(index + 1) 
                                            ? blackMaterial.clone() 
                                            : whiteMaterial.clone();
                                        child.castShadow = true;
                                        child.receiveShadow = true;
                                    }
                                });
                                object.rotation.x = Math.PI / 2;
                                scene.add(object);
                                keyObjs[index] = object;
                                loadedKeys++;
                                
                                // 진행률 업데이트 (10개마다 또는 마지막)
                                if (loadedKeys % 10 === 0 || loadedKeys === NUM_PIANO_KEYS) {
                                    const progress = 5 + (loadedKeys / NUM_PIANO_KEYS) * 10; // 5% ~ 15%
                                    updateLoadingOverlay("Loading piano model...", `${loadedKeys} / ${NUM_PIANO_KEYS} keys`, progress);
                                }
                                
                                if (loadedKeys === NUM_PIANO_KEYS) {
                                    console.log("All piano keys loaded!");
                                }
                                resolve();
                            },
                            undefined,
                            (error) => {
                                console.error(`Error loading piano key ${index}:`, error);
                                reject(error);
                            }
                        );
                    },
                    undefined,
                    (error) => {
                        console.error(`Error loading piano material ${index}:`, error);
                        reject(error);
                    }
                );
            });
        };
        
        const loadPromises = Array.from({length: NUM_PIANO_KEYS}, (_, i) => loadKey(i));
        await Promise.allSettled(loadPromises);
    }
    
    await loadPianoKeys();
    
    // Fetch the mesh data (vertices and faces)
    updateLoadingOverlay("Loading piece information...", "Fetching metadata", 15);
    const metadata = await fetchMetaData();
    totalFrames = metadata.num_frames;
    
    // 곡 제목과 작곡가 업데이트
    const pieceTitleElement = document.getElementById('pieceTitle');
    const pieceComposerElement = document.getElementById('pieceComposer');
    const loadingContainer = document.getElementById('loadingContainer');
    const infoContent = document.getElementById('infoContent');
    
    if (pieceTitleElement) {
        pieceTitleElement.textContent = metadata.name || 'Unknown';
    }
    
    if (pieceComposerElement) {
        pieceComposerElement.textContent = metadata.composer || 'Unknown';
    }
    
    // 로딩 완료 시 로딩 컨테이너 숨기고 제목 표시
    if (loadingContainer && infoContent) {
        setTimeout(() => {
            loadingContainer.classList.add('hidden');
            infoContent.classList.remove('loading');
            setTimeout(() => {
                loadingContainer.style.display = 'none';
            }, 300);
        }, 500);
    }
    
    totalDuration = totalFrames / FPS;
    const totalDurationElement = document.getElementById('totalDuration');
    if (totalDurationElement) {
        const minutes = Math.floor(totalDuration / 60);
        const seconds = Math.floor(totalDuration % 60).toString().padStart(2, '0');
        totalDurationElement.textContent = `${minutes}:${seconds}`;
    }
    
    // 모든 메쉬 데이터와 핑거링 데이터를 로드 (모든 프레임 로드 완료까지 대기)
    await Promise.all([fetchMeshData(), fetchFingeringData(), fetchMidiNotes()]).catch(err => {
        console.error("Error loading data:", err);
    });
    
    console.log(`All ${loadedFrames} frames loaded, preparing scene...`);
    updateLoadingOverlay("Preparing 3D scene...", "Setting up hand meshes", 100);

    const facesData = await fetchFacesData();
    const leftFaces = facesData.left_faces.flat();
    const rightFaces = facesData.right_faces.flat();

    // 손 메쉬 지오메트리 생성
    const leftGeometry = new THREE.BufferGeometry();
    const rightGeometry = new THREE.BufferGeometry();

    const leftPositionAttribute = new Float32Array(NUM_VERTICES * 3);
    const rightPositionAttribute = new Float32Array(NUM_VERTICES * 3);

    leftGeometry.setAttribute('position', new THREE.BufferAttribute(leftPositionAttribute, 3));
    rightGeometry.setAttribute('position', new THREE.BufferAttribute(rightPositionAttribute, 3));

    leftGeometry.setIndex(leftFaces);
    rightGeometry.setIndex(rightFaces);


    // 손 메쉬 재질 생성
    const handMaterial = new THREE.MeshStandardMaterial({
        color: renderingParams.handColor,
        roughness: 1.0,
        metalness: 1.0
    });

    const leftMaterial = handMaterial.clone();
    const rightMaterial = handMaterial.clone();

    const leftMesh = new THREE.Mesh(leftGeometry, leftMaterial);
    const rightMesh = new THREE.Mesh(rightGeometry, rightMaterial);
    leftMesh.frustumCulled = false;
    rightMesh.frustumCulled = false;

    scene.add(leftMesh);
    scene.add(rightMesh);
    
    // 초기 프레임 데이터로 손 메쉬 초기화
    const initialFrameData = preloadedFrames[0];
        if (initialFrameData) {
            console.log("Loading initial frame data...");
            updateLoadingOverlay("Rendering first frame...", "Almost ready!", 100);
            for (let i = 0; i < NUM_VERTICES; i++) {
                for (let j = 0; j < 3; j++) {
                    leftPositionAttribute[i * 3 + j] = initialFrameData.left_vertices[i][j];
                    rightPositionAttribute[i * 3 + j] = initialFrameData.right_vertices[i][j];
                }
            }
            leftGeometry.computeVertexNormals();
            rightGeometry.computeVertexNormals();
            leftGeometry.attributes.position.needsUpdate = true;
            rightGeometry.attributes.position.needsUpdate = true;
            
            console.log("Initial frame loaded and ready");
            renderer.render(scene, camera);
            
            // Three.js 로딩 오버레이 숨기기
            const loadingOverlay = document.getElementById('threejsLoadingOverlay');
            if (loadingOverlay) {
                loadingOverlay.classList.add('hidden');
            }
            
            // Dispatch custom event to notify React that loading is complete
            window.dispatchEvent(new CustomEvent('visualizerDataLoaded', { detail: { pieceId: pieceId } }));
    } else {
        console.warn("Initial frame not found in memory cache");
        updateLoadingOverlay("Error", "Initial frame not loaded", 0);
    }

    // 핑거링 시각화를 위한 그룹
    const fingeringLabelGroup = new THREE.Group();
    scene.add(fingeringLabelGroup);
    
    // 손가락 번호별 텍스처 캐시
    const fingerTextures = {};
    
    /**
     * 손가락 번호에 해당하는 텍스처를 생성하거나 캐시에서 가져옵니다.
     * @param {number} fingerNumber - 손가락 번호 (1-5)
     * @param {boolean} isEdited - 편집된 핑거링인지 여부 (true: 빨간색, false: 파란색)
     * @param {boolean} isAmbiguous - ambiguous 핑거링인지 여부 (true: 노란색)
     */
    function getFingerTexture(fingerNumber, isEdited = false, isAmbiguous = false, wasAiCorrected = false, isMissing = false) {
        // 캐시 키에 색상 정보 포함
        const cacheKey = `${fingerNumber}_${isEdited ? 'edited' : 'original'}_${isAmbiguous ? 'ambiguous' : 'normal'}_${wasAiCorrected ? 'ai' : 'human'}_${isMissing ? 'missing' : 'exists'}`;
        if (fingerTextures[cacheKey]) {
            return fingerTextures[cacheKey];
        }
        
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 64;
        canvas.height = 64;
        
        context.clearRect(0, 0, 64, 64);
        // 누락된 노트에 추가된 핑거링: 노란색
        // AI가 수정한 핑거링: 빨간색
        // ambiguous: 주황색
        // 편집된 핑거링: 빨간색
        // 원본: 진한 파란색
        if (isMissing) {
            context.fillStyle = '#ffcc00'; // 누락 노트 추가: 노란색
        } else if (wasAiCorrected) {
            context.fillStyle = '#ff3333'; // AI 수정: 밝은 빨간색
        } else if (isAmbiguous) {
            context.fillStyle = '#FF6F00'; // 진한 주황색
        } else if (isEdited) {
            context.fillStyle = '#ff0000'; // 사람 편집: 빨간색
        } else {
            context.fillStyle = '#0033cc'; // 원본: 진한 파란색
        }
        context.font = 'bold 48px Arial';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText(fingerNumber.toString(), 32, 32);
        
        const texture = new THREE.CanvasTexture(canvas);
        texture.needsUpdate = true;
        fingerTextures[cacheKey] = texture;
        return texture;
    }
    
    // fingerTipSprites와 spriteToFingeringMap은 전역 변수로 이미 선언됨 (중복 선언 제거)

    // 씬 회전 설정
    scene.rotation.x = -Math.PI / 2;

    // 오디오 초기화
    const audio = document.getElementById('audio');
    const playButton = document.getElementById('playPauseBtn');
    const playPauseIcon = document.getElementById('playPauseIcon');
    const scrubber = document.getElementById('scrubber');
    const bufferedBar = document.getElementById('bufferedBar');

    let isPlaying = false;
    
    // 재생/일시정지 버튼 이벤트
    playButton.addEventListener('click', () => {
        if (isPlaying) {
            audio.pause();
            playPauseIcon.className = 'fas fa-play';
        } else {
            audio.play();
            playPauseIcon.className = 'fas fa-pause';
            // 재생 시작 시 핑거링 네비게이션 상태 리셋
            // 재생 후 핑거링 이동 시 현재 재생 위치 기준으로 검색하기 위함
            currentNavigationIndex = -1;
            selectedFingeringKey = null;
            selectedFingeringFrameIndex = -1;
        }
        isPlaying = !isPlaying;
    });



    /**
     * 프레임 데이터를 화면에 표시합니다.
     * @param {number} frameIndex - 프레임 인덱스
     * @param {boolean} autoPlay - 로드 후 자동 재생 여부 (기본값: true)
     */
    function playOnLoad(frameIndex, autoPlay = true) {
        if (frameIndex < 0 || frameIndex >= totalFrames) {
            return;
        }
        
        const frameData = preloadedFrames[frameIndex];
            if (frameData) {
                // 프레임 데이터로 화면 업데이트
                updateMeshForFrame(frameData, frameIndex);
                // 자동 재생이면 재생 시작
                if (autoPlay && isPlaying) {
                    audio.play();
                }
        }
    }

    /**
     * 관절 데이터에서 손가락 끝 위치를 추출합니다.
     */
    function extractFingerTipPosition(joints, fingerName) {
        const tipIndex = FINGER_TIP_INDICES[fingerName];
        if (tipIndex === undefined) return null;
        
        const baseIndex = tipIndex * 3;
        if (joints.length < (tipIndex + 1) * 3) return null;
        
        return new THREE.Vector3(
            joints[baseIndex],
            joints[baseIndex + 1],
            joints[baseIndex + 2]
        );
    }

    /**
     * 핑거링 스프라이트를 업데이트합니다.
     */
    // 빈 손가락용 투명 텍스처 (클릭 가능한 영역)
    let emptyFingerTexture = null;
    function getEmptyFingerTexture() {
        if (emptyFingerTexture) return emptyFingerTexture;
        
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');
        canvas.width = 64;
        canvas.height = 64;
        
        context.clearRect(0, 0, 64, 64);
        
        // 반투명 원형 배경 (시안색)
        context.fillStyle = 'rgba(100, 255, 218, 0.3)';
        context.beginPath();
        context.arc(32, 32, 24, 0, Math.PI * 2);
        context.fill();
        
        // 원형 테두리 (시안색)
        context.strokeStyle = 'rgba(100, 255, 218, 0.8)';
        context.lineWidth = 2;
        context.beginPath();
        context.arc(32, 32, 24, 0, Math.PI * 2);
        context.stroke();
        
        // + 기호
        context.fillStyle = 'rgba(255, 255, 255, 0.9)';
        context.font = 'bold 28px Arial';
        context.textAlign = 'center';
        context.textBaseline = 'middle';
        context.fillText('+', 32, 32);
        
        emptyFingerTexture = new THREE.CanvasTexture(canvas);
        emptyFingerTexture.needsUpdate = true;
        return emptyFingerTexture;
    }
    
    const ALL_FINGERS = ['thumb', 'index', 'middle', 'ring', 'pinky'];
    const FINGER_TO_NUMBER = { thumb: 1, index: 2, middle: 3, ring: 4, pinky: 5 };
    
    function updateFingeringSprites(frameData, frameIndex) {
        // 모든 스프라이트 숨김
        Object.values(fingerTipSprites).forEach(sprite => {
            sprite.visible = false;
        });
        
        if (!renderingParams.showFingering || !frameData.left_joints || !frameData.right_joints) {
            return;
        }
        
        const leftJoints = frameData.left_joints.map(v => parseFloat(v));
        const rightJoints = frameData.right_joints.map(v => parseFloat(v));
        
        // 편집된 데이터가 있으면 우선 사용, 없으면 원본 데이터 사용
        const currentFingering = editedFingeringData[frameIndex] || fingeringData[frameIndex] || [];
        const originalFingering = originalFingeringData[frameIndex] || [];
        
        // 모든 손가락에 대해 스프라이트 생성/업데이트
        ['left', 'right'].forEach(hand => {
            const joints = hand === 'left' ? leftJoints : rightJoints;
            
            ALL_FINGERS.forEach(fingerName => {
                const tipPos = extractFingerTipPosition(joints, fingerName);
            if (!tipPos) return;
            
                const spriteKey = `${hand}_${fingerName}`;
                
                // 이 손가락에 핑거링이 있는지 확인
                const fingering = currentFingering.find(f => f.hand === hand && f.finger_name === fingerName);
                const hasFingering = !!fingering;
                
            let sprite = fingerTipSprites[spriteKey];
            
            if (!sprite) {
                    // 스프라이트 생성 (처음에는 빈 텍스처로)
                const spriteMaterial = new THREE.SpriteMaterial({ 
                        map: getEmptyFingerTexture(),
                    transparent: true,
                    depthTest: false,
                        depthWrite: false,
                        opacity: 0.0  // 기본적으로 투명
                });
                sprite = new THREE.Sprite(spriteMaterial);
                    sprite.scale.set(0.03, 0.03, 1);  // 클릭 영역 (축소)
                    sprite.renderOrder = 9999;  // 최상위 렌더링
                    sprite.userData.isFingeringSprite = true;
                fingeringLabelGroup.add(sprite);
                fingerTipSprites[spriteKey] = sprite;
            }
            
                if (hasFingering) {
                    // 핑거링이 있으면 숫자 표시
                    const fingerNumber = fingering.finger;
                    const keyIndex = fingering.key_index;
                    
                    // 편집 여부 확인: hand, finger_name, key_index가 모두 일치하는 원본이 있는지 확인
                    let isEdited = false;
                    const originalMatch = originalFingering.find(f => 
                        f.hand === hand && f.finger_name === fingerName && f.key_index === keyIndex
                    );
                    if (originalMatch) {
                        // 원본이 있으면, 손가락 번호가 변경되었는지 확인
                        if (originalMatch.finger !== fingerNumber) {
                            isEdited = true;
                        }
                        // ambiguous 상태가 변경되었는지도 확인
                        if ((fingering.ambiguous === true) !== (originalMatch.ambiguous === true)) {
                            isEdited = true;
                        }
                    } else {
                        // 원본에 없으면 새로 추가된 것이므로 편집됨
                        isEdited = true;
                    }
                    
                    // ambiguous 여부 확인
                    const isAmbiguous = fingering.ambiguous === true;
                    
                    // AI 수정 여부 확인 (r0 어노테이션에서 was_corrected 플래그)
                    const wasAiCorrected = isAiAnnotation && fingering.was_corrected === true;
                    
                    // 누락된 노트 여부 확인 (is_missing 플래그)
                    const isMissing = isAiAnnotation && fingering.is_missing === true;
                    
                    const texture = getFingerTexture(fingerNumber, isEdited, isAmbiguous, wasAiCorrected, isMissing);
                    sprite.material.map = texture;
                    sprite.material.opacity = 1.0;
                    
                    // 핑거링 정보 매핑
            spriteToFingeringMap.set(sprite, {
                frameIndex: frameIndex,
                hand: hand,
                fingerName: fingerName,
                fingerNumber: fingerNumber,
                fingering: fingering,
                        isEdited: isEdited,
                        wasAiCorrected: wasAiCorrected,
                        hasFingering: true
                    });
                } else {
                    // 핑거링이 없으면 투명한 클릭 가능 스프라이트
                    sprite.material.map = getEmptyFingerTexture();
                    sprite.material.opacity = 0.15;  // 약간 보이게 (호버 시 더 선명)
                    
                    // 빈 손가락 정보 매핑 (클릭으로 핑거링 추가 가능)
                    spriteToFingeringMap.set(sprite, {
                        frameIndex: frameIndex,
                        hand: hand,
                        fingerName: fingerName,
                        fingerNumber: FINGER_TO_NUMBER[fingerName],
                        fingering: null,
                        isEdited: false,
                        hasFingering: false
                    });
                }
                
                sprite.material.needsUpdate = true;
            sprite.position.copy(tipPos);
                // 스프라이트를 손 위로 약간 올려서 클릭이 잘 되도록
                sprite.position.z += 0.02;
            sprite.visible = true;
            });
        });
        
        // 선택된 핑거링 강조 유지 (선택된 프레임과 현재 프레임이 일치할 때만)
        // 애니메이션 루프에서 다른 프레임으로 덮어쓰는 것을 방지
        if (selectedFingeringFrameIndex < 0 || selectedFingeringFrameIndex === frameIndex) {
            highlightSelectedFingering();
        }
    }

    /**
     * 피아노 키 상태를 업데이트합니다.
     */
    function updatePianoKeys(frameData) {
        for (let i = 0; i < NUM_PIANO_KEYS; i++) {
            const keyObj = keyObjs[i];
            if (!keyObj || !keyObj.children[0]) continue;
            
            const material = keyObj.children[0].material;
            const isPressed = frameData.pressed_keys && frameData.pressed_keys[i] > 0;
            
            if (isPressed) {
                material.color.setHex(0x00ff00);
                keyObj.rotation.z = -Math.PI / 45;
            } else {
                keyObj.rotation.z = 0;
                material.color.setHex(isBlackKey(i + 1) ? 0x000000 : 0xffffff);
            }
            material.needsUpdate = true;
        }
    }

    /**
     * 재생 시간 표시를 업데이트합니다.
     */
    function updateTimeDisplay() {
        const currentTime = audio.currentTime || 0;
        const scrubberValue = (currentTime / totalDuration) * 100;
        
        scrubber.value = scrubberValue;
        const timeDisplay = document.getElementById('timeDisplay');
        const playedBar = document.getElementById('playedBar');
        const frameDisplay = document.getElementById('frameDisplay');
        
        if (timeDisplay) {
            const minutes = Math.floor(currentTime / 60);
            const seconds = Math.floor(currentTime % 60).toString().padStart(2, '0');
            timeDisplay.textContent = `${minutes}:${seconds}`;
        }
        
        if (playedBar) {
            playedBar.style.width = `${scrubberValue}%`;
        }
        
        // 프레임 번호 표시 업데이트
        if (frameDisplay) {
            const frameIndex = Math.floor(currentTime * FPS);
            frameDisplay.textContent = `(Frame ${frameIndex})`;
        }
    }

    // 현재 프레임 인덱스는 전역 변수 currentFrameIndex를 사용 (46번 줄에 선언됨)
    
    /**
     * 현재 프레임 데이터로 메쉬를 업데이트합니다.
     * 주의: 이 함수는 화면만 업데이트하고 currentFrameIndex는 변경하지 않습니다.
     * currentFrameIndex는 goToFrame, syncAnimationAndAudio, 스크러버에서만 업데이트됩니다.
     */
    function updateMeshForFrame(frameData, frameIndex = 0) {
        if (!frameData) return;
        
        // 손 메쉬 버텍스 업데이트
        for (let i = 0; i < NUM_VERTICES; i++) {
            for (let j = 0; j < 3; j++) {
                leftPositionAttribute[i * 3 + j] = frameData.left_vertices[i][j];
                rightPositionAttribute[i * 3 + j] = frameData.right_vertices[i][j];
            }
        }

        leftGeometry.computeVertexNormals();
        rightGeometry.computeVertexNormals();
        leftGeometry.attributes.position.needsUpdate = true;
        rightGeometry.attributes.position.needsUpdate = true;

        // 핑거링 스프라이트 업데이트
        updateFingeringSprites(frameData, frameIndex);
        
        // 스켈레톤 업데이트
        updateSkeletonForFrame(frameData);
        
        // 피아노 키 업데이트
        updatePianoKeys(frameData);
        
        // 시간 표시 업데이트
        updateTimeDisplay();
    }
    
    /**
     * 애니메이션과 오디오를 동기화합니다.
     */
    function syncAnimationAndAudio() {
        // 핑거링 네비게이션 중이면 프레임 업데이트 스킵 (goToFrame에서 이미 처리됨)
        if (isNavigatingToFrame) {
            updateTimeDisplay();
            return;
        }
        
        // 재생바 및 시간 표시 업데이트
        updateTimeDisplay();
        
        // 오디오가 일시정지 상태이면 현재 currentFrameIndex 유지 (수동 프레임 이동 보호)
        // 재생 중일 때만 오디오 시간 기반으로 프레임 업데이트
        if (audio.paused && !isPlaying) {
            // 일시정지 상태에서는 현재 프레임 유지, 업데이트하지 않음
            return;
        }
        
        const currentTime = audio.currentTime || 0;
        const frameIndex = Math.floor(currentTime * FPS);
        
        if (frameIndex < 0 || frameIndex >= totalFrames) {
            if (isPlaying) {
                audio.pause();
                isPlaying = false;
                playPauseIcon.className = 'fas fa-play';
            }
            return;
        }
        
        // 메모리 캐시에서 프레임 데이터 가져오기
        const frameData = preloadedFrames[frameIndex];
        
        if (frameData) {
            // 재생 중에는 currentFrameIndex도 업데이트 (프레임 이동 시 정확한 위치 사용)
            currentFrameIndex = frameIndex;
            updateMeshForFrame(frameData, frameIndex);
        } else {
            // 모든 프레임이 미리 로드되어야 하므로 이 경우는 드물다
            console.warn(`Frame ${frameIndex} not found in memory`);
        }
    }

    audio.ontimeupdate = syncAnimationAndAudio;
    
    // Seek 시 화면 업데이트 (스크러버, 클릭 등으로 인한 seek)
    // 주의: currentFrameIndex는 여기서 업데이트하지 않음 (goToFrame, 스크러버에서 처리)
    audio.onseeked = () => {
        // goToFrame에서 이미 처리된 경우 스킵
        if (isNavigatingToFrame) {
            updateTimeDisplay();
            return;
        }
        
        // 화면 업데이트만 수행 (currentFrameIndex는 유지)
        // 스크러버 조작 시에는 스크러버 핸들러에서 이미 currentFrameIndex를 설정함
        updateTimeDisplay();
    };

    scrubber.addEventListener('input', () => {
        const scrubberValue = scrubber.value;
        const newTime = (scrubberValue / 100) * totalDuration;
        audio.currentTime = newTime;
        
        // 스크러버 조작 시 핑거링 네비게이션 상태 리셋
        currentNavigationIndex = -1;
        selectedFingeringKey = null;
        selectedFingeringFrameIndex = -1;
        
        // 직접 프레임 업데이트 (일시정지 상태에서도 동작하도록)
        const frameIndex = Math.floor(newTime * FPS);
        if (frameIndex >= 0 && frameIndex < totalFrames) {
            currentFrameIndex = frameIndex;
            const frameData = preloadedFrames[frameIndex];
            if (frameData) {
                updateMeshForFrame(frameData, frameIndex);
            }
            window.dispatchEvent(new CustomEvent('frameChange', { detail: { frame: frameIndex } }));
        }
        updateTimeDisplay();
    });

    // 카메라 좌표 표시 UI 생성
    const cameraInfoDiv = document.createElement('div');
    cameraInfoDiv.id = 'camera-info';
    cameraInfoDiv.style.cssText = `
        position: fixed;
        top: 70px;
        right: 20px;
        background: rgba(0, 0, 0, 0.7);
        color: #64ffda;
        padding: 12px 16px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', 'Fira Code', monospace;
        font-size: 12px;
        z-index: 1000;
        border: 1px solid rgba(100, 255, 218, 0.3);
        backdrop-filter: blur(10px);
        min-width: 180px;
        display: none;
    `;
    cameraInfoDiv.innerHTML = `
        <div style="color: #fff; font-weight: 600; margin-bottom: 8px; font-size: 11px; opacity: 0.7;">CAMERA POSITION</div>
        <div style="display: grid; grid-template-columns: 20px 1fr; gap: 4px; line-height: 1.6;">
            <span style="color: #ff6b6b;">X:</span><span id="cam-x">0.000</span>
            <span style="color: #69db7c;">Y:</span><span id="cam-y">0.000</span>
            <span style="color: #74c0fc;">Z:</span><span id="cam-z">0.000</span>
        </div>
        <div style="color: #fff; font-weight: 600; margin: 10px 0 8px; font-size: 11px; opacity: 0.7;">TARGET</div>
        <div style="display: grid; grid-template-columns: 20px 1fr; gap: 4px; line-height: 1.6;">
            <span style="color: #ff6b6b;">X:</span><span id="target-x">0.000</span>
            <span style="color: #69db7c;">Y:</span><span id="target-y">0.000</span>
            <span style="color: #74c0fc;">Z:</span><span id="target-z">0.000</span>
        </div>
    `;
    document.body.appendChild(cameraInfoDiv);
    
    // 카메라 좌표 토글 함수를 전역으로 노출
    window.visualizerControls.toggleCameraInfo = () => {
        const isVisible = cameraInfoDiv.style.display !== 'none';
        cameraInfoDiv.style.display = isVisible ? 'none' : 'block';
        return !isVisible;
    };
    window.visualizerControls.isCameraInfoVisible = () => {
        return cameraInfoDiv.style.display !== 'none';
    };

    /**
     * 애니메이션 루프
     */
    function animate() {
        syncAnimationAndAudio();
        controls.update();
        renderer.render(scene, camera);
        
        // 카메라 좌표 업데이트
        const camX = document.getElementById('cam-x');
        const camY = document.getElementById('cam-y');
        const camZ = document.getElementById('cam-z');
        const targetX = document.getElementById('target-x');
        const targetY = document.getElementById('target-y');
        const targetZ = document.getElementById('target-z');
        
        if (camX && camY && camZ) {
            camX.textContent = camera.position.x.toFixed(3);
            camY.textContent = camera.position.y.toFixed(3);
            camZ.textContent = camera.position.z.toFixed(3);
        }
        if (targetX && targetY && targetZ) {
            targetX.textContent = controls.target.x.toFixed(3);
            targetY.textContent = controls.target.y.toFixed(3);
            targetZ.textContent = controls.target.z.toFixed(3);
        }
    }

    let lastTime = 0;
    const frameInterval = 1000 / FPS;

    function animationLoop(time) {
        if (time - lastTime >= frameInterval) {
            animate();
            lastTime = time;
        }
        requestAnimationFrame(animationLoop);
    }

    console.log("Starting animation loop...");
    animationLoop(0);

    // Handle window resize
    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });
    
    // 키보드 단축키 및 재생 컨트롤 초기화 (DOM이 준비될 때까지 대기)
    setTimeout(() => {
        setupPlaybackControls();
    }, 500);
}

/**
 * 모든 핑거링을 프레임 순서대로 flat하게 정리합니다.
 * 같은 핑거링이 연속된 프레임에 있으면 시작 프레임만 수집합니다.
 * @returns {Array} [{frameIndex, hand, fingerName, fingerNumber, endFrameIndex}, ...]
 */
function getAllFingeringsFlat() {
    const allFingerings = [];
    
    // editedFingeringData가 있으면 그것을, 없으면 fingeringData 사용
    // Object.keys()는 문자열 배열을 반환하므로 Number로 변환하고 NaN 필터링
    const getMaxFrame = (dataObj) => {
        const keys = Object.keys(dataObj);
        if (keys.length === 0) return 0;
        const numbers = keys.map(k => Number(k)).filter(n => !isNaN(n));
        return numbers.length > 0 ? Math.max(...numbers) : 0;
    };
    
    const maxFrame = Math.max(
        getMaxFrame(fingeringData),
        getMaxFrame(editedFingeringData)
    );
    
    if (maxFrame === 0) return allFingerings;
    
    // 현재 활성화된 핑거링 추적 (hand_fingerName -> {startFrame, fingerNumber, ...})
    const activeFingeringsMap = new Map();
    
    for (let frameIdx = 0; frameIdx <= maxFrame; frameIdx++) {
        const frameFingering = editedFingeringData[frameIdx] || fingeringData[frameIdx] || [];
        
        // 현재 프레임의 핑거링 키 세트
        const currentKeys = new Set();
        
        frameFingering.forEach(f => {
            // key_index를 포함하여 같은 손가락이라도 다른 건반이면 다른 핑거링으로 인식
            const key = `${f.hand}_${f.finger_name}_${f.key_index}`;
            currentKeys.add(key);
            
            // 이 핑거링이 이미 활성화되어 있는지 확인
            const existing = activeFingeringsMap.get(key);
            
            if (!existing) {
                // 새로운 핑거링 시작 - 수집!
                const fingeringEntry = {
                    frameIndex: frameIdx,
                    hand: f.hand,
                    fingerName: f.finger_name,
                    fingerNumber: f.finger,
                    keyIndex: f.key_index,
                    endFrameIndex: frameIdx  // 나중에 업데이트됨
                };
                allFingerings.push(fingeringEntry);
                activeFingeringsMap.set(key, fingeringEntry);
            } else if (existing.fingerNumber !== f.finger) {
                // 같은 손가락이지만 번호가 바뀜 - 새 구간 시작
                existing.endFrameIndex = frameIdx - 1;  // 이전 구간 종료
                
                const fingeringEntry = {
                    frameIndex: frameIdx,
                    hand: f.hand,
                    fingerName: f.finger_name,
                    fingerNumber: f.finger,
                    keyIndex: f.key_index,
                    endFrameIndex: frameIdx
                };
                allFingerings.push(fingeringEntry);
                activeFingeringsMap.set(key, fingeringEntry);
            } else {
                // 같은 핑거링 유지 - endFrameIndex만 업데이트
                existing.endFrameIndex = frameIdx;
            }
        });
        
        // 이전 프레임에 있었지만 현재 프레임에 없는 핑거링 종료 처리
        for (const [key, entry] of activeFingeringsMap.entries()) {
            if (!currentKeys.has(key)) {
                // 핑거링 구간 종료
                activeFingeringsMap.delete(key);
            }
        }
    }
    
    // 프레임 순서로 정렬
    allFingerings.sort((a, b) => a.frameIndex - b.frameIndex);
    
    console.log(`Found ${allFingerings.length} fingering segments (was checking ${maxFrame + 1} frames)`);
    
    return allFingerings;
}

/**
 * 모든 네비게이션 포인트를 프레임 순서대로 정리합니다.
 * MIDI 노트 기반으로 각 노트의 onset 프레임을 네비게이션 포인트로 사용합니다.
 * 이렇게 하면 모든 MIDI 노트를 순차적으로 확인할 수 있습니다.
 * @returns {Array} [{frameIndex, type: 'fingering'|'missing', keyIndex, ...}, ...]
 */
function getAllNavigationPoints() {
    const navigationPoints = [];
    
    // MIDI 노트 기반 네비게이션 포인트
    if (midiNotes && midiNotes.length > 0) {
        for (const note of midiNotes) {
            const frameIndex = note.onset_frame;
            const keyIndex = note.key_idx;
            
            // 해당 프레임의 핑거링 데이터에서 이 키에 대한 핑거링 찾기
            const frameFingering = editedFingeringData[frameIndex] || fingeringData[frameIndex] || [];
            const fingering = frameFingering.find(f => f.key_index === keyIndex);
            
            if (fingering) {
                // 핑거링이 할당된 노트
                navigationPoints.push({
                    frameIndex: frameIndex,
                    type: 'fingering',
                    hand: fingering.hand,
                    fingerName: fingering.finger_name,
                    fingerNumber: fingering.finger,
                    keyIndex: keyIndex,
                    isMissing: fingering.is_missing || false
                });
            } else {
                // 핑거링이 없는 노트 (missing)
                navigationPoints.push({
                    frameIndex: frameIndex,
                    type: 'missing',
                    keyIndex: keyIndex
                });
            }
        }
    } else {
        // Fallback: MIDI 노트가 없으면 기존 핑거링 세그먼트 방식 사용
        const allFingerings = getAllFingeringsFlat();
        allFingerings.forEach(f => {
            navigationPoints.push({
                frameIndex: f.frameIndex,
                type: 'fingering',
                hand: f.hand,
                fingerName: f.fingerName,
                fingerNumber: f.fingerNumber,
                keyIndex: f.keyIndex,
                endFrameIndex: f.endFrameIndex
            });
        });
    }
    
    // 프레임 순서로 정렬, 같은 프레임이면 keyIndex 순서
    navigationPoints.sort((a, b) => {
        if (a.frameIndex !== b.frameIndex) {
            return a.frameIndex - b.frameIndex;
        }
        return (a.keyIndex || 0) - (b.keyIndex || 0);
    });
    
    // 중복 제거 (같은 프레임, 같은 keyIndex)
    const uniquePoints = [];
    const seenKeys = new Set();
    for (const point of navigationPoints) {
        const key = `${point.frameIndex}_${point.keyIndex}`;
        if (!seenKeys.has(key)) {
            uniquePoints.push(point);
            seenKeys.add(key);
        }
    }
    
    const fingeringCount = uniquePoints.filter(p => p.type === 'fingering').length;
    const missingCount = uniquePoints.filter(p => p.type === 'missing').length;
    console.log(`Found ${uniquePoints.length} navigation points (${fingeringCount} with fingering, ${missingCount} missing)`);
    
    return uniquePoints;
}

/**
 * 다음 네비게이션 포인트로 이동합니다 (핑거링 또는 비어있는 구간).
 * currentNavigationIndex가 유효하면 인덱스 기반으로, 아니면 프레임 기반으로 다음 포인트를 찾습니다.
 * 이를 통해 같은 프레임에 여러 핑거링(화음)이 있어도 모두 순회할 수 있습니다.
 */
function goToNextFingering() {
    const navigationPoints = getAllNavigationPoints();
    if (navigationPoints.length === 0) {
        console.log('No navigation points found');
        return;
    }
    
    const audio = document.getElementById('audio');
    const currentTime = audio ? audio.currentTime : 0;
    const audioFrame = Math.floor(currentTime * FPS);
    
    let nextIndex = -1;
    
    // currentNavigationIndex가 유효하면 항상 인덱스 기반으로 이동
    // 사용자가 스크러버를 조작하면 currentNavigationIndex가 리셋되므로 안전함
    const useNavigationIndex = currentNavigationIndex >= 0;
    
    if (useNavigationIndex) {
        // 인덱스 기반: 현재 인덱스 + 1로 이동
        nextIndex = currentNavigationIndex + 1;
        if (nextIndex >= navigationPoints.length) {
            console.log('Already at last navigation point');
            return;
        }
    } else {
        // 프레임 기반: 현재 프레임보다 크거나 같은 첫 번째 포인트 찾기
        // (audioFrame과 같은 프레임의 첫 번째 핑거링부터 시작)
        for (let i = 0; i < navigationPoints.length; i++) {
            if (navigationPoints[i].frameIndex >= audioFrame) {
                nextIndex = i;
                break;
            }
        }
        
        // 현재 프레임에 해당하는 포인트가 없으면 다음 프레임 찾기
        if (nextIndex === -1) {
            console.log('No more navigation points');
            return;
        }
    }
    
    if (nextIndex >= 0 && nextIndex < navigationPoints.length) {
        const nextPoint = navigationPoints[nextIndex];
        const pointType = nextPoint.type === 'fingering' ? 'fingering' : 'missing';
        const fingerInfo = nextPoint.type === 'fingering' 
            ? `${nextPoint.hand} ${nextPoint.fingerName} key=${nextPoint.keyIndex}` 
            : `key=${nextPoint.keyIndex}`;
        console.log(`Next: ${nextIndex + 1}/${navigationPoints.length}, Frame ${nextPoint.frameIndex} (${pointType}) ${fingerInfo}`);
        navigateToPoint(nextPoint, nextIndex);
    }
}

/**
 * 이전 네비게이션 포인트로 이동합니다 (핑거링 또는 비어있는 구간).
 * currentNavigationIndex가 유효하면 인덱스 기반으로, 아니면 프레임 기반으로 이전 포인트를 찾습니다.
 * 이를 통해 같은 프레임에 여러 핑거링(화음)이 있어도 모두 순회할 수 있습니다.
 */
function goToPrevFingering() {
    const navigationPoints = getAllNavigationPoints();
    if (navigationPoints.length === 0) {
        console.log('No navigation points found');
        return;
    }
    
    const audio = document.getElementById('audio');
    const currentTime = audio ? audio.currentTime : 0;
    const audioFrame = Math.floor(currentTime * FPS);
    
    let prevIndex = -1;
    
    // currentNavigationIndex가 유효하면 항상 인덱스 기반으로 이동
    // 사용자가 스크러버를 조작하면 currentNavigationIndex가 리셋되므로 안전함
    const useNavigationIndex = currentNavigationIndex >= 0;
    
    if (useNavigationIndex) {
        // 인덱스 기반: 현재 인덱스 - 1로 이동
        prevIndex = currentNavigationIndex - 1;
        if (prevIndex < 0) {
            console.log('Already at first navigation point');
            return;
        }
    } else {
        // 프레임 기반: 현재 프레임보다 작거나 같은 마지막 포인트 찾기
        for (let i = navigationPoints.length - 1; i >= 0; i--) {
            if (navigationPoints[i].frameIndex <= audioFrame) {
                prevIndex = i;
                break;
            }
        }
        
        if (prevIndex === -1) {
            console.log('No previous navigation points');
            return;
        }
    }
    
    if (prevIndex >= 0 && prevIndex < navigationPoints.length) {
        const prevPoint = navigationPoints[prevIndex];
        const pointType = prevPoint.type === 'fingering' ? 'fingering' : 'missing';
        const fingerInfo = prevPoint.type === 'fingering' 
            ? `${prevPoint.hand} ${prevPoint.fingerName} key=${prevPoint.keyIndex}` 
            : `key=${prevPoint.keyIndex}`;
        console.log(`Prev: ${prevIndex + 1}/${navigationPoints.length}, Frame ${prevPoint.frameIndex} (${pointType}) ${fingerInfo}`);
        navigateToPoint(prevPoint, prevIndex);
    }
}

/**
 * 특정 네비게이션 포인트로 이동하고 강조 표시합니다.
 * @param {Object} pointInfo - 네비게이션 포인트 정보 (핑거링 또는 비어있는 구간)
 * @param {number} navIndex - 전역 네비게이션 인덱스 (선택적)
 */
async function navigateToPoint(pointInfo, navIndex = -1) {
    const audio = document.getElementById('audio');
    if (!audio) return;
    
    // 네비게이션 인덱스 저장
    if (navIndex >= 0) {
        currentNavigationIndex = navIndex;
    }
    
    // 선택된 포인트의 프레임 인덱스 저장
    selectedFingeringFrameIndex = pointInfo.frameIndex;
    
    // 재생 중이면 먼저 일시정지
    const playPauseIcon = document.getElementById('playPauseIcon');
    if (!audio.paused) {
        audio.pause();
        if (playPauseIcon) {
            playPauseIcon.className = 'fas fa-play';
        }
    }
    
    // 해당 프레임으로 이동
    const targetFrame = pointInfo.frameIndex;
    const newTime = targetFrame / FPS;
    
    // 핑거링인 경우에만 선택된 핑거링 키 업데이트 (missing은 null)
    // keyIndex도 포함하여 정확한 핑거링 식별
    if (pointInfo.type === 'fingering') {
        selectedFingeringKey = `${pointInfo.hand}_${pointInfo.fingerName}_${pointInfo.keyIndex}`;
    } else {
        // missing 타입의 경우 keyIndex만으로 하이라이트 시도
        selectedFingeringKey = pointInfo.keyIndex !== undefined ? `missing_${pointInfo.keyIndex}` : null;
    }
    
    // 직접 targetFrame을 사용하여 프레임 업데이트 (audio.currentTime 의존 X)
    // 이렇게 하면 정확한 프레임이 항상 표시됨
    if (window.visualizerControls && window.visualizerControls.goToFrame) {
        window.visualizerControls.goToFrame(targetFrame);
    } else if (window.visualizerControls && window.visualizerControls.syncAnimationAndAudio) {
        // fallback: goToFrame이 없으면 기존 방식 사용
        audio.currentTime = newTime;
        window.visualizerControls.syncAnimationAndAudio();
    }
    
    // 핑거링인 경우에만 강조 표시 (비어있는 구간은 강조 표시 안 함)
    if (pointInfo.type === 'fingering') {
    highlightSelectedFingering();
    } else {
        // 비어있는 구간이면 강조 표시 해제
        clearFingeringHighlight();
    }
}

/**
 * 특정 핑거링으로 이동하고 강조 표시합니다. (하위 호환성을 위해 유지)
 * @param {Object} fingeringInfo - 핑거링 정보
 * @param {number} navIndex - 전역 네비게이션 인덱스 (선택적)
 */
async function navigateToFingering(fingeringInfo, navIndex = -1) {
    // navigateToPoint로 위임
    await navigateToPoint(fingeringInfo, navIndex);
}

/**
 * 선택된 핑거링을 강조 표시합니다 (크기와 테두리 효과로 강조).
 */
function highlightSelectedFingering() {
    if (!selectedFingeringKey) return;
    
    // 선택된 핑거링 정보 파싱
    const parts = selectedFingeringKey.split('_');
    
    let selectedHand = null;
    let selectedFingerName = null;
    let selectedKeyIndex = null;
    let isMissingType = false;
    
    // missing_keyIndex 형태 또는 hand_fingerName_keyIndex 형태
    if (parts[0] === 'missing' && parts.length >= 2) {
        isMissingType = true;
        selectedKeyIndex = parseInt(parts[1], 10);
    } else if (parts.length >= 3) {
        selectedHand = parts[0];
        selectedFingerName = parts[1];
        selectedKeyIndex = parseInt(parts[2], 10);
    } else {
        return;
    }
    
    let foundSelected = false;
    let visibleCount = 0;
    
    // 모든 핑거링 스프라이트를 순회하며 선택된 것만 강조
    Object.entries(fingerTipSprites).forEach(([spriteKey, sprite]) => {
        if (!sprite || !sprite.visible) return;
        visibleCount++;
        
        const info = spriteToFingeringMap.get(sprite);
        if (!info || !info.hasFingering) return;
        
        let isSelected = false;
        
        if (isMissingType) {
            // missing 타입: keyIndex만으로 매칭 (해당 키에 핑거링이 있으면 하이라이트)
            isSelected = info.fingering && info.fingering.key_index === selectedKeyIndex;
        } else {
            // fingering 타입: hand, fingerName, keyIndex 모두 매칭
            isSelected = info.hand === selectedHand && 
                        info.fingerName === selectedFingerName &&
                        info.fingering && info.fingering.key_index === selectedKeyIndex;
        }
        
        if (isSelected) {
            foundSelected = true;
            // 선택된 핑거링: 크기 확대 및 밝기 증가
            sprite.scale.set(0.045, 0.045, 1);  // 1.5배 확대
            sprite.material.opacity = 1.0;
        } else {
            // 선택되지 않은 핑거링: 기본 크기 및 약간 투명
            sprite.scale.set(0.03, 0.03, 1);  // 기본 크기
            sprite.material.opacity = 0.6;  // 약간 투명
        }
    });
    
    console.log(`[highlightSelectedFingering] key=${selectedFingeringKey}, visibleSprites=${visibleCount}, foundSelected=${foundSelected}`);
}

/**
 * 핑거링 강조 표시를 해제합니다.
 */
function clearFingeringHighlight() {
    // 모든 핑거링 스프라이트를 기본 상태로 복원
    Object.values(fingerTipSprites).forEach(sprite => {
        if (!sprite) return;
        sprite.scale.set(0.03, 0.03, 1);  // 기본 크기
        if (sprite.material) {
            sprite.material.opacity = 1.0;  // 완전 불투명
        }
    });
    selectedFingeringKey = null;
}

/**
 * 재생 컨트롤 및 키보드 단축키 설정
 */
function setupPlaybackControls() {
    const audio = document.getElementById('audio');
    const playPauseBtn = document.getElementById('playPauseBtn');
    const playPauseIcon = document.getElementById('playPauseIcon');
    const prevFrameBtn = document.getElementById('prevFrameBtn');
    const nextFrameBtn = document.getElementById('nextFrameBtn');
    const timeInput = document.getElementById('timeInput');
    const goToTimeBtn = document.getElementById('goToTimeBtn');
    const scrubber = document.getElementById('scrubber');
    
    // audio만 필수, 나머지는 없어도 계속 진행
    if (!audio) {
        console.warn("Audio element not found, retrying in 500ms...");
        setTimeout(setupPlaybackControls, 500);
        return;
    }
    
    // 일부 UI 버튼이 없어도 키보드 단축키는 정상 작동
    
    let isPlaying = false;
    
    // 시간 파싱 함수 (예: "1:23" 또는 "83" -> 초)
    function parseTimeInput(input) {
        if (!input || input.trim() === '') return null;
        
        const trimmed = input.trim();
        
        // "분:초" 형식
        if (trimmed.includes(':')) {
            const parts = trimmed.split(':');
            if (parts.length === 2) {
                const minutes = parseInt(parts[0]) || 0;
                const seconds = parseFloat(parts[1]) || 0;
                return minutes * 60 + seconds;
            }
        }
        
        // 초 단위 숫자
        const seconds = parseFloat(trimmed);
        if (!isNaN(seconds)) {
            return seconds;
        }
        
        return null;
    }
    
    // 시간으로 이동
    function goToTime() {
        const timeInputField = document.getElementById('timeInput');
        if (!timeInputField) {
            console.warn("Time input field not found");
            return;
        }
        
        const timeValue = parseTimeInput(timeInputField.value);
        if (timeValue !== null && window.visualizerControls && window.visualizerControls.goToTime) {
            const totalDuration = window.visualizerControls.getTotalDuration();
            if (timeValue >= 0 && timeValue <= totalDuration) {
                window.visualizerControls.goToTime(timeValue);
                timeInputField.value = '';
            } else {
                // 잘못된 입력 시 현재 시간 표시
                const minutes = Math.floor(audio.currentTime / 60);
                const seconds = Math.floor(audio.currentTime % 60);
                timeInputField.value = `${minutes}:${seconds.toString().padStart(2, '0')}`;
            }
        }
    }
    
    // 프레임 이동 (동영상 플레이어처럼 즉시 화면 갱신)
    async function moveFrame(direction) {
        // audio 요소를 직접 가져옴 (클로저 문제 방지)
        const audioEl = document.getElementById('audio');
        if (!audioEl) {
            console.warn('[moveFrame] Audio element not found');
            return;
        }
        
        const vc = window.visualizerControls;
        if (!vc) {
            console.warn('[moveFrame] visualizerControls not available');
            return;
        }
        
        // 전역 변수 currentFrameIndex를 직접 사용 (클로저 문제 방지)
        const maxFrames = totalFrames || 1000; // fallback
        const newFrame = Math.max(0, Math.min(maxFrames - 1, currentFrameIndex + direction));
        
        console.log(`[moveFrame] direction=${direction}, currentFrame=${currentFrameIndex}, newFrame=${newFrame}`);
        
        // 재생 중이면 먼저 일시정지
        if (!audioEl.paused) {
            audioEl.pause();
            isPlaying = false;
            if (playPauseIcon) {
                playPauseIcon.className = 'fas fa-play';
            }
        }
        
        // 핑거링 네비게이션 상태 리셋 (프레임 수동 이동 시)
        currentNavigationIndex = -1;
        selectedFingeringKey = null;
        selectedFingeringFrameIndex = -1;
        
        // goToFrame을 사용하여 정확한 프레임으로 이동 (오디오 + 화면 동기화)
        if (vc.goToFrame) {
            vc.goToFrame(newFrame);
            console.log(`[moveFrame] goToFrame(${newFrame}) completed`);
        }
    }
    
    // 이전/다음 프레임 버튼 (React에서 처리하므로 여기서는 선택적)
    if (prevFrameBtn) {
        prevFrameBtn.addEventListener('click', () => {
            moveFrame(-1);
        });
    }
    
    if (nextFrameBtn) {
        nextFrameBtn.addEventListener('click', () => {
            moveFrame(1);
        });
    }
    
    // 시간 입력 이동 버튼
    if (goToTimeBtn) {
        goToTimeBtn.addEventListener('click', goToTime);
    }
    
    // 시간 입력 필드에서 Enter 키
    if (timeInput) {
        timeInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                goToTime();
            }
        });
    }
    
    // 키보드 이벤트 리스너 중복 등록 방지
    if (keyboardListenerSetup) {
        console.log('Keyboard shortcuts already initialized, skipping...');
        return;
    }
    
    console.log('Keyboard shortcuts initialized');
    
    // 키보드 입력 표시 함수
    let keyOverlayTimeout = null;
    function showKeyOverlay(keyName, description) {
        const overlay = document.getElementById('keyOverlay');
        if (!overlay) return;
        
        // 이전 타이머 취소
        if (keyOverlayTimeout) {
            clearTimeout(keyOverlayTimeout);
        }
        
        // 내용 설정
        overlay.innerHTML = `<span class="key-name">${keyName}</span> <span class="key-label">${description}</span>`;
        overlay.classList.remove('hidden');
        
        // 1.5초 후 숨김
        keyOverlayTimeout = setTimeout(() => {
            overlay.classList.add('hidden');
        }, 1500);
    }
    
    // 키보드 단축키 이벤트 리스너
    keyboardListener = (e) => {
        // 입력 필드 체크: INPUT이나 TEXTAREA에 포커스가 있으면 대부분의 단축키 무시
        const isInputField = e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA';
        const isTimeInput = e.target.id === 'timeInput';
        
        // e.code를 사용하여 한글 입력 모드에서도 단축키가 작동하도록 함
        // e.key는 IME 영향을 받지만, e.code는 물리적 키 위치를 반환
        const code = e.code;
        
        // 키 반복 방지 (핑거링/프레임 네비게이션 키에 대해서만)
        const navigationKeyCodes = ['ArrowLeft', 'ArrowRight', 'KeyA', 'KeyD', 'KeyQ', 'KeyE'];
        if (e.repeat && navigationKeyCodes.includes(code)) {
            e.preventDefault();
            return;
        }
        
        // ? 키: 도움말 토글 (입력 필드에서도 작동)
        if (e.key === '?' && !e.shiftKey) {
            e.preventDefault();
            const popup = document.getElementById('keyboardShortcutsPopup');
            if (popup) {
                popup.classList.toggle('hidden');
            }
            return;
        }
        
        // 도움말 팝업이 열려있으면 다른 단축키 무시
        const popup = document.getElementById('keyboardShortcutsPopup');
        if (popup && !popup.classList.contains('hidden')) {
            if (e.key === 'Escape') {
                popup.classList.add('hidden');
            }
            return;
        }
        
        // 입력 필드에서 a, d, q, e, w, s 키는 일반 입력으로 허용 (단축키 작동 안 함)
        if (isInputField && !isTimeInput) {
            const textInputKeyCodes = ['KeyA', 'KeyD', 'KeyQ', 'KeyE', 'KeyW', 'KeyS', 'KeyF'];
            if (textInputKeyCodes.includes(code)) {
                return; // 입력 필드에서는 일반 문자로 입력
            }
        }
        
        // e.code 기반 단축키 처리 (한글 입력 모드 호환)
        switch(code) {
            case 'KeyA': // A: Previous Fingering (WASD style)
                e.preventDefault();
                goToPrevFingering();
                showKeyOverlay('A', 'Prev Fingering');
                return;
                
            case 'KeyD': // D: Next Fingering (WASD style)
                e.preventDefault();
                goToNextFingering();
                showKeyOverlay('D', 'Next Fingering');
                return;
            
            case 'KeyQ': // Q: Previous Frame
                e.preventDefault();
                console.log('[Keyboard] Q pressed - moving to previous frame');
                moveFrame(-1);
                showKeyOverlay('Q', 'Prev Frame');
                return;
            
            case 'KeyE': // E: Next Frame
                e.preventDefault();
                console.log('[Keyboard] E pressed - moving to next frame');
                moveFrame(1);
                showKeyOverlay('E', 'Next Frame');
                return;
                
            case 'KeyW': // W: 카메라 구도 전환 (이전) - 화살표 ↑와 동일
                e.preventDefault();
                if (window.visualizerControls) {
                    const viewPresets = ['top', 'front', 'back'];
                    const currentPreset = window.visualizerControls.cameraSettings.preset;
                    const currentIdx = viewPresets.indexOf(currentPreset);
                    const newIdx = currentIdx <= 0 ? viewPresets.length - 1 : currentIdx - 1;
                    const newPreset = viewPresets[newIdx];
                    window.visualizerControls.updateCamera(newPreset);
                    showKeyOverlay('W', `${newPreset.charAt(0).toUpperCase() + newPreset.slice(1)} View`);
                    window.dispatchEvent(new CustomEvent('cameraPresetChange', { detail: newPreset }));
                }
                return;
                
            case 'KeyS': // S: 카메라 구도 전환 (다음) - 화살표 ↓와 동일
                e.preventDefault();
                if (window.visualizerControls) {
                    const viewPresets = ['top', 'front', 'back'];
                    const currentPreset = window.visualizerControls.cameraSettings.preset;
                    const currentIdx = viewPresets.indexOf(currentPreset);
                    const newIdx = (currentIdx + 1) % viewPresets.length;
                    const newPreset = viewPresets[newIdx];
                    window.visualizerControls.updateCamera(newPreset);
                    showKeyOverlay('S', `${newPreset.charAt(0).toUpperCase() + newPreset.slice(1)} View`);
                    window.dispatchEvent(new CustomEvent('cameraPresetChange', { detail: newPreset }));
                }
                return;
                
            case 'KeyF': // F: 핑거링 표시 토글
                e.preventDefault();
                if (window.visualizerControls) {
                    const currentValue = window.visualizerControls.getShowFingering();
                    window.visualizerControls.setShowFingering(!currentValue);
                    showKeyOverlay('F', currentValue ? 'Hide Fingering' : 'Show Fingering');
                    const showFingeringBtn = document.getElementById('showFingeringBtn');
                    if (showFingeringBtn) {
                        showFingeringBtn.click();
                    }
                }
                return;
                
            case 'KeyM': // M: 음소거 토글
                e.preventDefault();
                const audioEl = document.getElementById('audio');
                const volumeControl = document.getElementById('volumeControl');
                if (audioEl) {
                    if (audioEl.muted || audioEl.volume === 0) {
                        // Unmute
                        audioEl.muted = false;
                        audioEl.volume = audioEl._previousVolume || 1;
                        if (volumeControl) volumeControl.value = audioEl.volume;
                        showKeyOverlay('M', 'Unmuted');
                        // React 상태 업데이트
                        window.dispatchEvent(new CustomEvent('muteChange', { detail: false }));
                    } else {
                        // Mute
                        audioEl._previousVolume = audioEl.volume;
                        audioEl.muted = true;
                        audioEl.volume = 0;
                        if (volumeControl) volumeControl.value = 0;
                        showKeyOverlay('M', 'Muted');
                        // React 상태 업데이트
                        window.dispatchEvent(new CustomEvent('muteChange', { detail: true }));
                    }
                }
                return;
        }
        
        // e.key 기반 단축키 처리 (영문 입력 모드 및 특수키)
        switch(e.key) {
            case ' ': // Space: 재생/일시정지
                if (isInputField && !isTimeInput) return; // 입력 필드에서는 무시
                e.preventDefault();
                if (isPlaying) {
                    audio.pause();
                    isPlaying = false;
                    if (playPauseIcon) {
                        playPauseIcon.className = 'fas fa-play';
                    }
                    showKeyOverlay('Space', 'Pause');
                } else {
                    audio.play();
                    isPlaying = true;
                    if (playPauseIcon) {
                        playPauseIcon.className = 'fas fa-pause';
                    }
                    // 재생 시작 시 핑거링 네비게이션 상태 리셋
                    currentNavigationIndex = -1;
                    selectedFingeringKey = null;
                    selectedFingeringFrameIndex = -1;
                    showKeyOverlay('Space', 'Play');
                }
                break;
                
            case 'ArrowLeft': // ←: Previous Fingering
                if (isInputField && !isTimeInput) return; // 입력 필드에서는 무시
                e.preventDefault();
                goToPrevFingering();
                showKeyOverlay('←', 'Prev Fingering');
                break;
                
            case 'ArrowRight': // →: Next Fingering
                if (isInputField && !isTimeInput) return; // 입력 필드에서는 무시
                e.preventDefault();
                goToNextFingering();
                showKeyOverlay('→', 'Next Fingering');
                break;
                
            // a, d, q, e, w, s, f 키는 위의 e.code switch에서 처리됨 (한글 입력 모드 호환)
                
            case 'Escape': // ESC: 핑거링 선택 해제
                selectedFingeringKey = null;
                highlightSelectedFingering();
                showKeyOverlay('ESC', 'Deselect');
                break;
                
            case '[': // [: 재생 속도 감소
                e.preventDefault();
                if (window.visualizerControls) {
                    const speeds = [0.1, 0.25, 0.5, 1, 1.5, 2];
                    const currentRate = window.visualizerControls.getPlaybackRate();
                    const currentIndex = speeds.indexOf(currentRate);
                    if (currentIndex > 0) {
                        const newRate = speeds[currentIndex - 1];
                        window.visualizerControls.setPlaybackRate(newRate);
                        showKeyOverlay('[', `Speed ${newRate}x`);
                        // React 상태 업데이트를 위해 이벤트 발생
                        window.dispatchEvent(new CustomEvent('playbackRateChange', { detail: newRate }));
                    }
                }
                break;
                
            case ']': // ]: 재생 속도 증가
                e.preventDefault();
                if (window.visualizerControls) {
                    const speeds = [0.1, 0.25, 0.5, 1, 1.5, 2];
                    const currentRate = window.visualizerControls.getPlaybackRate();
                    const currentIndex = speeds.indexOf(currentRate);
                    if (currentIndex < speeds.length - 1) {
                        const newRate = speeds[currentIndex + 1];
                        window.visualizerControls.setPlaybackRate(newRate);
                        showKeyOverlay(']', `Speed ${newRate}x`);
                        // React 상태 업데이트를 위해 이벤트 발생
                        window.dispatchEvent(new CustomEvent('playbackRateChange', { detail: newRate }));
                    }
                }
                break;
                
            case 'ArrowUp': // ↑: 카메라 구도 전환 (이전) - W는 e.code switch에서 처리
                e.preventDefault();
                if (window.visualizerControls) {
                    const viewPresets = ['top', 'front', 'back'];
                    const currentPreset = window.visualizerControls.cameraSettings.preset;
                    const currentIdx = viewPresets.indexOf(currentPreset);
                    const newIdx = currentIdx <= 0 ? viewPresets.length - 1 : currentIdx - 1;
                    const newPreset = viewPresets[newIdx];
                    window.visualizerControls.updateCamera(newPreset);
                    showKeyOverlay('↑', `${newPreset.charAt(0).toUpperCase() + newPreset.slice(1)} View`);
                    window.dispatchEvent(new CustomEvent('cameraPresetChange', { detail: newPreset }));
                }
                break;
                
            case 'ArrowDown': // ↓: 카메라 구도 전환 (다음) - S는 e.code switch에서 처리
                e.preventDefault();
                if (window.visualizerControls) {
                    const viewPresets = ['top', 'front', 'back'];
                    const currentPreset = window.visualizerControls.cameraSettings.preset;
                    const currentIdx = viewPresets.indexOf(currentPreset);
                    const newIdx = (currentIdx + 1) % viewPresets.length;
                    const newPreset = viewPresets[newIdx];
                    window.visualizerControls.updateCamera(newPreset);
                    showKeyOverlay('↓', `${newPreset.charAt(0).toUpperCase() + newPreset.slice(1)} View`);
                    window.dispatchEvent(new CustomEvent('cameraPresetChange', { detail: newPreset }));
                }
                break;
        }
    };
    
    // 키보드 이벤트 리스너 등록
    document.addEventListener('keydown', keyboardListener);
    keyboardListenerSetup = true;
    
    // 오디오 재생 상태 추적
    audio.addEventListener('play', () => {
        isPlaying = true;
        if (playPauseIcon) {
            playPauseIcon.className = 'fas fa-pause';
        }
    });
    
    audio.addEventListener('pause', () => {
        isPlaying = false;
        if (playPauseIcon) {
            playPauseIcon.className = 'fas fa-play';
        }
    });
    
    // 페이지 로드 시 Assign 창 자동 열기
    setTimeout(() => {
        showAssignFingeringFromAppBar();
    }, 1000);
}

/**
 * 핑거링 편집을 위한 이벤트 리스너 설정
 */
function setupFingeringEditListeners() {
    if (editListenersSetup) {
        console.log("Edit listeners already setup");
        return; // 이미 설정되어 있으면 중복 방지
    }
    
    const canvas = document.getElementById('threeCanvas');
    if (!canvas) {
        console.error("Canvas not found for edit listeners");
        return;
    }
    
    console.log("Setting up fingering edit listeners");
    
    // 클릭 이벤트 리스너 (capture phase에서 먼저 처리)
    canvas.addEventListener('click', onCanvasClick, true);
    canvas.addEventListener('mousemove', onCanvasMouseMove, false);
    
    editListenersSetup = true;
}

/**
 * 교차하는 스프라이트들 중 마우스와 2D 화면 거리가 가장 가까운 스프라이트 찾기
 */
function findClosestSpriteByScreenDistance(intersects, mouseNDC, camera) {
    if (intersects.length === 0) return null;
    
    let closestSprite = null;
    let minDistance = Infinity;
    
    for (const intersect of intersects) {
        const sprite = intersect.object;
        // 스프라이트의 3D 위치를 화면상 2D 좌표 (NDC)로 투영
        const screenPos = sprite.position.clone().project(camera);
        
        // 마우스 위치와의 2D 거리 계산
        const dx = screenPos.x - mouseNDC.x;
        const dy = screenPos.y - mouseNDC.y;
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < minDistance) {
            minDistance = distance;
            closestSprite = sprite;
        }
    }
    
    return closestSprite;
}

/**
 * 캔버스 클릭 이벤트 핸들러
 */
function onCanvasClick(event) {
    if (!raycaster || !currentCamera || !currentRenderer) return;
    
    const canvas = event.target;
    const rect = canvas.getBoundingClientRect();
    
    // 마우스 좌표를 정규화된 디바이스 좌표로 변환 (-1 ~ +1)
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    // Raycaster 업데이트
    raycaster.setFromCamera(mouse, currentCamera);
    
    // 1. 먼저 핑거링 스프라이트 클릭 확인
    const sprites = Object.values(fingerTipSprites).filter(sprite => sprite.visible);
    const spriteIntersects = raycaster.intersectObjects(sprites, false);
    
    if (spriteIntersects.length > 0) {
        // 마우스와 2D 화면 거리가 가장 가까운 스프라이트 선택
        const clickedSprite = findClosestSpriteByScreenDistance(spriteIntersects, mouse, currentCamera);
        const fingeringInfo = spriteToFingeringMap.get(clickedSprite);
        
        if (fingeringInfo) {
            if (fingeringInfo.hasFingering) {
                // 기존 핑거링 편집
            showFingeringEditPopup(event.clientX, event.clientY, fingeringInfo);
            } else {
                // 빈 손가락 클릭 → 핑거링 추가
                showAddFingeringPopup(event.clientX, event.clientY, fingeringInfo);
            }
            return;
        }
    }
    
    // 2. 피아노 키 클릭 확인 (핑거링 추가용)
    const keyObjects = Object.values(keyObjs).filter(obj => obj && obj.children && obj.children[0]);
    const keyIntersects = raycaster.intersectObjects(keyObjects, true);
    
    if (keyIntersects.length > 0) {
        // 클릭된 키 찾기
        let clickedKeyIndex = -1;
        for (const [index, keyObj] of Object.entries(keyObjs)) {
            if (keyObj && keyIntersects[0].object.parent === keyObj) {
                clickedKeyIndex = parseInt(index);
                break;
            }
        }
        
        if (clickedKeyIndex >= 0) {
            // 현재 프레임에서 이 키에 대한 핑거링이 있는지 확인
            const currentFingering = editedFingeringData[currentFrameIndex] || fingeringData[currentFrameIndex] || [];
            const existingFingering = currentFingering.find(f => f.key_index === clickedKeyIndex);
            
            if (!existingFingering) {
                // 핑거링이 없으면 추가 팝업 표시
                // 비어있는 구간 정보 찾기
                let segmentInfo = null;
                const missingSegments = findMissingFingeringSegments();
                for (const segment of missingSegments) {
                    const segmentStartFrame = Math.floor(segment.start_time * FPS);
                    const segmentEndFrame = Math.floor(segment.end_time * FPS);
                    if (segment.keyIndex === clickedKeyIndex && 
                        segmentStartFrame <= currentFrameIndex && 
                        currentFrameIndex <= segmentEndFrame) {
                        segmentInfo = {
                            startFrame: segmentStartFrame,
                            endFrame: segmentEndFrame,
                            keyIndex: clickedKeyIndex
                        };
                        break;
                    }
                }
                
                // 임시 fingeringInfo 객체 생성 (hand와 fingerName은 나중에 선택)
                const tempFingeringInfo = {
                    frameIndex: currentFrameIndex,
                    hand: 'left', // 기본값, 나중에 선택 가능
                    fingerName: 'thumb', // 기본값, 나중에 선택 가능
                    fingerNumber: 1, // 기본값, 나중에 선택 가능
                    keyIndex: clickedKeyIndex,
                    segmentInfo: segmentInfo
                };
                
                showAddFingeringPopup(event.clientX, event.clientY, tempFingeringInfo);
            }
        }
    }
}

/**
 * 캔버스 마우스 이동 이벤트 핸들러 (호버 효과)
 */
function onCanvasMouseMove(event) {
    if (!raycaster || !currentCamera) return;
    
    const canvas = event.target;
    const rect = canvas.getBoundingClientRect();
    
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    
    raycaster.setFromCamera(mouse, currentCamera);
    
    const sprites = Object.values(fingerTipSprites).filter(sprite => sprite.visible);
    const intersects = raycaster.intersectObjects(sprites, false);
    
    // 모든 스프라이트의 호버 효과 제거
    Object.values(fingerTipSprites).forEach(sprite => {
        if (sprite.material) {
            const info = spriteToFingeringMap.get(sprite);
            // 핑거링이 있으면 불투명, 없으면 약간 보이게 복원
            sprite.material.opacity = (info && info.hasFingering) ? 1.0 : 0.15;
        }
    });
    
    // 호버된 스프라이트 강조 (마우스와 2D 거리가 가장 가까운 스프라이트 선택)
    if (intersects.length > 0) {
        const hoveredSprite = findClosestSpriteByScreenDistance(intersects, mouse, currentCamera);
        if (hoveredSprite && hoveredSprite.material) {
            const info = spriteToFingeringMap.get(hoveredSprite);
            if (info && info.hasFingering) {
                hoveredSprite.material.opacity = 0.8;
            } else {
                // 빈 손가락은 호버 시 선명하게 보이게
                hoveredSprite.material.opacity = 0.9;
            }
        }
        canvas.style.cursor = 'pointer';
    } else {
        canvas.style.cursor = 'default';
    }
}

/**
 * AI 어노테이션 배지 표시 (화면 상단)
 */
function showAiAnnotationBadge() {
    hideAiAnnotationBadge(); // 기존 배지 제거
    
    const badge = document.createElement('div');
    badge.id = 'aiAnnotationBadge';
    badge.style.cssText = `
        position: fixed;
        top: 10px;
        right: 10px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: bold;
        z-index: 9999;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        display: flex;
        align-items: center;
        gap: 8px;
    `;
    badge.innerHTML = `
        <span style="font-size: 18px;">🤖</span>
        <span>AI Annotation (r0) - AI Prior</span>
    `;
    document.body.appendChild(badge);
}

/**
 * AI 어노테이션 배지 숨기기
 */
function hideAiAnnotationBadge() {
    const badge = document.getElementById('aiAnnotationBadge');
    if (badge) {
        badge.remove();
    }
}

/**
 * AI 어노테이션 경고 메시지 표시
 */
function showAiAnnotationWarning() {
    // 기존 경고 제거
    const existingWarning = document.getElementById('aiAnnotationWarning');
    if (existingWarning) {
        existingWarning.remove();
    }
    
    const warning = document.createElement('div');
    warning.id = 'aiAnnotationWarning';
    warning.style.cssText = `
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: linear-gradient(135deg, #ff6b6b, #ee5a5a);
        color: white;
        padding: 20px 30px;
        border-radius: 12px;
        font-size: 16px;
        font-weight: bold;
        z-index: 10001;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        text-align: center;
        animation: fadeIn 0.3s ease;
    `;
    warning.innerHTML = `
        <div style="font-size: 24px; margin-bottom: 10px;">🤖 AI Annotation (r0)</div>
        <div style="font-weight: normal; margin-bottom: 15px;">
            이 피스는 AI가 보정한 어노테이션입니다.<br>
            AI prior를 기반으로 수정할 수 있습니다.
        </div>
        <button onclick="this.parentElement.remove()" style="
            background: white;
            color: #ee5a5a;
            border: none;
            padding: 8px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: bold;
        ">확인</button>
    `;
    document.body.appendChild(warning);
    
    // 3초 후 자동 제거
    setTimeout(() => {
        if (warning.parentElement) {
            warning.remove();
        }
    }, 3000);
}

/**
 * 핑거링 추가 팝업 표시 (빈 손가락 클릭 시)
 */
function showAddFingeringPopup(x, y, fingeringInfo) {
    // r0 (AI prior)도 수정 가능 - 어노테이터가 AI 보정 결과를 수정할 수 있음
    
    // 기존 팝업 제거
    const existingPopup = document.getElementById('fingeringEditPopup');
    if (existingPopup) {
        existingPopup.remove();
    }
    
    const frameIndex = fingeringInfo.frameIndex;
    const hand = fingeringInfo.hand || 'left'; // 기본값
    const fingerName = fingeringInfo.fingerName || 'thumb'; // 기본값
    const defaultFingerNumber = fingeringInfo.fingerNumber || 1; // 기본값
    
    // 현재 프레임에서 눌린 키 가져오기
    const frameData = preloadedFrames[frameIndex];
    if (!frameData || !frameData.pressed_keys) {
        console.warn("No pressed keys data for frame", frameIndex);
        return;
    }
    
    const pressedKeys = frameData.pressed_keys
        .map((v, i) => ({ index: i, pressed: parseFloat(v) > 0.5 }))
        .filter(k => k.pressed)
        .map(k => k.index);
    
    if (pressedKeys.length === 0) {
        console.warn("No keys pressed in frame", frameIndex);
        return;
    }
    
    // 이미 핑거링이 있는 키 제외
    const currentFingering = editedFingeringData[frameIndex] || fingeringData[frameIndex] || [];
    const assignedKeys = new Set(currentFingering.map(f => f.key_index));
    const availableKeys = pressedKeys.filter(k => !assignedKeys.has(k));
    
    if (availableKeys.length === 0) {
        console.warn("All pressed keys already have fingering");
        return;
    }
    
    // 선택된 손가락 번호 (기본값은 클릭한 손가락)
    let selectedFingerNumber = defaultFingerNumber;
    let selectedKeyIndex = null;
    
    // Ambiguous 버튼 컨테이너 (나중에 표시)
    let ambiguousButtonContainer = null;
    
    // 선택 완료 확인 및 Ambiguous 버튼 표시
    function checkSelectionComplete() {
        if (selectedFingerNumber !== null && selectedKeyIndex !== null) {
            // 둘 다 선택되면 Ambiguous 버튼 표시
            if (ambiguousButtonContainer) {
                ambiguousButtonContainer.style.display = 'flex';
            }
        } else {
            if (ambiguousButtonContainer) {
                ambiguousButtonContainer.style.display = 'none';
            }
        }
    }
    
    // 최종 저장 함수 (Ambiguous 선택 시 호출)
    function saveWithAmbiguous(isAmbiguous) {
        if (selectedFingerNumber === null || selectedKeyIndex === null) return;
        
        const selectedFingerName = FINGER_NUMBER_TO_NAME[selectedFingerNumber];
        
        // 비어있는 구간 정보 찾기 (해당 프레임과 키에 대해)
        let segmentInfo = null;
        const missingSegments = findMissingFingeringSegments();
        for (const segment of missingSegments) {
            const segmentStartFrame = Math.floor(segment.start_time * FPS);
            const segmentEndFrame = Math.floor(segment.end_time * FPS);
            if (segment.keyIndex === selectedKeyIndex && 
                segmentStartFrame <= frameIndex && 
                frameIndex <= segmentEndFrame) {
                segmentInfo = {
                    startFrame: segmentStartFrame,
                    endFrame: segmentEndFrame,
                    keyIndex: selectedKeyIndex
                };
                break;
            }
        }
        
        addFingering(frameIndex, hand, selectedFingerName, selectedFingerNumber, selectedKeyIndex, isAmbiguous, segmentInfo);
        popup.remove();
    }
    
    // 팝업 생성
    const popup = document.createElement('div');
    popup.id = 'fingeringEditPopup';
    popup.style.cssText = `
        position: fixed;
        left: ${x + 10}px;
        top: ${y + 10}px;
        background: rgba(20, 20, 30, 0.95);
        border: 1px solid rgba(100, 255, 218, 0.4);
        border-radius: 12px;
        padding: 16px;
        z-index: 10000;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        min-width: 240px;
    `;
    
    // 제목
    const title = document.createElement('div');
    title.style.cssText = `
        color: #64ffda;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    `;
    title.textContent = `Add: ${hand}`;
    popup.appendChild(title);
    
    // 손가락 번호 선택 섹션
    const fingerLabel = document.createElement('div');
    fingerLabel.style.cssText = `
        color: rgba(255, 255, 255, 0.6);
        font-size: 11px;
        margin-bottom: 8px;
    `;
    fingerLabel.textContent = 'Finger number:';
    popup.appendChild(fingerLabel);
    
    // 손가락 번호 버튼 컨테이너
    const fingerContainer = document.createElement('div');
    fingerContainer.style.cssText = `
        display: flex;
        gap: 4px;
        margin-bottom: 16px;
    `;
    
    const fingerNames = ['👍', '☝️', '🖕', '💍', '🤙'];
    const fingerButtons = [];
    
    for (let i = 1; i <= 5; i++) {
        const btn = document.createElement('button');
        const isSelected = i === selectedFingerNumber;
        btn.style.cssText = `
            width: 36px;
            height: 36px;
            border: 2px solid ${isSelected ? '#64ffda' : 'rgba(100, 255, 218, 0.3)'};
            border-radius: 8px;
            background: ${isSelected ? 'rgba(100, 255, 218, 0.3)' : 'rgba(255, 255, 255, 0.1)'};
            color: white;
            font-size: 14px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
        `;
        btn.textContent = i;
        btn.dataset.finger = i;
        fingerButtons.push(btn);
        
        btn.addEventListener('click', () => {
            selectedFingerNumber = i;
            // 버튼 상태 업데이트
            fingerButtons.forEach((b, idx) => {
                const isNowSelected = (idx + 1) === selectedFingerNumber;
                b.style.borderColor = isNowSelected ? '#64ffda' : 'rgba(100, 255, 218, 0.3)';
                b.style.background = isNowSelected ? 'rgba(100, 255, 218, 0.3)' : 'rgba(255, 255, 255, 0.1)';
            });
            checkSelectionComplete();
        });
        
        fingerContainer.appendChild(btn);
    }
    
    popup.appendChild(fingerContainer);
    
    // 키 선택 섹션
    const keyLabel = document.createElement('div');
    keyLabel.style.cssText = `
        color: rgba(255, 255, 255, 0.6);
        font-size: 11px;
        margin-bottom: 8px;
    `;
    keyLabel.textContent = 'Select key:';
    popup.appendChild(keyLabel);
    
    // 키 버튼 컨테이너
    const keyContainer = document.createElement('div');
    keyContainer.style.cssText = `
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    `;
    
    const keyButtons = [];
    
    availableKeys.forEach(keyIndex => {
        const btn = document.createElement('button');
        btn.style.cssText = `
            width: 40px;
            height: 36px;
            border: 1px solid rgba(100, 255, 218, 0.3);
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        `;
        btn.textContent = keyIndex;
        btn.dataset.keyIndex = keyIndex;
        keyButtons.push(btn);
        
        btn.addEventListener('click', () => {
            selectedKeyIndex = keyIndex;
            // 키 버튼 상태 업데이트
            keyButtons.forEach(b => {
                const isNowSelected = parseInt(b.dataset.keyIndex) === selectedKeyIndex;
                b.style.borderColor = isNowSelected ? '#64ffda' : 'rgba(100, 255, 218, 0.3)';
                b.style.background = isNowSelected ? 'rgba(100, 255, 218, 0.3)' : 'rgba(255, 255, 255, 0.1)';
            });
            checkSelectionComplete();
        });
        
        btn.addEventListener('mouseenter', function() {
            if (parseInt(this.dataset.keyIndex) !== selectedKeyIndex) {
                this.style.background = 'rgba(100, 255, 218, 0.2)';
                this.style.borderColor = 'rgba(100, 255, 218, 0.6)';
            }
        });
        
        btn.addEventListener('mouseleave', function() {
            if (parseInt(this.dataset.keyIndex) !== selectedKeyIndex) {
                this.style.background = 'rgba(255, 255, 255, 0.1)';
                this.style.borderColor = 'rgba(100, 255, 218, 0.3)';
            }
        });
        
        keyContainer.appendChild(btn);
    });
    
    popup.appendChild(keyContainer);
    
    // Ambiguous Yes/No 버튼 컨테이너
    ambiguousButtonContainer = document.createElement('div');
    ambiguousButtonContainer.style.cssText = `
        display: none;
        flex-direction: column;
        gap: 8px;
        margin-top: 16px;
        padding-top: 12px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
    `;
    
    // 라벨
    const ambiguousLabel = document.createElement('div');
    ambiguousLabel.style.cssText = `
        color: rgba(255, 255, 255, 0.9);
        font-size: 13px;
        font-weight: 500;
        text-align: center;
        margin-bottom: 8px;
    `;
    ambiguousLabel.textContent = 'Is this fingering clear?';
    ambiguousButtonContainer.appendChild(ambiguousLabel);
    
    // 버튼 컨테이너
    const btnRow = document.createElement('div');
    btnRow.style.cssText = `
        display: flex;
        gap: 12px;
        justify-content: center;
    `;
    
    // Yes (확실함) 버튼 - 초록색
    const yesBtn = document.createElement('button');
    yesBtn.style.cssText = `
        width: 56px;
        height: 44px;
        border: 2px solid rgba(102, 187, 106, 0.6);
        border-radius: 10px;
        background: rgba(102, 187, 106, 0.2);
        color: #81c784;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    yesBtn.textContent = 'Yes';
    yesBtn.title = 'Clear fingering';
    yesBtn.addEventListener('mouseenter', () => {
        yesBtn.style.background = 'rgba(102, 187, 106, 0.4)';
        yesBtn.style.borderColor = 'rgba(102, 187, 106, 1)';
        yesBtn.style.transform = 'scale(1.05)';
    });
    yesBtn.addEventListener('mouseleave', () => {
        yesBtn.style.background = 'rgba(102, 187, 106, 0.2)';
        yesBtn.style.borderColor = 'rgba(102, 187, 106, 0.6)';
        yesBtn.style.transform = 'scale(1)';
    });
    yesBtn.addEventListener('click', () => saveWithAmbiguous(false));  // Yes = clear = not ambiguous
    btnRow.appendChild(yesBtn);
    
    // No (애매함) 버튼 - 주황색
    const noBtn = document.createElement('button');
    noBtn.style.cssText = `
        width: 56px;
        height: 44px;
        border: 2px solid rgba(255, 193, 7, 0.6);
        border-radius: 10px;
        background: rgba(255, 193, 7, 0.2);
        color: #ffca28;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.15s ease;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    noBtn.textContent = 'No';
    noBtn.title = 'Ambiguous fingering';
    noBtn.addEventListener('mouseenter', () => {
        noBtn.style.background = 'rgba(255, 193, 7, 0.4)';
        noBtn.style.borderColor = 'rgba(255, 193, 7, 1)';
        noBtn.style.transform = 'scale(1.05)';
    });
    noBtn.addEventListener('mouseleave', () => {
        noBtn.style.background = 'rgba(255, 193, 7, 0.2)';
        noBtn.style.borderColor = 'rgba(255, 193, 7, 0.6)';
        noBtn.style.transform = 'scale(1)';
    });
    noBtn.addEventListener('click', () => saveWithAmbiguous(true));  // No = not clear = ambiguous
    btnRow.appendChild(noBtn);
    
    ambiguousButtonContainer.appendChild(btnRow);
    popup.appendChild(ambiguousButtonContainer);
    
    // 닫기 버튼
    const closeBtn = document.createElement('button');
    closeBtn.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        background: none;
        border: none;
        color: rgba(255, 255, 255, 0.5);
        cursor: pointer;
        font-size: 16px;
        padding: 4px;
    `;
    closeBtn.innerHTML = '×';
    closeBtn.addEventListener('click', () => popup.remove());
    popup.appendChild(closeBtn);
    
    // 외부 클릭 시 닫기
    setTimeout(() => {
        document.addEventListener('click', function closePopup(e) {
            if (!popup.contains(e.target)) {
                popup.remove();
                document.removeEventListener('click', closePopup);
            }
        });
    }, 100);
    
    document.body.appendChild(popup);
}

// Assign 팝업 전역 상태
let assignPopupFrameChangeHandler = null;

/**
 * Assign 팝업 업데이트 (프레임 변경 시 호출)
 */
function refreshAssignPopup() {
    const popup = document.getElementById('assignFingeringPopup');
    if (popup && popup.style.display !== 'none') {
        // 팝업이 열려있으면 제거 후 다시 생성
        popup.remove();
        showAssignFingeringFromAppBar(true); // skipToggle = true
    }
}

/**
 * AppBar에서 핑거링 할당 팝업 표시 (토글 방식, 우측 상단 고정)
 * @param {boolean} skipToggle - true이면 토글 로직 건너뛰고 바로 생성
 */
function showAssignFingeringFromAppBar(skipToggle = false) {
    // 토글: 이미 존재하면 보이기/숨기기 토글
    const existingPopup = document.getElementById('assignFingeringPopup');
    if (existingPopup && !skipToggle) {
        if (existingPopup.style.display === 'none') {
            existingPopup.style.display = 'block';
            refreshAssignPopup(); // 다시 열릴 때 현재 프레임으로 갱신
        } else {
            existingPopup.style.display = 'none';
        }
        return;
    }
    
    // 기존 팝업 제거 (skipToggle인 경우)
    if (existingPopup) {
        existingPopup.remove();
    } else {
        // 팝업이 처음 생성될 때 (existingPopup이 없을 때) 선택 상태 초기화
        // Hand와 Finger만 초기화 (Key 선택은 프레임별로 다르므로 항상 리셋됨)
        assignPopupState.selectedHand = null;
        assignPopupState.selectedFingerNumber = null;
    }
    
    const audio = document.getElementById('audio');
    const currentTime = audio ? audio.currentTime : 0;
    // 선택된 핑거링 프레임이 있으면 해당 프레임 사용, 없으면 오디오 기준
    const frameIndex = selectedFingeringFrameIndex >= 0 ? selectedFingeringFrameIndex : Math.floor(currentTime * FPS);
    
    // 현재 프레임에서 활성화된 MIDI 노트 가져오기 (각 노트를 개별적으로 유지)
    const activeNotes = midiNotes.filter(note => 
        note.onset_frame <= frameIndex && frameIndex < note.offset_frame
    ).sort((a, b) => a.key_idx - b.key_idx);
    
    // pressed_keys도 확인하여 MIDI 노트에 없는 키 추가
    const frameData = preloadedFrames[frameIndex];
    if (frameData && frameData.pressed_keys) {
        const pressedKeys = frameData.pressed_keys
            .map((v, i) => ({ index: i, pressed: parseFloat(v) > 0.5 }))
            .filter(k => k.pressed)
            .map(k => k.index);
        
        // MIDI 노트에 없는 pressed_keys 추가
        const midiKeyIndices = new Set(activeNotes.map(n => n.key_idx));
        pressedKeys.forEach(keyIdx => {
            if (!midiKeyIndices.has(keyIdx)) {
                // MIDI 노트에 없는 키 - pressed_keys 기반으로 추가
                activeNotes.push({ 
                    key_idx: keyIdx, 
                    onset_frame: frameIndex, 
                    offset_frame: frameIndex + 1,
                    fromPressedKeys: true  // pressed_keys에서 추가됨 표시
                });
            }
        });
        
        // 다시 정렬
        activeNotes.sort((a, b) => a.key_idx - b.key_idx);
    }
    
    // 활성 키가 없어도 팝업은 열린 상태 유지 (프레임 이동하면 업데이트됨)
    
    // 각 MIDI 노트에 대해 해당 onset_frame에서 핑거링 정보 확인
    // 같은 키의 다른 노트들도 개별적으로 핑거링 상태를 확인
    const noteFingeringMap = new Map(); // "key_onset" -> fingering info
    activeNotes.forEach(note => {
        const noteKey = `${note.key_idx}_${note.onset_frame}`;
        // 해당 노트의 onset_frame에서 핑거링 데이터 확인
        const onsetFingering = editedFingeringData[note.onset_frame] || fingeringData[note.onset_frame] || [];
        const fingering = onsetFingering.find(f => f.key_index === note.key_idx);
        if (fingering) {
            noteFingeringMap.set(noteKey, fingering);
        }
    });
    
    // 현재 선택된 네비게이션 포인트의 키 인덱스 (하이라이트용)
    let highlightKeyIndex = -1;
    if (selectedFingeringKey) {
        if (selectedFingeringKey.startsWith('missing_')) {
            highlightKeyIndex = parseInt(selectedFingeringKey.split('_')[1], 10);
        } else {
            const parts = selectedFingeringKey.split('_');
            if (parts.length >= 3) {
                highlightKeyIndex = parseInt(parts[2], 10);
            }
        }
    }
    
    // 전역 상태에서 선택 값 참조 (getter/setter 함수 사용)
    const getState = () => assignPopupState;
    const setState = (key, value) => { assignPopupState[key] = value; };
    
    // 선택된 노트가 현재 프레임에서 시작하는지 확인하는 함수
    function shouldShowRangeOption() {
        if (getState().selectedOnsetFrame === null) return false;
        return getState().selectedOnsetFrame < frameIndex;  // 노트 시작이 현재 프레임보다 이전인 경우에만 표시
    }
    
    // 자동 추가 및 닫기 함수 (addFingering에서 기존 핑거링 자동 교체됨)
    function tryAutoAdd() {
        const state = getState();
        if (state.selectedHand && state.selectedFingerNumber !== null && state.selectedKeyIndex !== null) {
            const selectedFingerName = FINGER_NUMBER_TO_NAME[state.selectedFingerNumber];
            
            // 비어있는 구간 정보 찾기 (해당 프레임과 키에 대해)
            let segmentInfo = null;
            const missingSegments = findMissingFingeringSegments();
            for (const segment of missingSegments) {
                const segmentStartFrame = Math.floor(segment.start_time * FPS);
                const segmentEndFrame = Math.floor(segment.end_time * FPS);
                if (segment.keyIndex === state.selectedKeyIndex && 
                    segmentStartFrame <= frameIndex && 
                    frameIndex <= segmentEndFrame) {
                    segmentInfo = {
                        startFrame: segmentStartFrame,
                        endFrame: segmentEndFrame,
                        keyIndex: state.selectedKeyIndex
                    };
                    break;
                }
            }
            
            // 새 핑거링 추가
            addFingering(frameIndex, state.selectedHand, selectedFingerName, state.selectedFingerNumber, state.selectedKeyIndex, state.ambiguousState, segmentInfo, state.selectedOnsetFrame, state.applyFromCurrentFrame);
            
            // 팝업 닫지 않고 손가락 선택만 리셋 (손과 키는 유지)
            setState('selectedFingerNumber', null);
            fingerButtons.forEach(fb => fb.classList.remove('selected'));
            
            // 팝업 콘텐츠 갱신
            refreshAssignPopup();
        }
    }
    
    // 팝업 생성 (우측 상단 고정)
    const popup = document.createElement('div');
    popup.id = 'assignFingeringPopup';
    popup.style.cssText = `
        position: fixed;
        right: 20px;
        top: 70px;
        background: rgba(20, 20, 30, 0.95);
        border: 1px solid rgba(100, 255, 218, 0.4);
        border-radius: 12px;
        padding: 16px;
        z-index: 10000;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        min-width: 260px;
        max-height: calc(100vh - 150px);
        overflow-y: auto;
    `;
    
    // 제목
    const title = document.createElement('div');
    title.style.cssText = `
        color: #64ffda;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-align: center;
    `;
    const assignedCount = activeNotes.filter(note => noteFingeringMap.has(`${note.key_idx}_${note.onset_frame}`)).length;
    const missingCount = activeNotes.length - assignedCount;
    title.innerHTML = `Add Fingering <span style="font-size:11px;opacity:0.7;">(Frame ${frameIndex})</span>
        <div style="font-size:10px;margin-top:4px;font-weight:400;letter-spacing:0;">
            <span style="color:#ff9800;">●</span> ${assignedCount} assigned
            <span style="margin-left:8px;color:#ff5252;">●</span> ${missingCount} missing
        </div>`;
    popup.appendChild(title);
    
    // ====== UI 요소 생성 (appendChild는 나중에 순서대로 호출) ======
    
    // 1. 키 선택 섹션
    const keyLabel = document.createElement('div');
    keyLabel.style.cssText = `
        color: rgba(255, 255, 255, 0.6);
        font-size: 11px;
        margin-bottom: 8px;
    `;
    keyLabel.textContent = 'Select key:';
    
    // 2. Apply Range 옵션 섹션 (항상 표시, 조건에 따라 From Current 비활성화)
    const rangeOptionContainer = document.createElement('div');
    rangeOptionContainer.id = 'assignRangeOption';
    rangeOptionContainer.style.cssText = `
        margin-bottom: 16px;
        padding: 10px;
        border-radius: 8px;
        background: rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.3);
    `;
    
    // Range 옵션 UI 업데이트 함수
    function updateRangeOptionUI() {
        const state = getState();
        const hasKeySelected = state.selectedOnsetFrame !== null;
        const canApplyFromCurrent = hasKeySelected && state.selectedOnsetFrame < frameIndex;
        
        // "From Current" 선택했는데 더 이상 적용할 수 없는 경우 리셋
        if (state.applyFromCurrentFrame && !canApplyFromCurrent) {
            setState('applyFromCurrentFrame', false);
        }
        
        // 노트 정보 표시
        const noteInfoText = hasKeySelected 
            ? `Note: ${state.selectedOnsetFrame} ~ ${state.selectedOffsetFrame} | Current: ${frameIndex}`
            : `Current Frame: ${frameIndex}`;
        
        rangeOptionContainer.innerHTML = `
            <div style="color: rgba(255, 255, 255, 0.7); font-size: 10px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">
                Apply Range
            </div>
            <div style="color: rgba(255, 255, 255, 0.5); font-size: 9px; margin-bottom: 8px;">
                ${noteInfoText}
            </div>
            <div style="display: flex; flex-direction: column; gap: 6px;">
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; color: rgba(255, 255, 255, 0.9); font-size: 11px;">
                    <input type="radio" name="assignApplyRange" value="full" ${!state.applyFromCurrentFrame ? 'checked' : ''} style="accent-color: rgba(102, 126, 234, 0.8);">
                    <span>Entire Note ${hasKeySelected ? `(${state.selectedOnsetFrame}~${state.selectedOffsetFrame})` : '(select key to see range)'}</span>
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; color: rgba(255, 255, 255, 0.9); font-size: 11px;">
                    <input type="radio" name="assignApplyRange" value="fromCurrent" ${state.applyFromCurrentFrame ? 'checked' : ''} style="accent-color: rgba(102, 126, 234, 0.8);">
                    <span>From Current Frame ${hasKeySelected ? `(${frameIndex}~${state.selectedOffsetFrame})` : `(from ${frameIndex})`}</span>
                </label>
            </div>
        `;
        
        // 라디오 버튼 이벤트 리스너
        const radios = rangeOptionContainer.querySelectorAll('input[type="radio"]');
        radios.forEach(radio => {
            radio.addEventListener('change', (e) => {
                setState('applyFromCurrentFrame', e.target.value === 'fromCurrent');
            });
        });
    }
    
    // 초기 UI 렌더링
    updateRangeOptionUI();
    
    // 3. 손 선택 섹션
    const handLabel = document.createElement('div');
    handLabel.style.cssText = `
        color: rgba(255, 255, 255, 0.6);
        font-size: 11px;
        margin-bottom: 8px;
    `;
    handLabel.textContent = 'Select hand:';
    
    const handContainer = document.createElement('div');
    handContainer.style.cssText = `
        display: flex;
        gap: 8px;
        margin-bottom: 16px;
    `;
    
    const handButtons = [];
    ['left', 'right'].forEach(hand => {
        const btn = document.createElement('button');
        const isSelected = getState().selectedHand === hand;
        btn.style.cssText = `
            flex: 1;
            padding: 10px;
            border: 2px solid ${isSelected ? '#64ffda' : 'rgba(100, 255, 218, 0.3)'};
            border-radius: 8px;
            background: ${isSelected ? 'rgba(100, 255, 218, 0.3)' : 'rgba(255, 255, 255, 0.1)'};
            color: white;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
        `;
        btn.textContent = hand === 'left' ? '🤚 Left' : 'Right 🤚';
        btn.dataset.hand = hand;
        handButtons.push(btn);
        
        btn.addEventListener('click', () => {
            setState('selectedHand', hand);
            handButtons.forEach(b => {
                const isNowSelected = b.dataset.hand === getState().selectedHand;
                b.style.borderColor = isNowSelected ? '#64ffda' : 'rgba(100, 255, 218, 0.3)';
                b.style.background = isNowSelected ? 'rgba(100, 255, 218, 0.3)' : 'rgba(255, 255, 255, 0.1)';
            });
            tryAutoAdd();
        });
        
        handContainer.appendChild(btn);
    });
    
    // 4. 손가락 번호 선택 섹션
    const fingerLabel = document.createElement('div');
    fingerLabel.style.cssText = `
        color: rgba(255, 255, 255, 0.6);
        font-size: 11px;
        margin-bottom: 8px;
    `;
    fingerLabel.textContent = 'Finger number:';
    
    const fingerContainer = document.createElement('div');
    fingerContainer.style.cssText = `
        display: flex;
        gap: 4px;
        margin-bottom: 16px;
    `;
    
    const fingerButtons = [];
    for (let i = 1; i <= 5; i++) {
        const btn = document.createElement('button');
        const isSelected = getState().selectedFingerNumber === i;
        btn.style.cssText = `
            width: 40px;
            height: 40px;
            border: 2px solid ${isSelected ? '#64ffda' : 'rgba(100, 255, 218, 0.3)'};
            border-radius: 8px;
            background: ${isSelected ? 'rgba(100, 255, 218, 0.3)' : 'rgba(255, 255, 255, 0.1)'};
            color: white;
            font-size: 16px;
            font-weight: 700;
            cursor: pointer;
            transition: all 0.15s ease;
        `;
        btn.textContent = i;
        btn.dataset.finger = i;
        fingerButtons.push(btn);
        
        btn.addEventListener('click', () => {
            setState('selectedFingerNumber', i);
            fingerButtons.forEach((b, idx) => {
                const isNowSelected = (idx + 1) === getState().selectedFingerNumber;
                b.style.borderColor = isNowSelected ? '#64ffda' : 'rgba(100, 255, 218, 0.3)';
                b.style.background = isNowSelected ? 'rgba(100, 255, 218, 0.3)' : 'rgba(255, 255, 255, 0.1)';
            });
            tryAutoAdd();
        });
        
        fingerContainer.appendChild(btn);
    }
    
    // 5. Ambiguous 체크박스 섹션
    const ambiguousContainer = document.createElement('div');
    ambiguousContainer.style.cssText = `
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 16px;
    `;
    ambiguousContainer.innerHTML = `
        <input type="checkbox" id="assignAmbiguousToggle" style="
            width: 18px;
            height: 18px;
            cursor: pointer;
            accent-color: rgba(255, 193, 7, 0.8);
        ">
        <label for="assignAmbiguousToggle" style="
            color: rgba(255, 255, 255, 0.9);
            font-size: 12px;
            cursor: pointer;
            user-select: none;
            flex: 1;
        ">Ambiguous</label>
    `;
    
    // Ambiguous 체크박스 이벤트 리스너
    const ambiguousToggle = ambiguousContainer.querySelector('input[type="checkbox"]');
    if (ambiguousToggle) {
        // 전역 상태에서 현재 값으로 초기화
        ambiguousToggle.checked = getState().ambiguousState;
        ambiguousToggle.addEventListener('change', (e) => {
            setState('ambiguousState', e.target.checked);
        });
    }
    
    // ====== UI 요소를 원하는 순서로 추가 ======
    // 순서: Key → Range → Ambiguous → Hand → Finger
    popup.appendChild(keyLabel);
    
    const keyContainer = document.createElement('div');
    keyContainer.style.cssText = `
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    `;
    
    const keyButtons = [];
    // selectedNoteKey는 전역 상태 (selectedKeyIndex + selectedOnsetFrame) 조합으로 표현
    
    // 활성 키가 없을 때 메시지 표시
    if (activeNotes.length === 0) {
        const noKeysMsg = document.createElement('div');
        noKeysMsg.style.cssText = `
            color: rgba(255, 255, 255, 0.5);
            font-size: 12px;
            padding: 20px 10px;
            text-align: center;
            width: 100%;
        `;
        noKeysMsg.textContent = 'No active keys in this frame';
        keyContainer.appendChild(noKeysMsg);
    }
    
    activeNotes.forEach((note, noteIdx) => {
        const keyIndex = note.key_idx;
        const noteKey = `${note.key_idx}_${note.onset_frame}`;
        const isAssigned = noteFingeringMap.has(noteKey);
        const existingInfo = isAssigned ? noteFingeringMap.get(noteKey) : null;
        const isHighlighted = keyIndex === highlightKeyIndex; // 현재 네비게이션 포인트 키
        
        // 같은 키가 여러 번 나타나는지 확인
        const sameKeyNotes = activeNotes.filter(n => n.key_idx === keyIndex);
        const hasDuplicateKey = sameKeyNotes.length > 1;
        const noteOrder = hasDuplicateKey ? sameKeyNotes.indexOf(note) + 1 : 0;
        
        const btn = document.createElement('button');
        // 색상 결정: 할당됨(오렌지), 미할당(빨간색) - 사용자가 클릭해야 청록색 선택 표시
        let borderColor, bgColor;
        if (isAssigned) {
            // 핑거링 있음 - 오렌지
            borderColor = 'rgba(255, 152, 0, 0.7)';
            bgColor = 'rgba(255, 152, 0, 0.2)';
        } else {
            // 핑거링 없음 (Missing) - 빨간색
            borderColor = 'rgba(255, 82, 82, 0.7)';
            bgColor = 'rgba(255, 82, 82, 0.2)';
        }
        
        btn.style.cssText = `
            width: 52px;
            height: 44px;
            border: 2px solid ${borderColor};
            border-radius: 6px;
            background: ${bgColor};
            color: white;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.15s ease;
            position: relative;
        `;
        
        // 이미 할당된 키는 기존 핑거링 정보 표시
        let keyLabel = `${keyIndex}`;
        if (hasDuplicateKey) {
            // 같은 키가 여러 번 나타나면 순서 표시
            keyLabel = `${keyIndex}<span style="position:absolute;bottom:2px;left:3px;font-size:8px;color:rgba(255,255,255,0.5);">#${noteOrder}</span>`;
        }
        
        if (isAssigned) {
            btn.innerHTML = `${keyLabel}<span style="position:absolute;top:2px;right:3px;font-size:9px;color:#ff9800;">${existingInfo.hand[0].toUpperCase()}${existingInfo.finger}</span>`;
        } else {
            // 미할당 키는 "!" 표시
            btn.innerHTML = `${keyLabel}<span style="position:absolute;top:2px;right:3px;font-size:10px;color:#ff5252;">!</span>`;
        }
        btn.dataset.keyIndex = keyIndex;
        btn.dataset.noteKey = noteKey;
        btn.dataset.onsetFrame = note.onset_frame;
        btn.dataset.isAssigned = isAssigned;
        btn.dataset.originalBorderColor = borderColor;
        btn.dataset.originalBgColor = bgColor;
        keyButtons.push(btn);
        
        btn.addEventListener('click', () => {
            setState('selectedKeyIndex', keyIndex);
            setState('selectedOnsetFrame', note.onset_frame);  // 선택된 노트의 onset_frame 저장
            setState('selectedOffsetFrame', note.offset_frame);  // 선택된 노트의 offset_frame 저장
            const currentNoteKey = `${getState().selectedKeyIndex}_${getState().selectedOnsetFrame}`;
            keyButtons.forEach(b => {
                const isNowSelected = b.dataset.noteKey === currentNoteKey;
                if (isNowSelected) {
                    b.style.borderColor = '#64ffda';
                    b.style.background = 'rgba(100, 255, 218, 0.3)';
                } else {
                    b.style.borderColor = b.dataset.originalBorderColor;
                    b.style.background = b.dataset.originalBgColor;
                }
            });
            
            // Range 옵션 UI 업데이트
            updateRangeOptionUI();
            
            tryAutoAdd();
        });
        
        btn.addEventListener('mouseenter', function() {
            const currentNoteKey = getState().selectedKeyIndex !== null ? `${getState().selectedKeyIndex}_${getState().selectedOnsetFrame}` : null;
            if (this.dataset.noteKey !== currentNoteKey) {
                this.style.background = 'rgba(100, 255, 218, 0.2)';
                this.style.borderColor = 'rgba(100, 255, 218, 0.6)';
            }
        });
        
        btn.addEventListener('mouseleave', function() {
            const currentNoteKey = getState().selectedKeyIndex !== null ? `${getState().selectedKeyIndex}_${getState().selectedOnsetFrame}` : null;
            if (this.dataset.noteKey !== currentNoteKey) {
                this.style.background = this.dataset.originalBgColor;
                this.style.borderColor = this.dataset.originalBorderColor;
            }
        });
        
        keyContainer.appendChild(btn);
    });
    popup.appendChild(keyContainer);
    
    // Range 옵션 (키 선택 후 동적 표시)
    popup.appendChild(rangeOptionContainer);
    
    // Ambiguous 체크박스 (Hand/Finger 선택 전에 설정)
    popup.appendChild(ambiguousContainer);
    
    // Hand 선택
    popup.appendChild(handLabel);
    popup.appendChild(handContainer);
    
    // Finger 선택
    popup.appendChild(fingerLabel);
    popup.appendChild(fingerContainer);
    
    // 닫기 버튼
    const closeBtn = document.createElement('button');
    closeBtn.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        background: none;
        border: none;
        color: rgba(255, 255, 255, 0.5);
        cursor: pointer;
        font-size: 18px;
        padding: 4px;
    `;
    closeBtn.innerHTML = '×';
    closeBtn.addEventListener('click', () => popup.style.display = 'none');
    popup.appendChild(closeBtn);
    
    // ESC 키로 숨기기
    const escHandler = (e) => {
        if (e.key === 'Escape' && popup.style.display !== 'none') {
            popup.style.display = 'none';
        }
    };
    document.addEventListener('keydown', escHandler);
    
    // Assign 버튼으로만 토글 - 외부 클릭 시 닫지 않음
    
    // frameChange 이벤트 리스너 등록 (프레임 변경 시 팝업 갱신)
    if (assignPopupFrameChangeHandler) {
        window.removeEventListener('frameChange', assignPopupFrameChangeHandler);
    }
    assignPopupFrameChangeHandler = () => {
        refreshAssignPopup();
    };
    window.addEventListener('frameChange', assignPopupFrameChangeHandler);
    
    document.body.appendChild(popup);
}

/**
 * 핑거링 편집 팝업 표시
 */
function showFingeringEditPopup(x, y, fingeringInfo) {
    // r0 (AI prior)도 수정 가능 - 어노테이터가 AI 보정 결과를 수정할 수 있음
    
    // 기존 팝업 제거
    const existingPopup = document.getElementById('fingeringEditPopup');
    if (existingPopup) {
        existingPopup.remove();
    }
    
    // 현재 프레임이 속한 노트 범위 찾기 (pressed_keys 기반)
    const keyIndex = fingeringInfo.fingering ? fingeringInfo.fingering.key_index : null;
    let noteRange = null;
    let showRangeOption = false;
    
    if (keyIndex !== null && keyIndex !== undefined) {
        noteRange = findNoteRangeFromPressedKeys(fingeringInfo.frameIndex, keyIndex);
        // 현재 프레임이 노트 시작이 아닌 경우에만 옵션 표시
        if (noteRange && noteRange.onset < fingeringInfo.frameIndex) {
            showRangeOption = true;
        }
    }
    
    // 팝업 생성
    const popup = document.createElement('div');
    popup.id = 'fingeringEditPopup';
    popup.style.cssText = `
        position: fixed;
        left: ${x + 10}px;
        top: ${y + 10}px;
        background: linear-gradient(180deg, rgba(20, 20, 30, 0.98) 0%, rgba(30, 30, 40, 0.98) 100%);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
        z-index: 1000;
        min-width: 240px;
        font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    `;
    
    const fingerNames = {
        'thumb': 'Thumb',
        'index': 'Index',
        'middle': 'Middle',
        'ring': 'Ring',
        'pinky': 'Pinky'
    };
    
    // 현재 핑거링의 ambiguous 상태 확인
    const currentFingering = fingeringInfo.fingering;
    const isAmbiguous = currentFingering && currentFingering.ambiguous === true;
    
    // 범위 옵션 HTML 생성
    let rangeOptionHtml = '';
    if (showRangeOption && noteRange) {
        rangeOptionHtml = `
        <div style="margin-bottom: 12px; padding: 10px; border-radius: 8px; background: rgba(102, 126, 234, 0.1); border: 1px solid rgba(102, 126, 234, 0.3);">
            <div style="color: rgba(255, 255, 255, 0.7); font-size: 10px; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px;">
                Apply Range
            </div>
            <div style="color: rgba(255, 255, 255, 0.5); font-size: 9px; margin-bottom: 8px;">
                Note: ${noteRange.onset} ~ ${noteRange.offset} | Current: ${fingeringInfo.frameIndex}
            </div>
            <div style="display: flex; flex-direction: column; gap: 6px;">
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; color: rgba(255, 255, 255, 0.9); font-size: 11px;">
                    <input type="radio" name="applyRange" value="full" checked style="accent-color: rgba(102, 126, 234, 0.8);">
                    <span>Entire Note (${noteRange.onset}~${noteRange.offset})</span>
                </label>
                <label style="display: flex; align-items: center; gap: 8px; cursor: pointer; color: rgba(255, 255, 255, 0.9); font-size: 11px;">
                    <input type="radio" name="applyRange" value="fromCurrent" style="accent-color: rgba(102, 126, 234, 0.8);">
                    <span>From Current (${fingeringInfo.frameIndex}~${noteRange.offset})</span>
                </label>
            </div>
        </div>
        `;
    }
    
    // 간단한 번호 선택 버튼들
    popup.innerHTML = `
        <div style="color: white; margin-bottom: 10px; font-weight: 600; font-size: 13px; text-align: center;">
            Select Finger
        </div>
        ${rangeOptionHtml}
        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px; padding: 8px; border-radius: 8px; background: rgba(255, 255, 255, 0.05);">
            <input type="checkbox" id="ambiguousToggle" ${isAmbiguous ? 'checked' : ''} style="
                width: 18px;
                height: 18px;
                cursor: pointer;
                accent-color: rgba(255, 193, 7, 0.8);
            ">
            <label for="ambiguousToggle" style="
                color: rgba(255, 255, 255, 0.9);
                font-size: 12px;
                cursor: pointer;
                user-select: none;
                flex: 1;
            ">Ambiguous</label>
        </div>
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; margin-bottom: 12px;">
            ${[1, 2, 3, 4, 5].map(num => `
                <button class="finger-btn" data-finger="${num}" style="
                    padding: 12px 8px;
                    border-radius: 8px;
                    background: ${fingeringInfo.fingerNumber === num ? 'linear-gradient(135deg, rgba(102, 126, 234, 0.5) 0%, rgba(118, 75, 162, 0.5) 100%)' : 'rgba(255, 255, 255, 0.1)'};
                    border: 1px solid ${fingeringInfo.fingerNumber === num ? 'rgba(102, 126, 234, 0.6)' : 'rgba(255, 255, 255, 0.2)'};
                    color: white;
                    cursor: pointer;
                    font-size: 16px;
                    font-weight: ${fingeringInfo.fingerNumber === num ? '700' : '500'};
                    transition: all 0.2s ease;
                ">${num}</button>
            `).join('')}
        </div>
        <button id="deleteFingeringBtn" style="
            width: 100%;
            padding: 8px;
            border-radius: 8px;
            background: rgba(255, 59, 48, 0.2);
            border: 1px solid rgba(255, 59, 48, 0.4);
            color: rgba(255, 59, 48, 1);
            cursor: pointer;
            font-size: 11px;
            font-weight: 500;
            transition: all 0.2s ease;
        ">Delete</button>
    `;
    
    document.body.appendChild(popup);
    
    // 적용 범위 가져오는 헬퍼 함수
    const getApplyFromCurrentFrame = () => {
        if (!showRangeOption) return false;
        const selectedRadio = popup.querySelector('input[name="applyRange"]:checked');
        return selectedRadio && selectedRadio.value === 'fromCurrent';
    };
    
    // Ambiguous 토글 상태 저장
    let ambiguousState = isAmbiguous;
    const ambiguousToggle = document.getElementById('ambiguousToggle');
    ambiguousToggle.addEventListener('change', (e) => {
        ambiguousState = e.target.checked;
        // 토글만 변경해도 즉시 저장 (손가락 번호는 그대로 유지)
        const fromCurrentFrame = getApplyFromCurrentFrame();
        updateFingering(fingeringInfo.frameIndex, fingeringInfo.hand, fingeringInfo.fingerName, fingeringInfo.fingerNumber, ambiguousState, fromCurrentFrame);
    });
    
    // 번호 버튼 이벤트 리스너
    const fingerButtons = popup.querySelectorAll('.finger-btn');
    fingerButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const newFingerNumber = parseInt(btn.dataset.finger);
            const fromCurrentFrame = getApplyFromCurrentFrame();
            updateFingering(fingeringInfo.frameIndex, fingeringInfo.hand, fingeringInfo.fingerName, newFingerNumber, ambiguousState, fromCurrentFrame);
            popup.remove();
        });
        
        btn.addEventListener('mouseenter', function() {
            if (parseInt(this.dataset.finger) !== fingeringInfo.fingerNumber) {
                this.style.background = 'rgba(255, 255, 255, 0.2)';
                this.style.borderColor = 'rgba(255, 255, 255, 0.3)';
            }
        });
        
        btn.addEventListener('mouseleave', function() {
            if (parseInt(this.dataset.finger) !== fingeringInfo.fingerNumber) {
                this.style.background = 'rgba(255, 255, 255, 0.1)';
                this.style.borderColor = 'rgba(255, 255, 255, 0.2)';
            }
        });
    });
    
    // 삭제 버튼 이벤트 리스너
    const deleteBtn = document.getElementById('deleteFingeringBtn');
    deleteBtn.addEventListener('click', () => {
        const fromCurrentFrame = getApplyFromCurrentFrame();
        deleteFingering(fingeringInfo.frameIndex, fingeringInfo.hand, fingeringInfo.fingerName, fromCurrentFrame);
        popup.remove();
        // OrbitControls 다시 활성화
        if (window.orbitControls) {
            window.orbitControls.enabled = true;
        }
    });
    
    deleteBtn.addEventListener('mouseenter', function() {
        this.style.background = 'rgba(255, 59, 48, 0.3)';
    });
    
    deleteBtn.addEventListener('mouseleave', function() {
        this.style.background = 'rgba(255, 59, 48, 0.2)';
    });
    
    // 팝업 외부 클릭 시 닫기
    setTimeout(() => {
        const closePopupHandler = function(e) {
            if (!popup.contains(e.target)) {
                popup.remove();
                document.removeEventListener('click', closePopupHandler);
                // OrbitControls 다시 활성화
                if (window.orbitControls) {
                    window.orbitControls.enabled = true;
                }
            }
        };
        document.addEventListener('click', closePopupHandler);
        
        // 번호 선택 시에도 OrbitControls 다시 활성화
        const originalFingerBtnClick = fingerButtons.forEach.bind(fingerButtons);
        fingerButtons.forEach(btn => {
            const originalClick = btn.onclick;
            btn.addEventListener('click', function() {
                setTimeout(() => {
                    if (window.orbitControls) {
                        window.orbitControls.enabled = true;
                    }
                }, 100);
            });
        });
        
        // 삭제 버튼 클릭 시에도 OrbitControls 다시 활성화
        deleteBtn.addEventListener('click', function() {
            setTimeout(() => {
                if (window.orbitControls) {
                    window.orbitControls.enabled = true;
                }
            }, 100);
        });
    }, 100);
}

/**
 * 현재 프레임이 속한 핑거링 구간(시작~끝 프레임)을 찾습니다.
 * @param {number} frameIndex - 현재 프레임 인덱스
 * @param {string} hand - 손 (left/right)
 * @param {string} fingerName - 손가락 이름
 * @returns {{startFrame: number, endFrame: number, fingerNumber: number} | null}
 */
function findFingeringSegment(frameIndex, hand, fingerName) {
    // 현재 프레임의 핑거링 확인
    const currentFingering = editedFingeringData[frameIndex] || fingeringData[frameIndex] || [];
    const currentMatch = currentFingering.find(f => f.hand === hand && f.finger_name === fingerName);
    
    if (!currentMatch) return null;
    
    const fingerNumber = currentMatch.finger;
    const keyIndex = currentMatch.key_index;  // 건반 인덱스도 함께 추적
    let startFrame = frameIndex;
    let endFrame = frameIndex;
    
    // 시작 프레임 찾기 (뒤로 탐색)
    // hand + finger_name + finger + key_index가 모두 일치하는 연속 구간만 찾음
    for (let i = frameIndex - 1; i >= 0; i--) {
        const frameFingering = editedFingeringData[i] || fingeringData[i] || [];
        const match = frameFingering.find(f => 
            f.hand === hand && 
            f.finger_name === fingerName && 
            f.finger === fingerNumber &&
            f.key_index === keyIndex
        );
        if (match) {
            startFrame = i;
        } else {
            break;
        }
    }
    
    // 끝 프레임 찾기 (앞으로 탐색)
    // Object.keys()는 문자열 배열을 반환하므로 Number로 변환하고 NaN 필터링
    const getMaxFrame = (dataObj) => {
        const keys = Object.keys(dataObj);
        if (keys.length === 0) return 0;
        const numbers = keys.map(k => Number(k)).filter(n => !isNaN(n));
        return numbers.length > 0 ? Math.max(...numbers) : 0;
    };
    
    const maxFrame = Math.max(
        getMaxFrame(fingeringData),
        getMaxFrame(editedFingeringData)
    );
    
    for (let i = frameIndex + 1; i <= maxFrame; i++) {
        const frameFingering = editedFingeringData[i] || fingeringData[i] || [];
        const match = frameFingering.find(f => 
            f.hand === hand && 
            f.finger_name === fingerName && 
            f.finger === fingerNumber &&
            f.key_index === keyIndex
        );
        if (match) {
            endFrame = i;
        } else {
            break;
        }
    }
    
    return { startFrame, endFrame, fingerNumber, keyIndex };
}

/**
 * 손가락 번호를 이름으로 변환
 */
const FINGER_NUMBER_TO_NAME = {
    1: 'thumb',
    2: 'index',
    3: 'middle',
    4: 'ring',
    5: 'pinky'
};

/**
 * 새 핑거링 추가
 * @param {number} frameIndex - 프레임 인덱스
 * @param {string} hand - 손 (left/right)
 * @param {string} fingerName - 손가락 이름
 * @param {number} fingerNumber - 손가락 번호 (1-5)
 * @param {number} keyIndex - 건반 인덱스
 * @param {boolean} ambiguous - ambiguous 여부 (선택적, 기본값: false)
 * @param {Object} segmentInfo - 구간 정보 (선택적, {startFrame, endFrame, keyIndex})
 * @param {number} targetOnsetFrame - 특정 MIDI 노트를 식별하기 위한 onset_frame (선택적, 같은 키의 여러 노트 구분용)
 * @param {boolean} fromCurrentFrame - true면 현재 프레임부터만 적용 (기본: false, 전체 노트 적용)
 */
function addFingering(frameIndex, hand, fingerName, fingerNumber, keyIndex, ambiguous = false, segmentInfo = null, targetOnsetFrame = null, fromCurrentFrame = false) {
    // r0 (AI prior)도 수정 가능 - 어노테이터가 AI 보정 결과를 수정할 수 있음
    
    console.log(`Adding fingering: frame=${frameIndex}, ${hand} ${fingerName} (${fingerNumber}) -> key ${keyIndex}, ambiguous=${ambiguous}, targetOnsetFrame=${targetOnsetFrame}, fromCurrentFrame=${fromCurrentFrame}`);
    
    // 노트 범위 결정
    let startFrame, endFrame;
    
    // 1순위: targetOnsetFrame이 있으면 해당 MIDI 노트 직접 찾기 (가장 정확 - 사용자가 선택한 노트)
    if (targetOnsetFrame !== null) {
        const targetNote = midiNotes.find(note => 
            note.key_idx === keyIndex && note.onset_frame === targetOnsetFrame
        );
        if (targetNote) {
            startFrame = parseInt(targetNote.onset_frame, 10);
            endFrame = parseInt(targetNote.offset_frame, 10);
            console.log(`Using target MIDI note (onset=${targetOnsetFrame}): frames ${startFrame}~${endFrame} for key ${keyIndex}`);
        }
    }
    
    // 2순위: pressed_keys에서 노트 범위 찾기 (실제 프레임 데이터 기반)
    if (startFrame === undefined) {
        const pressedKeyRange = findNoteRangeFromPressedKeys(frameIndex, keyIndex);
        if (pressedKeyRange) {
            startFrame = parseInt(pressedKeyRange.onset, 10);
            endFrame = parseInt(pressedKeyRange.offset, 10);
            if (isNaN(startFrame) || isNaN(endFrame)) {
                console.warn(`Invalid pressed_keys range: ${pressedKeyRange.onset}, ${pressedKeyRange.offset}, using single frame`);
                startFrame = undefined;
            } else {
                console.log(`Using pressed_keys range: frames ${startFrame}~${endFrame} for key ${keyIndex}`);
            }
        }
    }
    
    // 3순위: MIDI 노트 정보 (서버에서 로드된 경우)
    if (startFrame === undefined) {
        const midiNote = findMidiNoteForFrame(frameIndex, keyIndex);
        if (midiNote) {
            startFrame = parseInt(midiNote.onset_frame, 10);
            endFrame = parseInt(midiNote.offset_frame, 10);
            if (isNaN(startFrame) || isNaN(endFrame)) {
                console.warn(`Invalid MIDI note frame values: ${midiNote.onset_frame}, ${midiNote.offset_frame}, using single frame`);
                startFrame = undefined;
            } else {
                console.log(`Found MIDI note: frames ${startFrame}~${endFrame} for key ${keyIndex}`);
            }
        }
    }
    
    // 4순위: 아무것도 없으면 단일 프레임
    if (startFrame === undefined) {
        startFrame = parseInt(frameIndex, 10);
        endFrame = parseInt(frameIndex, 10);
        if (isNaN(startFrame)) {
            console.error(`Invalid frameIndex: ${frameIndex}`);
            return;
        }
        console.log(`No note range found, adding to single frame ${startFrame}`);
    }
    
    // fromCurrentFrame이 true면 현재 프레임부터 시작
    const originalStartFrame = startFrame;
    if (fromCurrentFrame && frameIndex > startFrame) {
        startFrame = parseInt(frameIndex, 10);
        console.log(`Applying from current frame: ${startFrame} (original note start: ${originalStartFrame})`);
    }
    
    // segmentInfo는 더 이상 사용하지 않음 (pressed_keys가 더 정확함)
    
    // 새 핑거링 객체 생성
    const newFingering = {
        key_index: keyIndex,
        hand: hand,
        finger: fingerNumber,
        finger_name: fingerName,
        distance: 0
    };
    
    // ambiguous가 true인 경우에만 추가
    if (ambiguous) {
        newFingering.ambiguous = true;
    }
    
    // 구간의 모든 프레임에 핑거링 추가
    const framesToSave = [];
    
    for (let i = startFrame; i <= endFrame; i++) {
        // i가 정수인지 확인
        const frameIdx = parseInt(i, 10);
        if (isNaN(frameIdx)) {
            console.warn(`Skipping invalid frame index: ${i}`);
            continue;
        }
        
        // 편집된 데이터 초기화 (없으면 원본 복사 + 세션 편집 병합)
        if (!editedFingeringData[frameIdx]) {
            const original = originalFingeringData[frameIdx];
            const source = original !== undefined ? original : (fingeringData[frameIdx] || []);
            console.log(`[addFingering] Frame ${frameIdx}: initializing editedFingeringData from ${original !== undefined ? 'originalFingeringData' : 'fingeringData'}`);
            console.log(`[addFingering] Frame ${frameIdx}: source has ${source.length} items:`, source.map(f => `key${f.key_index}:${f.hand}${f.finger}`).join(', '));
            editedFingeringData[frameIdx] = JSON.parse(JSON.stringify(source));
            
            // 이미 편집된 다른 프레임에서 해당 프레임에 적용될 수 있는 핑거링 병합
            // (해당 노트가 이 프레임을 포함하는 경우)
            for (const otherFrameStr of Object.keys(editedFingeringData)) {
                const otherFrameIdx = parseInt(otherFrameStr, 10);
                if (otherFrameIdx === frameIdx || isNaN(otherFrameIdx)) continue;
                
                const otherFingering = editedFingeringData[otherFrameIdx];
                if (!otherFingering || !Array.isArray(otherFingering)) continue;
                
                for (const f of otherFingering) {
                    if (f.key_index === undefined) continue;
                    
                    // 해당 키의 노트가 현재 프레임을 포함하는지 확인
                    const note = midiNotes.find(n => 
                        n.key_idx === f.key_index && 
                        n.onset_frame <= frameIdx && 
                        frameIdx < n.offset_frame
                    );
                    
                    if (note) {
                        // 현재 프레임에 해당 키의 핑거링이 없으면 추가
                        const existsInCurrent = editedFingeringData[frameIdx].some(cf => cf.key_index === f.key_index);
                        if (!existsInCurrent) {
                            console.log(`[addFingering] Frame ${frameIdx}: merging key${f.key_index}:${f.hand}${f.finger} from frame ${otherFrameIdx}`);
                            editedFingeringData[frameIdx].push(JSON.parse(JSON.stringify(f)));
                        }
                    }
                }
            }
            
            console.log(`[addFingering] Frame ${frameIdx}: after merge has ${editedFingeringData[frameIdx].length} items:`, editedFingeringData[frameIdx].map(f => `key${f.key_index}:${f.hand}${f.finger}`).join(', '));
        } else {
            console.log(`[addFingering] Frame ${frameIdx}: editedFingeringData already exists with ${editedFingeringData[frameIdx].length} items:`, editedFingeringData[frameIdx].map(f => `key${f.key_index}:${f.hand}${f.finger}`).join(', '));
        }
        
        // 해당 키에 이미 핑거링이 있는지 확인
        const existingIndex = editedFingeringData[frameIdx].findIndex(f => f.key_index === keyIndex);
        if (existingIndex !== -1) {
            // 기존 핑거링이 있으면 교체 (덮어쓰기)
            console.log(`[addFingering] Frame ${frameIdx}: replacing key${keyIndex} at index ${existingIndex} (was ${editedFingeringData[frameIdx][existingIndex].hand}${editedFingeringData[frameIdx][existingIndex].finger}) -> ${hand}${fingerNumber}`);
            editedFingeringData[frameIdx][existingIndex] = JSON.parse(JSON.stringify(newFingering));
        } else {
            // 기존 핑거링이 없으면 추가
            console.log(`[addFingering] Frame ${frameIdx}: adding new key${keyIndex} -> ${hand}${fingerNumber}`);
            editedFingeringData[frameIdx].push(JSON.parse(JSON.stringify(newFingering)));
        }
        
        // 최종 상태 로그
        console.log(`[addFingering] Frame ${frameIdx}: final state has ${editedFingeringData[frameIdx].length} items:`, editedFingeringData[frameIdx].map(f => `key${f.key_index}:${f.hand}${f.finger}`).join(', '));
        // frameIndex를 명시적으로 정수로 보장
        framesToSave.push({ frameIndex: frameIdx, fingering: editedFingeringData[frameIdx] });
    }
    
    if (framesToSave.length > 0) {
        console.log(`Fingering added to ${framesToSave.length} frames (${startFrame}~${endFrame})`);
        
        // 저장 (블록 단위로 저장)
        saveFingeringBlock(framesToSave);
        
        // 수정된 핑거링 프레임 업데이트 이벤트 발생
        framesToSave.forEach(f => {
            window.dispatchEvent(new CustomEvent('fingeringEdited', { detail: { frameIndex: f.frameIndex } }));
        });
        
        // Ambiguous 변경 시 별도 이벤트 발생
        if (ambiguous) {
            window.dispatchEvent(new CustomEvent('ambiguousChanged', { 
                detail: { frameIndex: startFrame, isAmbiguous: true } 
            }));
        }
        
        // 화면 업데이트 트리거
        if (window.forceRenderUpdate) {
            window.forceRenderUpdate();
        }
        
        // 현재 프레임의 메시 즉시 업데이트 (스프라이트 반영)
        if (window.visualizerControls && window.visualizerControls.refreshCurrentFrame) {
            window.visualizerControls.refreshCurrentFrame();
        }
    } else {
        console.warn(`No frames updated - all frames already have fingering for key ${keyIndex}`);
    }
}

/**
 * 핑거링 업데이트 (블록 단위)
 * - finger 번호와 finger_name을 함께 업데이트하여 표시 위치도 변경됨
 * @param {number} frameIndex - 현재 프레임
 * @param {string} hand - 손 (left/right)
 * @param {string} fingerName - 현재 손가락 이름
 * @param {number} newFingerNumber - 새 손가락 번호
 * @param {boolean} ambiguous - ambiguous 상태
 * @param {boolean} fromCurrentFrame - true면 현재 프레임부터만 적용 (기본: false, 전체 노트 적용)
 */
function updateFingering(frameIndex, hand, fingerName, newFingerNumber, ambiguous = undefined, fromCurrentFrame = false) {
    // 현재 프레임이 속한 핑거링 구간 찾기
    const segment = findFingeringSegment(frameIndex, hand, fingerName);
    
    if (!segment) {
        console.warn(`No fingering segment found for frame ${frameIndex}, ${hand} ${fingerName}`);
        return;
    }
    
    // 타입 검증 및 변환 (OS 독립적)
    let startFrame = segment.startFrame;
    let endFrame = segment.endFrame;
    const segmentKeyIndex = segment.keyIndex;  // 세그먼트의 key_index 사용
    
    // 숫자로 변환 (플랫폼 독립적 타입 안전성)
    if (typeof startFrame !== 'number' || !Number.isInteger(startFrame)) {
        startFrame = parseInt(startFrame, 10);
        if (isNaN(startFrame)) {
            console.error(`Invalid startFrame: ${segment.startFrame}`);
            return;
        }
    }
    if (typeof endFrame !== 'number' || !Number.isInteger(endFrame)) {
        endFrame = parseInt(endFrame, 10);
        if (isNaN(endFrame)) {
            console.error(`Invalid endFrame: ${segment.endFrame}`);
            return;
        }
    }
    
    // fromCurrentFrame이 true면 현재 프레임부터 시작
    const originalStartFrame = startFrame;
    if (fromCurrentFrame && frameIndex > startFrame) {
        startFrame = parseInt(frameIndex, 10);
        console.log(`Applying from current frame: ${startFrame} (original start: ${originalStartFrame})`);
    }
    
    const frameCount = endFrame - startFrame + 1;
    
    // 새 finger_name 결정 (번호에 맞게 변경)
    const newFingerName = FINGER_NUMBER_TO_NAME[newFingerNumber] || fingerName;
    
    console.log(`Updating fingering block: frames ${startFrame}~${endFrame} (${frameCount} frames), ${hand} ${fingerName} -> ${newFingerNumber} (${newFingerName}), key ${segmentKeyIndex}${fromCurrentFrame ? ' [from current frame]' : ''}`);
    
    // 구간의 모든 프레임 업데이트
    const framesToSave = [];
    
    for (let i = startFrame; i <= endFrame; i++) {
        // 편집된 데이터 가져오기 또는 복사
        // originalFingeringData가 있으면 그것을 사용, 없으면 fingeringData 사용
        if (!editedFingeringData[i]) {
            const original = originalFingeringData[i];
            const source = original !== undefined ? original : (fingeringData[i] || []);
            editedFingeringData[i] = JSON.parse(JSON.stringify(source));
        }
        
        const frameFingering = editedFingeringData[i];
        
        // 기존 핑거링 찾아서 제거 (hand, finger_name, key_index 모두 일치해야 함)
        const oldIndex = frameFingering.findIndex(f => 
            f.hand === hand && f.finger_name === fingerName && f.key_index === segmentKeyIndex
        );
        
        let distance = null;
        let oldAmbiguous = false;
        
        if (oldIndex !== -1) {
            // 기존 데이터에서 distance, ambiguous 보존
            distance = frameFingering[oldIndex].distance;
            oldAmbiguous = frameFingering[oldIndex].ambiguous === true;
            // 기존 핑거링 제거
            frameFingering.splice(oldIndex, 1);
        }
        
        // 같은 손의 같은 새 finger_name + key_index가 이미 있으면 제거 (중복 방지)
        const duplicateIndex = frameFingering.findIndex(f => 
            f.hand === hand && f.finger_name === newFingerName && f.key_index === segmentKeyIndex
        );
        if (duplicateIndex !== -1) {
            frameFingering.splice(duplicateIndex, 1);
        }
        
        // 새 핑거링 추가 (finger와 finger_name 모두 새 값으로)
        const newFingering = {
            hand: hand,
            finger_name: newFingerName,
            finger: newFingerNumber,
            key_index: segmentKeyIndex  // 항상 key_index 포함
        };
        if (distance !== null) newFingering.distance = distance;
        // ambiguous 상태: 명시적으로 전달된 경우 그 값을 사용, 아니면 기존 값 유지
        if (ambiguous !== undefined) {
            newFingering.ambiguous = ambiguous;
        } else {
            newFingering.ambiguous = oldAmbiguous;
        }
        
        frameFingering.push(newFingering);
        
        framesToSave.push({ frameIndex: i, fingering: frameFingering });
    }
    
    // 블록 단위로 저장
    saveFingeringBlock(framesToSave);
    
    // 현재 프레임의 메시 즉시 업데이트 (스프라이트 반영)
    if (window.visualizerControls && window.visualizerControls.refreshCurrentFrame) {
        window.visualizerControls.refreshCurrentFrame();
    }
}

/**
 * 핑거링 삭제 (블록 단위)
 * @param {number} frameIndex - 현재 프레임
 * @param {string} hand - 손 (left/right)
 * @param {string} fingerName - 손가락 이름
 * @param {boolean} fromCurrentFrame - true면 현재 프레임부터만 삭제 (기본: false, 전체 노트 삭제)
 */
function deleteFingering(frameIndex, hand, fingerName, fromCurrentFrame = false) {
    // r0 (AI prior)도 수정 가능 - 어노테이터가 AI 보정 결과를 수정할 수 있음
    
    // 현재 프레임이 속한 핑거링 구간 찾기
    const segment = findFingeringSegment(frameIndex, hand, fingerName);
    
    if (!segment) {
        console.warn(`No fingering segment found for frame ${frameIndex}, ${hand} ${fingerName}`);
        return;
    }
    
    // 타입 검증 및 변환 (OS 독립적)
    let startFrame = segment.startFrame;
    let endFrame = segment.endFrame;
    const segmentKeyIndex = segment.keyIndex;  // 세그먼트의 key_index 사용
    
    // 숫자로 변환 (플랫폼 독립적 타입 안전성)
    if (typeof startFrame !== 'number' || !Number.isInteger(startFrame)) {
        startFrame = parseInt(startFrame, 10);
        if (isNaN(startFrame)) {
            console.error(`Invalid startFrame: ${segment.startFrame}`);
            return;
        }
    }
    if (typeof endFrame !== 'number' || !Number.isInteger(endFrame)) {
        endFrame = parseInt(endFrame, 10);
        if (isNaN(endFrame)) {
            console.error(`Invalid endFrame: ${segment.endFrame}`);
            return;
        }
    }
    
    // fromCurrentFrame이 true면 현재 프레임부터 시작
    const originalStartFrame = startFrame;
    if (fromCurrentFrame && frameIndex > startFrame) {
        startFrame = parseInt(frameIndex, 10);
        console.log(`Deleting from current frame: ${startFrame} (original start: ${originalStartFrame})`);
    }
    
    const frameCount = endFrame - startFrame + 1;
    
    console.log(`Deleting fingering block: frames ${startFrame}~${endFrame} (${frameCount} frames), ${hand} ${fingerName}, key ${segmentKeyIndex}${fromCurrentFrame ? ' [from current frame]' : ''}`);
    
    // 구간의 모든 프레임에서 핑거링 삭제
    const framesToSave = [];
    
    for (let i = startFrame; i <= endFrame; i++) {
        if (!editedFingeringData[i]) {
            // originalFingeringData가 있으면 그것을 사용, 없으면 fingeringData 사용
            const original = originalFingeringData[i];
            const source = original !== undefined ? original : (fingeringData[i] || []);
            editedFingeringData[i] = JSON.parse(JSON.stringify(source));
        }
        
        const frameFingering = editedFingeringData[i];
        // hand, finger_name, key_index 모두 일치하는 핑거링만 삭제
        const index = frameFingering.findIndex(f => 
            f.hand === hand && f.finger_name === fingerName && f.key_index === segmentKeyIndex
        );
        
        if (index !== -1) {
            frameFingering.splice(index, 1);
        }
        
        framesToSave.push({ frameIndex: i, fingering: frameFingering });
    }
    
    // 블록 단위로 저장
    saveFingeringBlock(framesToSave);
    
    // 현재 프레임의 메시 즉시 업데이트 (스프라이트 반영)
    if (window.visualizerControls && window.visualizerControls.refreshCurrentFrame) {
        window.visualizerControls.refreshCurrentFrame();
    }
}

/**
 * 핑거링 변경사항 저장 (블록 단위, 디바운스 적용)
 */
let saveTimeout = null;
let savingStatus = null;
let pendingFramesToSave = [];

function showSavingStatus() {
    if (!savingStatus) {
        savingStatus = document.createElement('div');
        savingStatus.id = 'savingStatus';
        savingStatus.style.cssText = `
            position: fixed;
            top: 70px;
            left: 20px;
            background: rgba(20, 20, 30, 0.9);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            padding: 8px 12px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 12px;
            z-index: 1000;
            display: none;
        `;
        document.body.appendChild(savingStatus);
    }
    return savingStatus;
}

/**
 * 블록 단위로 핑거링 저장
 * @param {Array} frames - [{frameIndex, fingering}, ...]
 */
function saveFingeringBlock(frames) {
    // 기존 대기 중인 프레임에 추가 (중복 제거: 같은 frameIndex는 최신 것으로 덮어쓰기)
    for (const frame of frames) {
        const existingIndex = pendingFramesToSave.findIndex(f => f.frameIndex === frame.frameIndex);
        if (existingIndex !== -1) {
            // 같은 프레임이 이미 있으면 최신 것으로 교체
            pendingFramesToSave[existingIndex] = frame;
        } else {
            // 없으면 추가
            pendingFramesToSave.push(frame);
        }
    }
    
    // 즉시 저장 (디바운스 제거하여 변경 즉시 저장)
    clearTimeout(saveTimeout);
    
    const status = showSavingStatus();
    status.textContent = `Saving ${pendingFramesToSave.length} frames...`;
    status.style.display = 'block';
    status.style.color = 'rgba(255, 255, 255, 0.8)';
    
    // 즉시 저장 실행
    (async () => {
        const framesToSave = [...pendingFramesToSave];
        pendingFramesToSave = [];
        
        if (framesToSave.length === 0) return;
        
        try {
            // URL에서 pieceId를 직접 읽기 (0도 유효한 값)
            const urlParams = new URLSearchParams(window.location.search);
            const urlId = urlParams.get('id');
            const actualPieceId = urlId !== null ? parseInt(urlId) : pieceId;
            
            // 블록 단위로 서버에 저장
            // frameIndex를 정수로 보장 (플랫폼 독립적 타입 안전성)
            const framesToSend = framesToSave.map(f => {
                // frameIndex가 정수인지 확인하고 변환
                let frameIdx = f.frameIndex;
                if (typeof frameIdx !== 'number' || !Number.isInteger(frameIdx)) {
                    console.warn(`Invalid frameIndex type: ${typeof frameIdx}, value: ${frameIdx}, converting to int`);
                    // 객체인 경우 일반적인 키에서 값 추출 시도
                    if (typeof frameIdx === 'object' && frameIdx !== null) {
                        frameIdx = frameIdx.frameIndex || frameIdx.frame_idx || frameIdx.index;
                    }
                    // 최종적으로 정수로 변환
                    frameIdx = parseInt(frameIdx, 10);
                    if (isNaN(frameIdx)) {
                        console.error(`Cannot convert frameIndex to int: ${f.frameIndex}`);
                        throw new Error(`Invalid frameIndex: ${f.frameIndex}`);
                    }
                }
                return {
                    frame_idx: frameIdx,
                    fingering: f.fingering
                };
            });
            
            const response = await fetch(`${BACKEND_URL}/fingering_data/${pieceId}/edit_block`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    frames: framesToSend
                })
            });
            
            if (response.ok) {
                const result = await response.json();
                const startFrame = Math.min(...framesToSave.map(f => f.frameIndex));
                const endFrame = Math.max(...framesToSave.map(f => f.frameIndex));
                console.log(`Fingering block saved: frames ${startFrame}~${endFrame} (${framesToSave.length} frames)`);
                
                // 어노테이션 진행상황도 저장 (마지막으로 편집한 프레임 기준)
                try {
                    const lastFrame = endFrame;
                    const lastTimeSeconds = lastFrame / FPS;
                    await fetch(`${BACKEND_URL}/annotation_progress/${pieceId}`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ last_frame: lastFrame, last_time_seconds: lastTimeSeconds })
                    });
                    console.log(`Annotation progress saved: frame ${lastFrame}`);
                } catch (progressError) {
                    console.warn('Failed to save annotation progress:', progressError);
                }
                
                if (savingStatus) {
                    savingStatus.textContent = `Saved ${framesToSave.length} frames (${startFrame}~${endFrame})`;
                    savingStatus.style.color = 'rgba(76, 175, 80, 1)';
                    setTimeout(() => {
                        if (savingStatus) {
                            savingStatus.style.display = 'none';
                            savingStatus.style.color = 'rgba(255, 255, 255, 0.8)';
                        }
                    }, 2000);
                }
            } else {
                const error = await response.json().catch(() => ({error: 'Unknown error'}));
                console.error(`Failed to save fingering block: ${response.status}`, error);
                if (savingStatus) {
                    savingStatus.textContent = `Save failed: ${error.error || 'Error'}`;
                    savingStatus.style.color = 'rgba(255, 59, 48, 1)';
                    setTimeout(() => {
                        if (savingStatus) {
                            savingStatus.style.display = 'none';
                            savingStatus.style.color = 'rgba(255, 255, 255, 0.8)';
                        }
                    }, 3000);
                }
            }
        } catch (error) {
            console.error("Error saving fingering block:", error);
            if (savingStatus) {
                savingStatus.textContent = `Error: ${error.message}`;
                savingStatus.style.color = 'rgba(255, 59, 48, 1)';
                setTimeout(() => {
                    if (savingStatus) {
                        savingStatus.style.display = 'none';
                        savingStatus.style.color = 'rgba(255, 255, 255, 0.8)';
                    }
                }, 3000);
            }
        }
    })();
}

/**
 * 단일 프레임 핑거링 저장 (레거시 호환용)
 */
function saveFingeringChange(frameIndex, fingering) {
    saveFingeringBlock([{ frameIndex, fingering }]);
}

/**
 * 비어있는 핑거링 구간 찾기 (눌린 키에 핑거링이 없는 경우)
 * @returns {Array<{start_time: number, end_time: number, keyIndex: number}>}
 */
function findMissingFingeringSegments() {
    const segments = [];
    const activeSegments = new Map(); // keyIndex -> {startFrame, startTime}
    
    // 모든 프레임 순회
    // Object.keys()는 문자열 배열을 반환하므로 Number로 변환하고 NaN 필터링
    const getMaxFrame = (dataObj) => {
        const keys = Object.keys(dataObj);
        if (keys.length === 0) return 0;
        const numbers = keys.map(k => Number(k)).filter(n => !isNaN(n));
        return numbers.length > 0 ? Math.max(...numbers) : 0;
    };
    
    const maxFrame = Math.max(
        getMaxFrame(fingeringData),
        getMaxFrame(editedFingeringData),
        getMaxFrame(preloadedFrames)
    );
    
    for (let frameIdx = 0; frameIdx <= maxFrame; frameIdx++) {
        // 프레임 데이터 가져오기
        const frameData = preloadedFrames[frameIdx];
        if (!frameData || !frameData.pressed_keys) continue;
        
        // 눌린 키 인덱스 찾기
        const pressedKeyIndices = [];
        for (let i = 0; i < frameData.pressed_keys.length; i++) {
            if (frameData.pressed_keys[i] > 0) {
                pressedKeyIndices.push(i);
            }
        }
        
        // 현재 프레임의 핑거링 데이터
        const currentFingering = editedFingeringData[frameIdx] || fingeringData[frameIdx] || [];
        const fingeredKeyIndices = new Set(
            currentFingering.map(f => f.key_index !== undefined ? f.key_index : null)
                .filter(idx => idx !== null)
        );
        
        // 눌린 키 중에서 핑거링이 없는 키 찾기
        for (const keyIndex of pressedKeyIndices) {
            if (!fingeredKeyIndices.has(keyIndex)) {
                // 핑거링이 없는 키
                const segmentKey = `key_${keyIndex}`;
                const existing = activeSegments.get(segmentKey);
                
                if (!existing) {
                    // 새로운 구간 시작
                    activeSegments.set(segmentKey, {
                        startFrame: frameIdx,
                        startTime: frameIdx / FPS,
                        keyIndex: keyIndex
                    });
                }
                // 기존 구간 계속 유지
            } else {
                // 핑거링이 있으면 구간 종료
                const segmentKey = `key_${keyIndex}`;
                const existing = activeSegments.get(segmentKey);
                if (existing) {
                    segments.push({
                        start_time: existing.startTime,
                        end_time: (frameIdx - 1) / FPS,
                        keyIndex: existing.keyIndex
                    });
                    activeSegments.delete(segmentKey);
                }
            }
        }
        
        // 이전 프레임에 눌려있던 키가 현재 프레임에서 안 눌려있으면 구간 종료
        for (const [segmentKey, segment] of activeSegments.entries()) {
            const keyIndex = segment.keyIndex;
            if (!pressedKeyIndices.includes(keyIndex)) {
                segments.push({
                    start_time: segment.startTime,
                    end_time: (frameIdx - 1) / FPS,
                    keyIndex: keyIndex
                });
                activeSegments.delete(segmentKey);
            }
        }
    }
    
    // 마지막까지 유지된 구간들 종료
    for (const segment of activeSegments.values()) {
        segments.push({
            start_time: segment.startTime,
            end_time: maxFrame / FPS,
            keyIndex: segment.keyIndex
        });
    }
    
    // 최소 길이 필터링 (0.1초 이상만 표시)
    return segments.filter(s => (s.end_time - s.start_time) >= 0.1);
}

/**
 * 수정된 핑거링 프레임 찾기 (editedFingeringData와 originalFingeringData 비교)
 * @returns {number[]} 수정된 프레임 인덱스 배열
 */
function findEditedFingeringFrames() {
    const editedFrames = new Set();
    
    // 핑거링 정규화 함수 (비교용)
        const normalizeFingering = (fingeringArray) => {
        if (!fingeringArray || !Array.isArray(fingeringArray)) return [];
            return fingeringArray.map(f => ({
                key_index: f.key_index,
                hand: f.hand,
                finger_name: f.finger_name,
                finger: f.finger,
            ambiguous: f.ambiguous === true
            })).sort((a, b) => {
                if (a.hand !== b.hand) return a.hand.localeCompare(b.hand);
                if (a.finger_name !== b.finger_name) return a.finger_name.localeCompare(b.finger_name);
                return (a.key_index || 0) - (b.key_index || 0);
            });
        };
        
    // 두 핑거링 배열이 다른지 비교
    const isDifferent = (arr1, arr2) => {
        const norm1 = normalizeFingering(arr1);
        const norm2 = normalizeFingering(arr2);
        return JSON.stringify(norm1) !== JSON.stringify(norm2);
    };
    
    // 1. 현재 세션에서 수정한 프레임 (editedFingeringData)
    for (const frameIdxStr in editedFingeringData) {
        const frameIdx = parseInt(frameIdxStr, 10);
        const edited = editedFingeringData[frameIdx];
        const original = originalFingeringData[frameIdx] || [];
        
        if (isDifferent(edited, original)) {
            editedFrames.add(frameIdx);
        }
    }
    
    // 2. 서버에서 이미 수정된 프레임 (fingeringData vs originalFingeringData)
    //    현재 세션에서 수정하지 않았지만 이전에 수정된 것들
    for (const frameIdxStr in fingeringData) {
        const frameIdx = parseInt(frameIdxStr, 10);
        
        // 이미 체크한 프레임은 스킵
        if (editedFrames.has(frameIdx)) continue;
        
        // 현재 세션에서 수정하지 않은 경우, 서버 데이터와 원본 비교
        if (!editedFingeringData[frameIdx]) {
            const serverFingering = fingeringData[frameIdx];
            const original = originalFingeringData[frameIdx];
            
            // 원본이 있고 서버 데이터와 다르면 수정된 것
            if (original !== undefined && isDifferent(serverFingering, original)) {
                editedFrames.add(frameIdx);
            }
        }
    }
    
    // Set을 배열로 변환하고 정렬
    return Array.from(editedFrames).sort((a, b) => a - b);
}

/**
 * 시각화 초기화 함수 (중복 실행 방지)
 */
let isInitialized = false;
window.initVisualizer = function() {
    if (isInitialized) {
        console.log("Visualizer already initialized, skipping...");
        return;
    }
    
    isInitialized = true;
    console.log("Initializing visualizer...");
    
    // IndexedDB 없이 메모리만 사용
            main();
};
