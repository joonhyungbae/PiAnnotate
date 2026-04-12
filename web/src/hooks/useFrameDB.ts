import { openDB, type IDBPDatabase } from 'idb';
import type { FrameData } from '../types';

const DB_NAME = 'MeshFramesDB';
const STORE_NAME = 'frames';

interface FrameEntry {
  index: number;
  data: FrameData;
}

// Module level singleton
let db: IDBPDatabase | null = null;
const preloadedFrames = new Map<number, FrameData>();

export async function initDB() {
  // Delete existing DB
  try {
    await indexedDB.deleteDatabase(DB_NAME);
  } catch {
    // ignore
  }

  try {
  db = await openDB(DB_NAME, 1, {
    upgrade(database) {
      if (!database.objectStoreNames.contains(STORE_NAME)) {
        database.createObjectStore(STORE_NAME, { keyPath: 'index' });
      }
    },
  });

  preloadedFrames.clear();
  console.log('IndexedDB initialized');
  return db;
  } catch (error) {
    console.error('Failed to initialize IndexedDB:', error);
    db = null;
    throw error;
  }
}

export async function storeFrame(frameIndex: number, frameData: FrameData) {
  if (!db) {
    console.warn('IndexedDB not initialized, skipping frame storage');
    return;
  }
  
  try {
  const entry: FrameEntry = { index: frameIndex, data: frameData };
  await db.put(STORE_NAME, entry);
  } catch (error) {
    console.error(`Error storing frame ${frameIndex}:`, error);
    // 에러가 발생해도 앱이 계속 작동하도록 에러를 무시하지 않고 로그만 남김
  }
}

export async function getFrame(frameIndex: number): Promise<FrameData | null> {
  if (!db) return null;
  try {
    const entry = await db.get(STORE_NAME, frameIndex) as FrameEntry | undefined;
    return entry?.data ?? null;
  } catch {
    return null;
  }
}

export function getPreloadedFrame(frameIndex: number): FrameData | null {
  return preloadedFrames.get(frameIndex) ?? null;
}

export async function preloadFramesAround(currentFrameIndex: number, totalFrames: number, windowSize = 600) {
  if (!db) return;

  const startFrame = Math.max(0, currentFrameIndex);
  const endFrame = Math.min(totalFrames, currentFrameIndex + windowSize);

  // Load new frames
  for (let i = startFrame; i < endFrame; i++) {
    if (!preloadedFrames.has(i)) {
      try {
        const entry = await db.get(STORE_NAME, i) as FrameEntry | undefined;
        if (entry?.data) {
          preloadedFrames.set(i, entry.data);
        }
      } catch {
        // ignore
      }
    }
  }

  // Remove out of range frames
  for (const key of preloadedFrames.keys()) {
    if (key < startFrame || key >= endFrame) {
      preloadedFrames.delete(key);
    }
  }
}

export function closeDB() {
  db?.close();
  db = null;
  preloadedFrames.clear();
}
