#!/usr/bin/env python3
"""
Mesh data preprocessing script
- Extract mesh data from motion.pkl
- Convert to JSON and apply maximum gzip compression
- Save in a format ready for direct server transmission
"""

import os
import json
import gzip
import pickle
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Configuration
DATASET_PATH = Path("./for_elise/dataset")
COMPRESSION_LEVEL = 9  # Maximum compression

def process_piece(piece_id: int) -> dict:
    """Compress mesh data for a single piece."""
    piece_dir = DATASET_PATH / f"{piece_id:03d}"
    motion_path = piece_dir / "motion.pkl"
    pressed_keys_path = piece_dir / "vis" / "pressed_keys.pkl"
    
    if not motion_path.exists():
        return {"piece_id": piece_id, "status": "skip", "reason": "no motion.pkl"}
    
    # Load data
    with open(motion_path, "rb") as f:
        motion = pickle.load(f)
    
    pressed_keys = None
    if pressed_keys_path.exists():
        with open(pressed_keys_path, "rb") as f:
            pressed_keys = pickle.load(f)
        if not isinstance(pressed_keys, np.ndarray):
            pressed_keys = np.array(pressed_keys)
    
    # Get frame count
    left_verts = motion.get("left", {}).get("mano_params", {}).get("verts")
    right_verts = motion.get("right", {}).get("mano_params", {}).get("verts")
    left_joints = motion.get("left", {}).get("joints")
    right_joints = motion.get("right", {}).get("joints")
    
    n_frames = 0
    if left_verts is not None:
        n_frames = max(n_frames, len(left_verts))
    if right_verts is not None:
        n_frames = max(n_frames, len(right_verts))
    
    if n_frames == 0:
        return {"piece_id": piece_id, "status": "skip", "reason": "no frames"}
    
    # Build all frames data
    frames = []
    for i in range(n_frames):
        frame_data = {
            "frame_idx": i,
            "left_vertices": np.round(left_verts[i], 4).tolist() if left_verts is not None and i < len(left_verts) else [],
            "right_vertices": np.round(right_verts[i], 4).tolist() if right_verts is not None and i < len(right_verts) else [],
            "left_joints": np.round(left_joints[i].flatten(), 4).tolist() if left_joints is not None and i < len(left_joints) else [],
            "right_joints": np.round(right_joints[i].flatten(), 4).tolist() if right_joints is not None and i < len(right_joints) else [],
            "pressed_keys": pressed_keys[i].tolist() if pressed_keys is not None and i < len(pressed_keys) else [],
        }
        frames.append(frame_data)
    
    # Save as gzip-compressed JSON in vis folder
    output_path = piece_dir / "vis" / "mesh_data.json.gz"
    json_data = json.dumps(frames, separators=(',', ':'))  # Minified JSON
    
    with gzip.open(output_path, 'wt', compresslevel=COMPRESSION_LEVEL, encoding='utf-8') as f:
        f.write(json_data)
    
    # Calculate sizes
    json_size = len(json_data.encode('utf-8'))
    compressed_size = output_path.stat().st_size
    ratio = (1 - compressed_size / json_size) * 100 if json_size > 0 else 0
    
    return {
        "piece_id": piece_id,
        "status": "ok",
        "n_frames": n_frames,
        "json_size": json_size,
        "compressed_size": compressed_size,
        "ratio": ratio
    }


def main():
    # Find all pieces
    piece_dirs = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(DATASET_PATH / d) and d.isdigit()
    ])
    piece_ids = [int(d) for d in piece_dirs]
    
    print(f"Found {len(piece_ids)} pieces")
    print(f"Output: for_elise/dataset/{{id}}/vis/mesh_data.json.gz")
    print(f"Compression level: {COMPRESSION_LEVEL} (max)")
    print()
    
    # Process with multiprocessing (12 cores)
    n_workers = 12
    print(f"Using {n_workers} workers")
    
    total_json_size = 0
    total_compressed_size = 0
    processed = 0
    skipped = 0
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_piece, pid): pid for pid in piece_ids}
        
        with tqdm(total=len(piece_ids), desc="Compressing mesh data") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.update(1)
                
                if result["status"] == "ok":
                    processed += 1
                    total_json_size += result["json_size"]
                    total_compressed_size += result["compressed_size"]
                    pbar.set_postfix({
                        "piece": result["piece_id"],
                        "frames": result["n_frames"],
                        "ratio": f"{result['ratio']:.1f}%"
                    })
                else:
                    skipped += 1
    
    print()
    print(f"=== Results ===")
    print(f"Processed: {processed} pieces")
    print(f"Skipped: {skipped} pieces")
    if total_json_size > 0:
        overall_ratio = (1 - total_compressed_size / total_json_size) * 100
        print(f"Total JSON size: {total_json_size / 1024 / 1024:.1f} MB")
        print(f"Total compressed size: {total_compressed_size / 1024 / 1024:.1f} MB")
        print(f"Overall compression ratio: {overall_ratio:.1f}%")


if __name__ == "__main__":
    main()

