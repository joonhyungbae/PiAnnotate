#!/usr/bin/env python3
"""
Fingering extraction script v2

Improved approach:
- Determine which key based on Y coordinate (pitch direction)
- Distinguish black/white keys and detect contact via Z coordinate
- Utilize MIDI note onset timing
"""

import pickle
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import mido
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# MANO joint indices for fingertips
FINGER_TIP_INDICES = {
    'thumb': 4,
    'index': 8,
    'middle': 12,
    'ring': 16,
    'pinky': 20
}

# Fingering number mapping (1-5)
FINGER_TO_NUMBER = {
    'thumb': 1,
    'index': 2,
    'middle': 3,
    'ring': 4,
    'pinky': 5
}

# Black key pattern (position within octave, A=0)
BLACK_KEY_OFFSETS = [1, 4, 6, 9, 11]


def is_black_key(key_idx: int) -> bool:
    """Check if key is black based on 88-key layout (A0 = 0)"""
    return (key_idx % 12) in BLACK_KEY_OFFSETS


@dataclass
class KeyBounds:
    """Key coordinate bounds (after rotation)"""
    key_idx: int
    y_min: float  # Left-right min
    y_max: float  # Left-right max
    x_min: float  # Front-back min
    x_max: float  # Front-back max
    z_surface: float  # Key surface height
    is_black: bool


def parse_obj_file(obj_path: Path) -> np.ndarray:
    """Extract vertices from OBJ file"""
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.array(vertices, dtype=np.float32)


def load_key_bounds(meshes_dir: Path) -> Dict[int, KeyBounds]:
    """
    Calculate coordinate bounds for all keys (with X-axis 90-degree rotation)
    Original coordinates: [x, y, z]
    After rotation: [x, -z, y] -> X=front-back, Y=left-right (pitch), Z=height
    """
    key_bounds = {}
    
    for key_idx in range(88):
        obj_path = meshes_dir / f"{key_idx}.obj"
        if not obj_path.exists():
            continue
        
        vertices = parse_obj_file(obj_path)
        
        # Original coordinate bounds
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        
        # X-axis 90-degree rotation: [x, y, z] -> [x, -z, y]
        rot_x_min, rot_x_max = x_min, x_max          # X: front-back (unchanged)
        rot_y_min, rot_y_max = -z_max, -z_min        # Y: left-right (pitch)
        rot_z_min, rot_z_max = y_min, y_max          # Z: height
        
        is_black = is_black_key(key_idx)
        
        key_bounds[key_idx] = KeyBounds(
            key_idx=key_idx,
            y_min=rot_y_min,
            y_max=rot_y_max,
            x_min=rot_x_min,
            x_max=rot_x_max,
            z_surface=rot_z_max,  # Key surface (highest Z)
            is_black=is_black
        )
    
    return key_bounds


def load_midi_note_events(midi_path: Path, fps: float = 60.0) -> Dict[int, List[Tuple[int, int]]]:
    """
    Convert MIDI note onset-offset intervals to frame indices
    
    Returns:
        Dict[key_idx, List[(onset_frame, offset_frame)]]
    """
    midi = mido.MidiFile(midi_path)
    
    # MIDI note number -> key index conversion (MIDI 21 = A0 = key index 0)
    MIDI_OFFSET = 21
    
    # Track active notes: key_idx -> onset_time
    active_notes = {}
    # Result: key_idx -> [(onset_frame, offset_frame), ...]
    note_events = {}
    
    current_time = 0.0  # In seconds
    
    for track in midi.tracks:
        track_time = 0.0
        for msg in track:
            track_time += mido.tick2second(msg.time, midi.ticks_per_beat, 500000)
            
            if msg.type == 'note_on' and msg.velocity > 0:
                key_idx = msg.note - MIDI_OFFSET
                if 0 <= key_idx < 88:
                    # Note start
                    active_notes[key_idx] = track_time
                    
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key_idx = msg.note - MIDI_OFFSET
                if 0 <= key_idx < 88 and key_idx in active_notes:
                    # Note end
                    onset_time = active_notes.pop(key_idx)
                    onset_frame = int(onset_time * fps)
                    offset_frame = int(track_time * fps)
                    
                    if key_idx not in note_events:
                        note_events[key_idx] = []
                    note_events[key_idx].append((onset_frame, offset_frame))
    
    # Handle unclosed notes (until last frame)
    for key_idx, onset_time in active_notes.items():
        onset_frame = int(onset_time * fps)
        # Last frame will be set later
        if key_idx not in note_events:
            note_events[key_idx] = []
        note_events[key_idx].append((onset_frame, -1))  # -1 means until end
    
    return note_events


def find_key_at_fingertip(
    tip_y: float,
    tip_z: float,
    key_bounds: Dict[int, KeyBounds],
    z_threshold: float = 0.02
) -> Optional[int]:
    """
    Find which key the fingertip is over based on Y, Z coordinates
    
    Args:
        tip_y: Fingertip Y coordinate (left-right, pitch direction)
        tip_z: Fingertip Z coordinate (height)
        key_bounds: Key coordinate bounds
        z_threshold: Maximum distance from key surface
    
    Returns:
        key_idx or None
    """
    candidates = []
    
    for key_idx, bounds in key_bounds.items():
        # Check Y range (which key is the fingertip over)
        if not (bounds.y_min <= tip_y <= bounds.y_max):
            continue
        
        # Calculate Z distance (distance from key surface)
        z_distance = tip_z - bounds.z_surface
        
        # On key surface or slightly below (pressing)
        if z_distance <= z_threshold and z_distance >= -0.05:
            candidates.append((key_idx, bounds.z_surface, bounds.is_black))
    
    if not candidates:
        return None
    
    # Prioritize black keys (higher Z)
    black_candidates = [c for c in candidates if c[2]]
    if black_candidates:
        return black_candidates[0][0]
    
    # White key
    return candidates[0][0]


def extract_fingering_v2(
    piece_id: int,
    dataset_path: Path,
    meshes_dir: Path,
    key_bounds: Dict[int, KeyBounds],
    fps: float = 60.0,
    z_threshold: float = 0.02,
    use_midi: bool = True
) -> List[dict]:
    """
    Improved fingering extraction
    
    Approach:
    1. Get note onset timing from MIDI
    2. Find fingertip within the key's Y range at onset time
    3. Distinguish black/white keys and check contact via Z coordinate
    """
    piece_dir = dataset_path / f"{piece_id:03d}"
    
    # Load motion.pkl
    motion_path = piece_dir / "motion.pkl"
    if not motion_path.exists():
        print(f"  Warning: motion.pkl not found for piece {piece_id}")
        return []
    
    with open(motion_path, 'rb') as f:
        motion = pickle.load(f)
    
    # Load pressed_keys.pkl (for fallback)
    pressed_keys_path = piece_dir / "vis" / "pressed_keys.pkl"
    pressed_keys = None
    if pressed_keys_path.exists():
        with open(pressed_keys_path, 'rb') as f:
            pressed_keys = pickle.load(f)
        if not isinstance(pressed_keys, np.ndarray):
            pressed_keys = np.array(pressed_keys)
    
    # Load MIDI note events (onset-offset intervals)
    midi_note_events = {}
    midi_path = piece_dir / "midi.mid"
    if use_midi and midi_path.exists():
        try:
            midi_note_events = load_midi_note_events(midi_path, fps)
            total_notes = sum(len(v) for v in midi_note_events.values())
            print(f"  Loaded MIDI: {total_notes} notes")
        except Exception as e:
            print(f"  Warning: Failed to load MIDI: {e}")
    
    # Check frame count
    left_joints = motion.get('left', {}).get('joints', None)
    right_joints = motion.get('right', {}).get('joints', None)
    
    if left_joints is None and right_joints is None:
        return []
    
    num_frames = 0
    if left_joints is not None:
        num_frames = left_joints.shape[0]
    if right_joints is not None:
        num_frames = max(num_frames, right_joints.shape[0])
    
    # Store results by frame
    fingering_by_frame = {i: [] for i in range(num_frames)}
    
    # Method 1: MIDI note event based (onset-offset intervals)
    if midi_note_events:
        for key_idx, note_ranges in midi_note_events.items():
            for onset_frame, offset_frame in note_ranges:
                if onset_frame >= num_frames:
                    continue
                
                # If offset is -1, use last frame
                if offset_frame == -1:
                    offset_frame = num_frames - 1
                
                # Clamp frame range
                offset_frame = min(offset_frame, num_frames - 1)
                
                # Find best finger at onset frame
                best_finger = None
                best_hand = None
                best_score = float('inf')
                best_z_distance = float('inf')
                
                bounds = key_bounds.get(key_idx)
                if bounds is None:
                    continue
                
                # Calculate key center
                key_x_center = (bounds.x_min + bounds.x_max) / 2
                key_y_center = (bounds.y_min + bounds.y_max) / 2
                key_x_length = bounds.x_max - bounds.x_min
                
                for hand in ['left', 'right']:
                    joints = motion.get(hand, {}).get('joints', None)
                    if joints is None or onset_frame >= joints.shape[0]:
                        continue
                    
                    frame_joints = joints[onset_frame]
                    
                    for finger_name, joint_idx in FINGER_TIP_INDICES.items():
                        tip_pos = frame_joints[joint_idx]
                        tip_x, tip_y, tip_z = tip_pos[0], tip_pos[1], tip_pos[2]
                        
                        # Check Y range (left-right, pitch) - allow some margin
                        y_margin = 0.005  # 5mm margin
                        if not (bounds.y_min - y_margin <= tip_y <= bounds.y_max + y_margin):
                            continue
                        
                        # Check X range (front-back) - allow some margin
                        x_margin = 0.01  # 1cm margin
                        if not (bounds.x_min - x_margin <= tip_x <= bounds.x_max + x_margin):
                            continue
                        
                        z_distance = abs(tip_z - bounds.z_surface)
                        if z_distance >= z_threshold:
                            continue
                        
                        # X distance from key center (normalized)
                        x_from_center = abs(tip_x - key_x_center) / key_x_length
                        
                        # Combined score: Z distance + X center distance weight
                        # Penalty if X is at front edge of key
                        score = z_distance + x_from_center * 0.01
                        
                        if score < best_score:
                            best_score = score
                            best_z_distance = z_distance
                            best_finger = finger_name
                            best_hand = hand
                
                # Add fingering to all frames from onset to offset
                if best_finger is not None:
                    fingering_entry = {
                        'key_index': key_idx,
                        'hand': best_hand,
                        'finger': FINGER_TO_NUMBER[best_finger],
                        'finger_name': best_finger,
                        'distance': float(best_z_distance)
                    }
                    for frame_idx in range(onset_frame, offset_frame + 1):
                        # Prevent duplicates: check if fingering for this key already exists
                        existing = [f for f in fingering_by_frame[frame_idx] if f['key_index'] == key_idx]
                        if not existing:
                            fingering_by_frame[frame_idx].append(fingering_entry.copy())
    
    # Method 2: pressed_keys based (fallback)
    elif pressed_keys is not None:
        for frame_idx in range(min(num_frames, len(pressed_keys))):
            pressed = pressed_keys[frame_idx]
            pressed_key_indices = np.where(pressed > 0.5)[0]
            
            for key_idx in pressed_key_indices:
                bounds = key_bounds.get(int(key_idx))
                if bounds is None:
                    continue
                
                best_finger = None
                best_hand = None
                best_score = float('inf')
                best_z_distance = float('inf')
                
                # Calculate key center
                key_x_center = (bounds.x_min + bounds.x_max) / 2
                key_x_length = bounds.x_max - bounds.x_min
                
                for hand in ['left', 'right']:
                    joints = motion.get(hand, {}).get('joints', None)
                    if joints is None or frame_idx >= joints.shape[0]:
                        continue
                    
                    frame_joints = joints[frame_idx]
                    
                    for finger_name, joint_idx in FINGER_TIP_INDICES.items():
                        tip_pos = frame_joints[joint_idx]
                        tip_x, tip_y, tip_z = tip_pos[0], tip_pos[1], tip_pos[2]
                        
                        # Check Y range (left-right, pitch) - allow some margin
                        y_margin = 0.005  # 5mm margin
                        if not (bounds.y_min - y_margin <= tip_y <= bounds.y_max + y_margin):
                            continue
                        
                        # Check X range (front-back) - allow some margin
                        x_margin = 0.01  # 1cm margin
                        if not (bounds.x_min - x_margin <= tip_x <= bounds.x_max + x_margin):
                            continue
                        
                        z_distance = abs(tip_z - bounds.z_surface)
                        if z_distance >= z_threshold:
                            continue
                        
                        # X distance from key center (normalized)
                        x_from_center = abs(tip_x - key_x_center) / key_x_length
                        
                        # Combined score: Z distance + X center distance weight
                        score = z_distance + x_from_center * 0.01
                        
                        if score < best_score:
                            best_score = score
                            best_z_distance = z_distance
                            best_finger = finger_name
                            best_hand = hand
                
                if best_finger is not None:
                    fingering_by_frame[frame_idx].append({
                        'key_index': int(key_idx),
                        'hand': best_hand,
                        'finger': FINGER_TO_NUMBER[best_finger],
                        'finger_name': best_finger,
                        'distance': float(best_z_distance)
                    })
    
    # Convert to list (per frame)
    fingering_list = [fingering_by_frame.get(i, []) for i in range(num_frames)]
    
    return fingering_list


def process_dataset(
    dataset_path: str,
    meshes_dir: str,
    output_dir: str = "./data/fingering",
    fps: float = 60.0,
    z_threshold: float = 0.02
):
    """Process entire dataset"""
    dataset_path = Path(dataset_path)
    meshes_dir = Path(meshes_dir)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading key bounds...")
    key_bounds = load_key_bounds(meshes_dir)
    print(f"Loaded {len(key_bounds)} key bounds")
    
    # Find all sample directories
    sample_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(dataset_path / d) and d.isdigit()
    ])
    
    print(f"\nProcessing {len(sample_dirs)} samples...")
    
    for sample_dir in tqdm(sample_dirs, desc="Extracting fingering"):
        piece_id = int(sample_dir)
        
        fingering_list = extract_fingering_v2(
            piece_id=piece_id,
            dataset_path=dataset_path,
            meshes_dir=meshes_dir,
            key_bounds=key_bounds,
            fps=fps,
            z_threshold=z_threshold,
            use_midi=True
        )
        
        if fingering_list:
            # Save to ./data/fingering/{id:03d}.pkl
            output_path = output_dir / f"{piece_id:03d}.pkl"
            
            with open(output_path, 'wb') as f:
                pickle.dump(fingering_list, f)
            
            # Statistics
            frames_with_fingering = sum(1 for f in fingering_list if f)
            total_fingerings = sum(len(f) for f in fingering_list)
            # print(f"  Piece {piece_id}: {total_fingerings} fingerings in {frames_with_fingering} frames")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract fingering data v2")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./for_elise/dataset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--meshes_dir",
        type=str,
        default="./for_elise/piano_meshes",
        help="Path to piano meshes directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/fingering",
        help="Output directory for fingering files (e.g., ./data/fingering/{id}.pkl)"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60000/1001,  # NTSC standard: 59.94 fps
        help="Frames per second (default: 60000/1001 = 59.94)"
    )
    parser.add_argument(
        "--z_threshold",
        type=float,
        default=0.02,
        help="Z distance threshold for fingertip-key contact"
    )
    
    args = parser.parse_args()
    
    process_dataset(
        dataset_path=args.dataset_path,
        meshes_dir=args.meshes_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        z_threshold=args.z_threshold
    )
