#!/usr/bin/env python3
"""
Create pressed_keys.pkl from MIDI

Parse note_on/note_off events from MIDI accurately to
generate pressed key states for each frame.
"""

import pickle
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import mido
from typing import Dict, List, Tuple


def load_midi_note_events(midi_path: Path, fps: float = 60.0) -> List[Tuple[int, int, int]]:
    """
    Extract note events from MIDI file
    
    Returns:
        List of (key_idx, onset_frame, offset_frame)
    """
    midi = mido.MidiFile(midi_path)
    
    # MIDI note number -> key index conversion (MIDI 21 = A0 = key index 0)
    MIDI_OFFSET = 21
    
    # Track active notes: key_idx -> onset_time
    active_notes: Dict[int, float] = {}
    note_events: List[Tuple[int, int, int]] = []
    
    for track in midi.tracks:
        track_time = 0.0
        
        for msg in track:
            # Convert delta time to seconds
            track_time += mido.tick2second(msg.time, midi.ticks_per_beat, 500000)
            
            if msg.type == 'note_on' and msg.velocity > 0:
                key_idx = msg.note - MIDI_OFFSET
                if 0 <= key_idx < 88:
                    active_notes[key_idx] = track_time
                    
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                key_idx = msg.note - MIDI_OFFSET
                if 0 <= key_idx < 88 and key_idx in active_notes:
                    onset_time = active_notes.pop(key_idx)
                    onset_frame = int(onset_time * fps)
                    offset_frame = int(track_time * fps)
                    note_events.append((key_idx, onset_frame, offset_frame))
    
    return note_events


def create_pressed_keys(
    midi_path: Path,
    num_frames: int,
    fps: float = 60.0
) -> np.ndarray:
    """
    Create pressed_keys array from MIDI
    
    Args:
        midi_path: Path to MIDI file
        num_frames: Total number of frames
        fps: Frames per second
    
    Returns:
        (num_frames, 88) numpy array
    """
    # Initialize result array
    pressed_keys = np.zeros((num_frames, 88), dtype=np.float32)
    
    # Load MIDI events
    note_events = load_midi_note_events(midi_path, fps)
    
    # Apply each note event to pressed_keys
    for key_idx, onset_frame, offset_frame in note_events:
        # Clamp frame range
        onset_frame = max(0, onset_frame)
        offset_frame = min(offset_frame, num_frames - 1)
        
        # Mark key as pressed in the range
        pressed_keys[onset_frame:offset_frame + 1, key_idx] = 1.0
    
    return pressed_keys


def process_piece(piece_id: int, dataset_path: Path, fps: float = 60.0) -> bool:
    """
    Process a single piece
    
    Returns:
        Success status
    """
    piece_dir = dataset_path / f"{piece_id:03d}"
    
    # Check MIDI file
    midi_path = piece_dir / "midi.mid"
    if not midi_path.exists():
        print(f"  Warning: midi.mid not found for piece {piece_id}")
        return False
    
    # Get frame count from motion.pkl
    motion_path = piece_dir / "motion.pkl"
    if not motion_path.exists():
        print(f"  Warning: motion.pkl not found for piece {piece_id}")
        return False
    
    with open(motion_path, 'rb') as f:
        motion = pickle.load(f)
    
    # Determine frame count
    num_frames = 0
    if 'left' in motion and 'joints' in motion['left']:
        num_frames = motion['left']['joints'].shape[0]
    elif 'right' in motion and 'joints' in motion['right']:
        num_frames = motion['right']['joints'].shape[0]
    
    if num_frames == 0:
        print(f"  Warning: Cannot determine frame count for piece {piece_id}")
        return False
    
    # Create pressed_keys
    pressed_keys = create_pressed_keys(midi_path, num_frames, fps)
    
    # Save
    output_path = piece_dir / "vis" / "pressed_keys.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(pressed_keys, f)
    
    # Statistics
    keys_pressed = (pressed_keys > 0.5).sum()
    frames_with_keys = (pressed_keys.sum(axis=1) > 0).sum()
    
    return True


def process_dataset(dataset_path: str, fps: float = 60.0):
    """Process entire dataset"""
    dataset_path = Path(dataset_path)
    
    # Find all sample directories
    sample_dirs = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(dataset_path / d) and d.isdigit()
    ])
    
    print(f"Processing {len(sample_dirs)} samples...")
    
    success_count = 0
    for sample_dir in tqdm(sample_dirs, desc="Creating pressed_keys"):
        piece_id = int(sample_dir)
        if process_piece(piece_id, dataset_path, fps):
            success_count += 1
    
    print(f"\nSuccessfully processed {success_count}/{len(sample_dirs)} pieces")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create pressed_keys.pkl from MIDI")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./for_elise/dataset",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=60000/1001,  # NTSC standard: 59.94 fps
        help="Frames per second (default: 60000/1001 = 59.94)"
    )
    
    args = parser.parse_args()
    
    process_dataset(args.dataset_path, args.fps)
