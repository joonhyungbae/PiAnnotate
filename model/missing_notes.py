#!/usr/bin/env python3
"""
Missing Notes Handler

Find notes that exist in MIDI but are missing from rule-based fingering, and assign fingering to them.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

try:
    import mido
except ImportError:
    mido = None


# MANO joint indices for fingertips
FINGERTIP_INDICES = {
    'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20
}

FINGER_TO_NUMBER = {
    'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5
}

NUMBER_TO_FINGER = {v: k for k, v in FINGER_TO_NUMBER.items()}


@dataclass
class KeyBounds:
    """Key coordinate bounds"""
    key_idx: int
    y_min: float
    y_max: float
    x_min: float
    x_max: float
    z_surface: float
    is_black: bool


def is_black_key(key_idx: int) -> bool:
    """Check if key is black based on 88-key layout (A0 = 0)"""
    BLACK_KEY_OFFSETS = [1, 4, 6, 9, 11]
    return (key_idx % 12) in BLACK_KEY_OFFSETS


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
    """Load key bounds from piano meshes (with X-axis 90-degree rotation)"""
    key_bounds = {}
    
    for key_idx in range(88):
        obj_path = meshes_dir / f"{key_idx}.obj"
        if not obj_path.exists():
            continue
        
        vertices = parse_obj_file(obj_path)
        
        x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
        y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
        z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
        
        # X-axis 90-degree rotation: [x, y, z] -> [x, -z, y]
        rot_y_min, rot_y_max = -z_max, -z_min
        rot_z_max = y_max
        
        key_bounds[key_idx] = KeyBounds(
            key_idx=key_idx,
            y_min=rot_y_min,
            y_max=rot_y_max,
            x_min=x_min,
            x_max=x_max,
            z_surface=rot_z_max,
            is_black=is_black_key(key_idx)
        )
    
    return key_bounds


def load_midi_note_events(midi_path: Path, fps: float = 60.0) -> List[Dict]:
    """
    Extract all note events from MIDI

    Returns:
        List of {key_idx, onset_frame, offset_frame}
    """
    if mido is None:
        raise ImportError("mido library is required for MIDI processing")
    
    midi = mido.MidiFile(midi_path)
    MIDI_OFFSET = 21  # MIDI 21 = A0 = key index 0
    
    active_notes = {}  # key_idx -> onset_time
    note_events = []
    
    for track in midi.tracks:
        track_time = 0.0
        for msg in track:
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
                    
                    note_events.append({
                        'key_idx': key_idx,
                        'onset_frame': onset_frame,
                        'offset_frame': offset_frame
                    })
    
    # Handle unclosed notes
    for key_idx, onset_time in active_notes.items():
        onset_frame = int(onset_time * fps)
        note_events.append({
            'key_idx': key_idx,
            'onset_frame': onset_frame,
            'offset_frame': -1  # Will be set to last frame
        })
    
    return note_events


def get_fingering_notes(fingering: List[List[Dict]]) -> Set[Tuple[int, int]]:
    """
    Extract (onset_frame, key_idx) pairs from rule-based fingering.

    Finds the start of consecutive fingering blocks to determine note onset frames.
    """
    notes = set()
    
    # Find the start of fingering blocks for each key
    key_active = {}  # key_idx -> (start_frame, hand, finger)
    
    for frame_idx, frame_fingerings in enumerate(fingering):
        current_keys = set()
        
        for f in frame_fingerings:
            key_idx = f['key_index']
            current_keys.add(key_idx)
            
            if key_idx not in key_active:
                # New note starts
                key_active[key_idx] = frame_idx
                notes.add((frame_idx, key_idx))
        
        # Remove keys that were active previously but are absent in the current frame
        ended_keys = set(key_active.keys()) - current_keys
        for key_idx in ended_keys:
            del key_active[key_idx]
    
    return notes


def find_missing_notes(
    midi_path: Path,
    fingering: List[List[Dict]],
    fps: float = 60.0,
    onset_tolerance: int = 5
) -> List[Dict]:
    """
    Find notes that exist in MIDI but are missing from fingering.

    Args:
        midi_path: Path to MIDI file
        fingering: Rule-based fingering
        fps: Frame rate
        onset_tolerance: Onset frame tolerance

    Returns:
        List of missing notes [{key_idx, onset_frame, offset_frame}, ...]
    """
    # Extract all notes from MIDI
    midi_notes = load_midi_note_events(midi_path, fps)
    
    # Extract already existing notes from fingering
    existing_notes = get_fingering_notes(fingering)
    
    # Find missing notes
    missing_notes = []

    for midi_note in midi_notes:
        key_idx = midi_note['key_idx']
        onset_frame = midi_note['onset_frame']

        # Check if there is a matching fingering within onset_tolerance range
        found = False
        for tol in range(-onset_tolerance, onset_tolerance + 1):
            if (onset_frame + tol, key_idx) in existing_notes:
                found = True
                break
        
        if not found:
            missing_notes.append(midi_note)
    
    return missing_notes


def build_missing_note_samples(
    missing_notes: List[Dict],
    motion: Dict,
    config,
    max_seq_len: int = 128
) -> List[Dict]:
    """
    Generate inference samples for missing notes.

    Args:
        missing_notes: List of missing notes [{key_idx, onset_frame, offset_frame}, ...]
        motion: motion.pkl data
        config: Config object
        max_seq_len: Maximum sequence length

    Returns:
        List of inference samples (original_class=0)
    """
    # Import here to avoid circular import
    from .features import FeatureExtractor, NoteEvent
    from .features import NoteGroup
    
    if not missing_notes:
        return []
    
    feature_extractor = FeatureExtractor(config)
    
    # Check number of frames
    left_joints = motion.get('left', {}).get('joints', None)
    right_joints = motion.get('right', {}).get('joints', None)
    num_frames = max(
        left_joints.shape[0] if left_joints is not None else 0,
        right_joints.shape[0] if right_joints is not None else 0
    )
    
    # Convert to NoteEvent
    note_events = []
    for note in missing_notes:
        onset_frame = note['onset_frame']
        offset_frame = note['offset_frame']
        
        if onset_frame >= num_frames:
            continue
        if offset_frame == -1 or offset_frame >= num_frames:
            offset_frame = num_frames - 1
        
        note_event = NoteEvent(
            onset_frame=onset_frame,
            key_idx=note['key_idx'],
            hand=None,  # no ground truth
            finger=None,
            original_hand=None,  # no original fingering
            original_finger=None,
            offset_frame=offset_frame,
        )
        note_events.append(note_event)
    
    if not note_events:
        return []
    
    # Sort by onset order
    note_events.sort(key=lambda x: (x.onset_frame, x.key_idx))
    
    # Grouping
    groups = []
    current_notes = [note_events[0]]
    current_onset = note_events[0].onset_frame
    
    for note in note_events[1:]:
        if note.onset_frame - current_onset <= config.onset_tolerance:
            current_notes.append(note)
        else:
            groups.append(NoteGroup(onset_frame=current_onset, notes=current_notes))
            current_notes = [note]
            current_onset = note.onset_frame
    
    if current_notes:
        groups.append(NoteGroup(onset_frame=current_onset, notes=current_notes))
    
    # Generate samples
    samples = []
    max_notes = config.max_notes_per_group
    feature_dim = feature_extractor.get_note_feature_dim()
    
    all_features = []
    all_original_classes = []
    all_masks = []
    all_note_infos = []
    
    for group in groups:
        group_features = np.zeros((max_notes, feature_dim), dtype=np.float32)
        group_original = np.zeros(max_notes, dtype=np.int64)  # all 0 (missing)
        group_mask = np.zeros(max_notes, dtype=bool)
        group_note_info = []
        
        for i, note in enumerate(group.notes[:max_notes]):
            features = feature_extractor.extract_note_features(
                motion=motion,
                note=note,
                frame_idx=group.onset_frame
            )
            group_features[i] = features
            group_original[i] = 0  # missing notes have original_class=0
            group_mask[i] = True
            group_note_info.append({
                'onset_frame': note.onset_frame,
                'offset_frame': note.offset_frame,
                'key_idx': note.key_idx,
                'original_class': 0,
                'is_missing': True,  # marks as missing note
            })
        
        all_features.append(group_features)
        all_original_classes.append(group_original)
        all_masks.append(group_mask)
        all_note_infos.append(group_note_info)
    
    # Split into sequences
    num_groups = len(groups)
    for start_idx in range(0, num_groups, max_seq_len):
        end_idx = min(start_idx + max_seq_len, num_groups)
        
        seq_features = np.stack(all_features[start_idx:end_idx], axis=0)
        seq_original = np.stack(all_original_classes[start_idx:end_idx], axis=0)
        seq_masks = np.stack(all_masks[start_idx:end_idx], axis=0)
        
        samples.append({
            'features': seq_features,
            'original_classes': seq_original,
            'note_mask': seq_masks,
            'note_infos': all_note_infos[start_idx:end_idx],
            'is_missing_batch': True,  # marks as missing note batch
        })
    
    return samples


def process_missing_notes_with_model(
    piece_id: int,
    fingering: List[List[Dict]],
    motion: Dict,
    model,
    config,
    device,
    correction_threshold: float = 0.5,
    verbose: bool = False
) -> Tuple[List[Dict], int]:
    """
    Process missing notes for a single piece using the model.

    Returns:
        (list of model prediction results, number of missing notes)
    """
    import torch
    
    # MIDI path
    midi_path = config.dataset_path / f"{piece_id:03d}" / "midi.mid"
    if not midi_path.exists():
        if verbose:
            print(f"  Warning: MIDI not found for piece {piece_id}")
        return [], 0
    
    # Find missing notes
    missing_notes = find_missing_notes(
        midi_path=midi_path,
        fingering=fingering,
        fps=config.fps,
        onset_tolerance=5
    )
    
    if verbose and missing_notes:
        print(f"  Found {len(missing_notes)} missing notes")
    
    if not missing_notes:
        return [], 0
    
    # Generate inference samples
    samples = build_missing_note_samples(
        missing_notes=missing_notes,
        motion=motion,
        config=config,
        max_seq_len=128
    )
    
    if not samples:
        return [], len(missing_notes)
    
    # Model inference
    all_results = []
    
    model.eval()
    with torch.no_grad():
        for sample in samples:
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0).to(device)
            original_classes = torch.tensor(sample['original_classes'], dtype=torch.long).unsqueeze(0).to(device)
            note_mask = torch.tensor(sample['note_mask'], dtype=torch.bool).unsqueeze(0).to(device)
            seq_len = features.shape[1]
            seq_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
            
            # Run model (original_classes are all 0)
            final_preds, needs_correction, correction_probs = model.predict(
                features, note_mask, seq_mask, original_classes,
                correction_threshold=correction_threshold
            )
            
            # Convert to results
            final_preds = final_preds[0].cpu().numpy()
            correction_probs = correction_probs[0].cpu().numpy()
            note_mask_np = sample['note_mask']
            
            for group_idx, note_infos in enumerate(sample['note_infos']):
                for note_idx, note_info in enumerate(note_infos):
                    if not note_mask_np[group_idx, note_idx]:
                        continue
                    
                    pred_class = int(final_preds[group_idx, note_idx])
                    corr_prob = float(correction_probs[group_idx, note_idx])
                    
                    # Class to hand/finger
                    if pred_class > 0:
                        if pred_class <= 5:
                            hand, finger = 'left', pred_class
                        else:
                            hand, finger = 'right', pred_class - 5
                    else:
                        hand, finger = None, None
                    
                    if hand and finger:
                        result = {
                            'onset_frame': note_info['onset_frame'],
                            'offset_frame': note_info.get('offset_frame', note_info['onset_frame']),
                            'key_idx': note_info['key_idx'],
                            'predicted_hand': hand,
                            'predicted_finger': finger,
                            'predicted_class': pred_class,
                            'original_hand': None,
                            'original_finger': None,
                            'original_class': 0,
                            'was_corrected': True,  # missing notes are always marked as corrected
                            'correction_prob': corr_prob,
                            'is_missing': True,
                        }
                        all_results.append(result)
    
    if verbose:
        print(f"  Model predicted {len(all_results)}/{len(missing_notes)} missing notes")
    
    return all_results, len(missing_notes)


def merge_results_with_missing(
    model_results: List[Dict],
    missing_results: List[Dict]
) -> List[Dict]:
    """
    Merge model results with missing note results.
    """
    # Create (onset_frame, key_idx) keys for duplicate checking
    existing_keys = {(r['onset_frame'], r['key_idx']) for r in model_results}
    
    # Add only non-duplicate missing results
    merged = list(model_results)
    for r in missing_results:
        key = (r['onset_frame'], r['key_idx'])
        if key not in existing_keys:
            merged.append(r)
    
    # Sort by onset_frame
    merged.sort(key=lambda x: (x['onset_frame'], x['key_idx']))
    
    return merged
