"""
Dataset for Fingering Correction Model (MIDI-based)

Key changes:
- Input is organized based on MIDI notes
- Missing notes can also be handled during inference
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False
    print("Warning: mido not installed. MIDI-based features will be disabled.")

from .config import Config
from .features import FeatureExtractor, NoteEvent, NoteGroup


@dataclass
class SequenceSample:
    """A sequence sample for training"""
    piece_id: int
    note_groups: List[NoteGroup]  # Sequence of note groups
    features: np.ndarray  # (seq_len, num_notes, feature_dim)
    targets: np.ndarray  # (seq_len, num_notes) - class indices
    masks: np.ndarray  # (seq_len, num_notes) - valid positions


class FingeringSequenceDataset(Dataset):
    """Dataset for sequence-to-sequence fingering correction"""
    
    def __init__(
        self,
        samples: List[Dict],
        config: Config,
        max_seq_len: int = 128
    ):
        self.samples = samples
        self.config = config
        self.max_seq_len = max_seq_len
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Features: (seq_len, max_notes, feature_dim)
        features = sample['features']
        targets = sample['targets']
        note_mask = sample['note_mask']
        original_classes = sample['original_classes']  # Original prediction
        
        seq_len = features.shape[0]
        
        return {
            'features': torch.tensor(features, dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.long),
            'original_classes': torch.tensor(original_classes, dtype=torch.long),
            'note_mask': torch.tensor(note_mask, dtype=torch.bool),
            'seq_len': seq_len,
            'piece_id': sample['piece_id'],
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable length sequences"""
    
    # Get max sequence length in batch
    max_seq_len = max(item['seq_len'] for item in batch)
    max_notes = batch[0]['features'].shape[1]
    feature_dim = batch[0]['features'].shape[2]
    
    batch_size = len(batch)
    
    # Prepare padded tensors
    features = torch.zeros(batch_size, max_seq_len, max_notes, feature_dim)
    targets = torch.zeros(batch_size, max_seq_len, max_notes, dtype=torch.long)
    original_classes = torch.zeros(batch_size, max_seq_len, max_notes, dtype=torch.long)
    note_mask = torch.zeros(batch_size, max_seq_len, max_notes, dtype=torch.bool)
    seq_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    
    piece_ids = []
    
    for i, item in enumerate(batch):
        seq_len = item['seq_len']
        features[i, :seq_len] = item['features']
        targets[i, :seq_len] = item['targets']
        original_classes[i, :seq_len] = item['original_classes']
        note_mask[i, :seq_len] = item['note_mask']
        seq_mask[i, :seq_len] = True
        piece_ids.append(item['piece_id'])
    
    return {
        'features': features,
        'targets': targets,
        'original_classes': original_classes,
        'note_mask': note_mask,
        'seq_mask': seq_mask,
        'piece_ids': piece_ids,
    }


class DatasetBuilder:
    """Build dataset from fingering data (Note-level)"""
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
    
    def build_samples(
        self,
        piece_ids: Optional[List[int]] = None,
        max_seq_len: int = 128,
        verbose: bool = True
    ) -> List[Dict]:
        """
        Build note-level sequence samples
        
        Returns:
            List of sample dictionaries with:
            - features: (seq_len, max_notes, feature_dim)
            - targets: (seq_len, max_notes)
            - note_mask: (seq_len, max_notes)
        """
        if piece_ids is None:
            piece_ids = self._find_common_piece_ids()
        
        if verbose:
            print(f"Building note-level samples from {len(piece_ids)} pieces...")
        
        all_samples = []
        
        iterator = tqdm(piece_ids, desc="Processing pieces") if verbose else piece_ids
        
        for piece_id in iterator:
            piece_samples = self._process_piece(piece_id, max_seq_len)
            all_samples.extend(piece_samples)
        
        if verbose:
            print(f"Total samples: {len(all_samples)}")
            self._print_statistics(all_samples)
        
        return all_samples
    
    def _find_common_piece_ids(self) -> List[int]:
        """Find piece IDs with both original and edited fingering, and R1 review completed"""
        import json
        
        common_ids = []
        status_dir = self.config.project_root / "data" / "metadata" / "annotation" / "status"
        
        for edited_path in self.config.fingering_edited_path.glob("*.pkl"):
            piece_id = int(edited_path.stem)
            original_path = self.config.fingering_path / f"{piece_id:03d}.pkl"
            motion_path = self.config.dataset_path / f"{piece_id:03d}" / "motion.pkl"
            status_path = status_dir / f"{piece_id:03d}.json"
            
            if not (original_path.exists() and motion_path.exists()):
                continue
            
            # Check R1 review status
            if not status_path.exists():
                continue
            
            try:
                with open(status_path, 'r') as f:
                    status = json.load(f)
                
                # Only include pieces with completed review1
                if status.get('review1') and status['review1'].get('completed_at'):
                    common_ids.append(piece_id)
            except Exception:
                continue
        
        return sorted(common_ids)
    
    def _process_piece(self, piece_id: int, max_seq_len: int) -> List[Dict]:
        """Process a single piece into sequence samples"""
        samples = []
        
        # Load data
        try:
            original_path = self.config.fingering_path / f"{piece_id:03d}.pkl"
            edited_path = self.config.fingering_edited_path / f"{piece_id:03d}.pkl"
            motion_path = self.config.dataset_path / f"{piece_id:03d}" / "motion.pkl"
            
            with open(original_path, 'rb') as f:
                original = pickle.load(f)
            with open(edited_path, 'rb') as f:
                edited = pickle.load(f)
            with open(motion_path, 'rb') as f:
                motion = pickle.load(f)
        except Exception as e:
            print(f"Error loading piece {piece_id}: {e}")
            return []
        
        # Extract note groups from edited data (ground truth)
        note_groups = self._extract_note_groups(edited, original)
        
        if len(note_groups) == 0:
            return []
        
        # Extract features for all note groups
        all_features = []
        all_targets = []
        all_original_classes = []  # Original predictions
        all_masks = []
        
        max_notes = self.config.max_notes_per_group
        feature_dim = self.feature_extractor.get_note_feature_dim()
        
        for group in note_groups:
            # Extract features for this group
            group_features = np.zeros((max_notes, feature_dim), dtype=np.float32)
            group_targets = np.zeros(max_notes, dtype=np.int64)
            group_original = np.zeros(max_notes, dtype=np.int64)  # Original prediction
            group_mask = np.zeros(max_notes, dtype=bool)
            
            for i, note in enumerate(group.notes[:max_notes]):
                # Features for this note
                features = self.feature_extractor.extract_note_features(
                    motion=motion,
                    note=note,
                    frame_idx=group.onset_frame
                )
                group_features[i] = features
                
                # Target class (ground truth)
                if note.hand and note.finger:
                    group_targets[i] = Config.hand_finger_to_class(note.hand, note.finger)
                else:
                    group_targets[i] = 0  # Invalid
                
                # Original class (original prediction)
                if note.original_hand and note.original_finger:
                    group_original[i] = Config.hand_finger_to_class(note.original_hand, note.original_finger)
                else:
                    group_original[i] = 0  # No prediction
                
                group_mask[i] = True
            
            all_features.append(group_features)
            all_targets.append(group_targets)
            all_original_classes.append(group_original)
            all_masks.append(group_mask)
        
        # Split into sequences of max_seq_len
        num_groups = len(note_groups)
        
        for start_idx in range(0, num_groups, max_seq_len):
            end_idx = min(start_idx + max_seq_len, num_groups)
            seq_len = end_idx - start_idx
            
            # Stack features for this sequence
            seq_features = np.stack(all_features[start_idx:end_idx], axis=0)
            seq_targets = np.stack(all_targets[start_idx:end_idx], axis=0)
            seq_original = np.stack(all_original_classes[start_idx:end_idx], axis=0)
            seq_masks = np.stack(all_masks[start_idx:end_idx], axis=0)
            
            samples.append({
                'piece_id': piece_id,
                'start_group_idx': start_idx,
                'features': seq_features,
                'targets': seq_targets,
                'original_classes': seq_original,
                'note_mask': seq_masks,
                'note_groups': note_groups[start_idx:end_idx],
            })
        
        return samples
    
    def _extract_note_groups(
        self,
        edited: List,
        original: List
    ) -> List[NoteGroup]:
        """
        Extract note groups from fingering data
        
        A note group contains notes that start within onset_tolerance frames
        """
        # First, extract individual note events
        note_events = []
        prev_keys = {}  # key_idx -> (start_frame, entry)
        
        for frame_idx, frame in enumerate(edited):
            curr_keys = {}
            for entry in frame:
                if isinstance(entry, dict) and 'key_index' in entry:
                    curr_keys[entry['key_index']] = entry
            
            # Notes that just started
            for key_idx in curr_keys.keys() - prev_keys.keys():
                entry = curr_keys[key_idx]
                
                # Get original prediction for this note
                orig_entry = None
                if frame_idx < len(original):
                    for orig in original[frame_idx]:
                        if isinstance(orig, dict) and orig.get('key_index') == key_idx:
                            orig_entry = orig
                            break
                
                note = NoteEvent(
                    onset_frame=frame_idx,
                    key_idx=key_idx,
                    hand=entry.get('hand'),
                    finger=entry.get('finger'),
                    original_hand=orig_entry.get('hand') if orig_entry else None,
                    original_finger=orig_entry.get('finger') if orig_entry else None,
                )
                note_events.append(note)
            
            prev_keys = curr_keys
        
        # Sort by onset time
        note_events.sort(key=lambda x: (x.onset_frame, x.key_idx))
        
        # Group simultaneous notes
        groups = []
        if not note_events:
            return groups
        
        current_group_notes = [note_events[0]]
        current_onset = note_events[0].onset_frame
        
        for note in note_events[1:]:
            if note.onset_frame - current_onset <= self.config.onset_tolerance:
                current_group_notes.append(note)
            else:
                # Finalize current group
                groups.append(NoteGroup(
                    onset_frame=current_onset,
                    notes=current_group_notes
                ))
                current_group_notes = [note]
                current_onset = note.onset_frame
        
        # Add last group
        if current_group_notes:
            groups.append(NoteGroup(
                onset_frame=current_onset,
                notes=current_group_notes
            ))
        
        return groups
    
    def _print_statistics(self, samples: List[Dict]):
        """Print dataset statistics"""
        total_groups = sum(s['features'].shape[0] for s in samples)
        total_notes = sum(s['note_mask'].sum() for s in samples)
        
        # Class distribution
        class_counts = Counter()
        needs_correction_count = 0
        correct_original_count = 0
        
        for s in samples:
            targets = s['targets']
            original = s['original_classes']
            mask = s['note_mask']
            
            for t in targets[mask]:
                class_counts[int(t)] += 1
            
            # Correction statistics
            needs_correction = (targets != original) & mask
            needs_correction_count += needs_correction.sum()
            correct_original_count += ((targets == original) & mask).sum()
        
        print(f"\n=== Dataset Statistics ===")
        print(f"Sequences: {len(samples)}")
        print(f"Total note groups: {total_groups}")
        print(f"Total notes: {total_notes}")
        print(f"Avg notes per group: {total_notes/total_groups:.2f}")
        
        print(f"\n=== Correction Statistics ===")
        print(f"Original correct: {correct_original_count} ({100*correct_original_count/total_notes:.1f}%)")
        print(f"Needs correction: {needs_correction_count} ({100*needs_correction_count/total_notes:.1f}%)")
        
        print(f"\nClass distribution:")
        for cls in range(11):
            count = class_counts.get(cls, 0)
            if count > 0:
                if cls == 0:
                    print(f"  Invalid/Pad: {count}")
                else:
                    hand, finger = Config.class_to_hand_finger(cls)
                    finger_name = self.config.number_to_finger.get(finger, str(finger))
                    print(f"  {hand:5s} {finger_name:8s}: {count:7d} ({100*count/total_notes:.1f}%)")


def create_dataloaders(
    config: Config,
    samples: Optional[List[Dict]] = None,
    max_seq_len: int = 128,
    verbose: bool = True
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """Create train/val/test dataloaders (val can be None if val_ratio=0)"""
    
    if samples is None:
        builder = DatasetBuilder(config)
        samples = builder.build_samples(max_seq_len=max_seq_len, verbose=verbose)
    
    # Group by piece for splitting
    piece_samples = defaultdict(list)
    for sample in samples:
        piece_samples[sample['piece_id']].append(sample)
    
    # Split pieces
    piece_ids = list(piece_samples.keys())
    np.random.seed(config.random_seed)
    np.random.shuffle(piece_ids)
    
    n_total = len(piece_ids)
    n_train = int(n_total * config.train_ratio)
    n_val = int(n_total * config.val_ratio)
    
    train_pieces = piece_ids[:n_train]
    val_pieces = piece_ids[n_train:n_train + n_val] if n_val > 0 else []
    test_pieces = piece_ids[n_train + n_val:]
    
    # Collect samples
    train_samples = [s for pid in train_pieces for s in piece_samples[pid]]
    val_samples = [s for pid in val_pieces for s in piece_samples[pid]] if val_pieces else []
    test_samples = [s for pid in test_pieces for s in piece_samples[pid]]
    
    if verbose:
        print(f"\n=== Data Split ===")
        print(f"Train: {len(train_pieces)} pieces, {len(train_samples)} sequences")
        if val_samples:
            print(f"Val:   {len(val_pieces)} pieces, {len(val_samples)} sequences")
        else:
            print(f"Val:   None (val_ratio=0)")
        print(f"Test:  {len(test_pieces)} pieces, {len(test_samples)} sequences")
    
    # Create datasets
    train_dataset = FingeringSequenceDataset(train_samples, config, max_seq_len)
    test_dataset = FingeringSequenceDataset(test_samples, config, max_seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    # Val loader (None if no val data)
    val_loader = None
    if val_samples:
        val_dataset = FingeringSequenceDataset(val_samples, config, max_seq_len)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def save_samples(samples: List[Dict], path: Path):
    """Save samples to pickle file"""
    clean_samples = []
    for s in samples:
        clean = {
            'piece_id': s['piece_id'],
            'start_group_idx': s.get('start_group_idx', 0),
            'features': s['features'],
            'targets': s['targets'],
            'original_classes': s['original_classes'],
            'note_mask': s['note_mask'],
        }
        clean_samples.append(clean)
    
    with open(path, 'wb') as f:
        pickle.dump(clean_samples, f)
    print(f"Saved {len(clean_samples)} samples to {path}")


def load_samples(path: Path) -> List[Dict]:
    """Load samples from pickle file"""
    with open(path, 'rb') as f:
        samples = pickle.load(f)
    print(f"Loaded {len(samples)} samples from {path}")
    return samples


def load_midi_note_events(midi_path: Path, fps: float = 60.0) -> Dict[int, List[Tuple[int, int]]]:
    """
    Extract note events from a MIDI file

    Returns:
        Dict[key_idx, List[(onset_frame, offset_frame)]]
    """
    if not HAS_MIDO:
        raise ImportError("mido is required for MIDI processing")
    
    midi = mido.MidiFile(midi_path)
    
    # MIDI note number -> key index conversion (MIDI 21 = A0 = key index 0)
    MIDI_OFFSET = 21
    
    # Track active notes: key_idx -> onset_time
    active_notes = {}
    # Result: key_idx -> [(onset_frame, offset_frame), ...]
    note_events = defaultdict(list)
    
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
                if key_idx in active_notes:
                    onset_time = active_notes.pop(key_idx)
                    onset_frame = int(onset_time * fps)
                    offset_frame = int(track_time * fps)
                    note_events[key_idx].append((onset_frame, offset_frame))
    
    # Handle notes without note_off
    for key_idx, onset_time in active_notes.items():
        onset_frame = int(onset_time * fps)
        note_events[key_idx].append((onset_frame, -1))
    
    return dict(note_events)


class InferenceDatasetBuilder:
    """
    Dataset builder for inference

    Constructs input based on rule-based fingering data:
    - Extracts all notes from fingering
    - No frame synchronization issues (same frame basis as motion)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.feature_extractor = FeatureExtractor(config)
    
    def build_samples_from_fingering(
        self,
        original_fingering: List,
        motion: Dict,
        max_seq_len: int = 128
    ) -> List[Dict]:
        """
        Build inference samples from fingering data

        Args:
            original_fingering: Rule-based fingering data
            motion: Motion data
            max_seq_len: Maximum sequence length

        Returns:
            List of inference samples
        """
        # 1. Extract note onsets from fingering
        note_events = self._extract_note_events(original_fingering)
        
        # 2. Sort by onset and group
        note_events.sort(key=lambda x: (x.onset_frame, x.key_idx))
        note_groups = self._group_notes(note_events)
        
        # 3. Extract features and build samples
        return self._build_samples_from_groups(note_groups, motion, max_seq_len)
    
    def _extract_note_events(self, fingering: List) -> List[NoteEvent]:
        """Extract note events from fingering data (including onset + offset)"""
        # Step 1: Extract fingering blocks for each key (onset, offset, entry)
        # key_idx -> [(onset, offset, entry), ...]
        key_blocks = {}
        active_keys = {}  # key_idx -> (onset_frame, entry)
        
        for frame_idx, frame in enumerate(fingering):
            curr_keys = {}
            for entry in frame:
                if isinstance(entry, dict) and 'key_index' in entry:
                    curr_keys[entry['key_index']] = entry
            
            # Newly started notes
            for key_idx in set(curr_keys.keys()) - set(active_keys.keys()):
                entry = curr_keys[key_idx]
                active_keys[key_idx] = (frame_idx, entry)
            
            # Ended notes
            for key_idx in set(active_keys.keys()) - set(curr_keys.keys()):
                onset_frame, entry = active_keys.pop(key_idx)
                offset_frame = frame_idx - 1  # Previous frame is the last
                if key_idx not in key_blocks:
                    key_blocks[key_idx] = []
                key_blocks[key_idx].append((onset_frame, offset_frame, entry))
        
        # Notes still active at the end
        last_frame = len(fingering) - 1
        for key_idx, (onset_frame, entry) in active_keys.items():
            if key_idx not in key_blocks:
                key_blocks[key_idx] = []
            key_blocks[key_idx].append((onset_frame, last_frame, entry))
        
        # Step 2: Create NoteEvent list
        note_events = []
        for key_idx, blocks in key_blocks.items():
            for onset_frame, offset_frame, entry in blocks:
                note = NoteEvent(
                    onset_frame=onset_frame,
                    key_idx=key_idx,
                    hand=None,  # No ground truth during inference
                    finger=None,
                    original_hand=entry.get('hand'),
                    original_finger=entry.get('finger'),
                    offset_frame=offset_frame,
                )
                note_events.append(note)
        
        return note_events
    
    def _group_notes(self, note_events: List[NoteEvent]) -> List[NoteGroup]:
        """Group notes by onset time"""
        if not note_events:
            return []
        
        groups = []
        current_notes = [note_events[0]]
        current_onset = note_events[0].onset_frame
        
        for note in note_events[1:]:
            if note.onset_frame - current_onset <= self.config.onset_tolerance:
                current_notes.append(note)
            else:
                groups.append(NoteGroup(onset_frame=current_onset, notes=current_notes))
                current_notes = [note]
                current_onset = note.onset_frame
        
        if current_notes:
            groups.append(NoteGroup(onset_frame=current_onset, notes=current_notes))
        
        return groups
    
    def _build_samples_from_groups(
        self,
        note_groups: List[NoteGroup],
        motion: Dict,
        max_seq_len: int
    ) -> List[Dict]:
        """Build samples from note groups"""
        samples = []
        max_notes = self.config.max_notes_per_group
        feature_dim = self.feature_extractor.get_note_feature_dim()
        
        all_features = []
        all_original_classes = []
        all_masks = []
        all_note_infos = []  # For mapping results later
        
        for group in note_groups:
            group_features = np.zeros((max_notes, feature_dim), dtype=np.float32)
            group_original = np.zeros(max_notes, dtype=np.int64)
            group_mask = np.zeros(max_notes, dtype=bool)
            group_note_info = []
            
            for i, note in enumerate(group.notes[:max_notes]):
                features = self.feature_extractor.extract_note_features(
                    motion=motion,
                    note=note,
                    frame_idx=group.onset_frame
                )
                group_features[i] = features
                
                # Original class
                if note.original_hand and note.original_finger:
                    group_original[i] = Config.hand_finger_to_class(
                        note.original_hand, note.original_finger
                    )
                else:
                    group_original[i] = 0  # Missing (important!)
                
                group_mask[i] = True
                group_note_info.append({
                    'onset_frame': note.onset_frame,
                    'offset_frame': note.offset_frame,  # Block end frame
                    'key_idx': note.key_idx,
                    'original_class': group_original[i],
                })
            
            all_features.append(group_features)
            all_original_classes.append(group_original)
            all_masks.append(group_mask)
            all_note_infos.append(group_note_info)
        
        # Split into sequences
        num_groups = len(note_groups)
        for start_idx in range(0, num_groups, max_seq_len):
            end_idx = min(start_idx + max_seq_len, num_groups)
            
            seq_features = np.stack(all_features[start_idx:end_idx], axis=0)
            seq_original = np.stack(all_original_classes[start_idx:end_idx], axis=0)
            seq_masks = np.stack(all_masks[start_idx:end_idx], axis=0)
            
            samples.append({
                'features': seq_features,
                'original_classes': seq_original,
                'note_mask': seq_masks,
                'note_infos': all_note_infos[start_idx:end_idx],  # For result mapping
            })
        
        return samples
    
    def get_inference_statistics(self, samples: List[Dict]) -> Dict:
        """Inference data statistics"""
        total_notes = sum(s['note_mask'].sum() for s in samples)
        total_groups = sum(s['note_mask'].shape[0] for s in samples)
        
        return {
            'total_notes': int(total_notes),
            'total_groups': int(total_groups),
            'avg_notes_per_group': float(total_notes / total_groups) if total_groups > 0 else 0,
        }
