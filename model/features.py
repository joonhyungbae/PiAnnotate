"""
Feature extraction for Fingering Correction Model (Note-level)
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from .config import Config


@dataclass
class NoteEvent:
    """A single note event"""
    onset_frame: int
    key_idx: int
    hand: Optional[str] = None
    finger: Optional[int] = None
    original_hand: Optional[str] = None
    original_finger: Optional[int] = None
    offset_frame: Optional[int] = None  # end frame of fingering block


@dataclass
class NoteGroup:
    """A group of simultaneous notes (chord)"""
    onset_frame: int
    notes: List[NoteEvent] = field(default_factory=list)


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


class FeatureExtractor:
    """Extract features for note-level fingering prediction"""
    
    BLACK_KEY_OFFSETS = [1, 4, 6, 9, 11]
    
    def __init__(self, config: Config):
        self.config = config
        self.key_bounds: Dict[int, KeyBounds] = {}
        self._load_key_bounds()
    
    def _load_key_bounds(self):
        """Load key bounds from piano mesh files"""
        meshes_dir = self.config.meshes_path
        
        if not meshes_dir.exists():
            print(f"Warning: Piano meshes directory not found: {meshes_dir}")
            return
        
        for key_idx in range(88):
            obj_path = meshes_dir / f"{key_idx}.obj"
            if not obj_path.exists():
                continue
            
            vertices = self._parse_obj_file(obj_path)
            
            x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
            y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
            z_max = vertices[:, 2].max()
            
            # X-axis 90-degree rotation: [x, y, z] -> [x, -z, y]
            rot_y_min, rot_y_max = -z_max, -vertices[:, 2].min()
            rot_z_max = y_max
            
            self.key_bounds[key_idx] = KeyBounds(
                key_idx=key_idx,
                y_min=-vertices[:, 2].max(),
                y_max=-vertices[:, 2].min(),
                x_min=x_min,
                x_max=x_max,
                z_surface=y_max,
                is_black=self._is_black_key(key_idx)
            )
    
    @staticmethod
    def _parse_obj_file(obj_path: Path) -> np.ndarray:
        """Extract vertices from OBJ file"""
        vertices = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.split()
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(vertices, dtype=np.float32)
    
    @staticmethod
    def _is_black_key(key_idx: int) -> bool:
        return (key_idx % 12) in FeatureExtractor.BLACK_KEY_OFFSETS
    
    def get_note_feature_dim(self) -> int:
        """Get feature dimension for a single note"""
        # Key features: 5 (index, is_black, center_y, center_x, z_surface)
        # Per-finger features: 10 fingers × 6 features = 60
        # Original prediction: 12 (hand one-hot + finger one-hot + flags)
        return 5 + 60 + 12
    
    def extract_note_features(
        self,
        motion: Dict,
        note: NoteEvent,
        frame_idx: int
    ) -> np.ndarray:
        """
        Extract features for a single note
        
        Args:
            motion: Motion data with 'left' and 'right' hand joints
            note: NoteEvent object
            frame_idx: Frame index for feature extraction
        
        Returns:
            Feature vector (77 dims)
        """
        features = []
        
        key_idx = note.key_idx
        bounds = self.key_bounds.get(key_idx)
        
        # 1. Key features (5 dims)
        if bounds:
            key_center_y = (bounds.y_min + bounds.y_max) / 2
            key_center_x = (bounds.x_min + bounds.x_max) / 2
            features.extend([
                key_idx / 87.0,  # Normalized key index
                float(bounds.is_black),
                key_center_y,
                key_center_x,
                bounds.z_surface,
            ])
        else:
            features.extend([key_idx / 87.0, 0.0, 0.0, 0.0, 0.0])
        
        # 2. Per-finger features (60 dims: 10 fingers × 6 features)
        for hand in ['left', 'right']:
            joints = motion.get(hand, {}).get('joints', None)
            
            if joints is None or frame_idx >= joints.shape[0]:
                features.extend([0.0] * 30)  # 5 fingers × 6 features
                continue
            
            frame_joints = joints[frame_idx]
            
            for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                tip_idx = self.config.fingertip_indices[finger_name]
                tip_pos = frame_joints[tip_idx]
                tip_x, tip_y, tip_z = tip_pos[0], tip_pos[1], tip_pos[2]
                
                if bounds:
                    key_center_y = (bounds.y_min + bounds.y_max) / 2
                    key_center_x = (bounds.x_min + bounds.x_max) / 2
                    
                    # Relative position to key
                    y_dist = tip_y - key_center_y
                    x_dist = tip_x - key_center_x
                    z_dist = tip_z - bounds.z_surface
                    
                    # In range checks
                    in_y_range = float(bounds.y_min - 0.01 <= tip_y <= bounds.y_max + 0.01)
                    in_z_range = float(abs(z_dist) < 0.02)
                else:
                    y_dist, x_dist, z_dist = 0, 0, 0
                    in_y_range, in_z_range = 0, 0
                
                features.extend([
                    y_dist,
                    x_dist,
                    z_dist,
                    tip_z,  # Absolute height
                    in_y_range,
                    in_z_range,
                ])
        
        # 3. Original prediction features (12 dims)
        orig_hand = note.original_hand
        orig_finger = note.original_finger
        
        # Hand one-hot (2 dims)
        features.append(float(orig_hand == 'left'))
        features.append(float(orig_hand == 'right'))
        
        # Finger one-hot (5 dims)
        for f in range(1, 6):
            features.append(float(orig_finger == f))
        
        # Prediction confidence flags (5 dims)
        has_prediction = float(orig_hand is not None and orig_finger is not None)
        features.extend([
            has_prediction,
            float(orig_hand == note.hand) if has_prediction else 0,  # Hand match
            float(orig_finger == note.finger) if has_prediction else 0,  # Finger match
            0.0, 0.0  # Reserved
        ])
        
        return np.array(features, dtype=np.float32)
    
    def extract_context_features(
        self,
        motion: Dict,
        note_groups: List[NoteGroup],
        group_idx: int,
        context_size: int = 8
    ) -> np.ndarray:
        """
        Extract context features from previous note groups
        
        Returns:
            Context feature vector
        """
        context_features = []
        
        for offset in range(-context_size, 0):
            ctx_idx = group_idx + offset
            
            if ctx_idx < 0 or ctx_idx >= len(note_groups):
                # Padding
                context_features.extend([0.0] * 15)  # Per-group context dim
            else:
                group = note_groups[ctx_idx]
                
                # Group-level features
                num_notes = len(group.notes)
                avg_key = np.mean([n.key_idx for n in group.notes]) / 87.0
                
                # Fingers used (one-hot sum for each hand)
                left_fingers = [0] * 5
                right_fingers = [0] * 5
                
                for n in group.notes:
                    if n.finger:
                        if n.hand == 'left':
                            left_fingers[n.finger - 1] = 1
                        else:
                            right_fingers[n.finger - 1] = 1
                
                context_features.extend([
                    num_notes / 8.0,  # Normalized note count
                    avg_key,
                    *left_fingers,
                    *right_fingers,
                    (group_idx - ctx_idx) / context_size,  # Relative position
                    0.0, 0.0  # Reserved
                ])
        
        return np.array(context_features, dtype=np.float32)
