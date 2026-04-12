"""
Configuration for Fingering Correction Model (Note-level)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict
import json


@dataclass
class Config:
    """Model configuration"""
    
    # Paths
    project_root: Path = Path("/root/mf2m_fingering")
    dataset_path: Path = field(default_factory=lambda: Path("/root/mf2m_fingering/for_elise/dataset"))
    fingering_path: Path = field(default_factory=lambda: Path("/root/mf2m_fingering/data/fingering"))
    fingering_edited_path: Path = field(default_factory=lambda: Path("/root/mf2m_fingering/data/fingering_edited"))
    meshes_path: Path = field(default_factory=lambda: Path("/root/mf2m_fingering/for_elise/piano_meshes"))
    output_dir: Path = field(default_factory=lambda: Path("/root/mf2m_fingering/model/checkpoints"))
    
    # Data split (no validation set due to limited data)
    train_ratio: float = 0.85
    val_ratio: float = 0.0  # No validation set
    test_ratio: float = 0.15
    random_seed: int = 42
    
    # Note grouping
    onset_tolerance: int = 3  # Frames tolerance for simultaneous notes
    max_notes_per_group: int = 8  # Maximum notes in a chord
    context_notes: int = 16  # Number of previous notes as context
    
    # MANO joint indices for fingertips
    fingertip_indices: Dict[str, int] = field(default_factory=lambda: {
        'thumb': 4, 'index': 8, 'middle': 12, 'ring': 16, 'pinky': 20
    })
    
    # Finger mapping
    finger_to_number: Dict[str, int] = field(default_factory=lambda: {
        'thumb': 1, 'index': 2, 'middle': 3, 'ring': 4, 'pinky': 5
    })
    number_to_finger: Dict[int, str] = field(default_factory=lambda: {
        1: 'thumb', 2: 'index', 3: 'middle', 4: 'ring', 5: 'pinky'
    })
    
    # Model architecture (Transformer only)
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # Training
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 100
    early_stopping_patience: int = 15
    warmup_steps: int = 1000
    gradient_clip: float = 1.0
    
    # Label smoothing
    label_smoothing: float = 0.1
    
    # FPS for MIDI timing
    fps: float = 60000 / 1001  # NTSC standard
    
    # Number of classes
    num_fingers: int = 5  # 1-5
    num_hands: int = 2  # left, right
    num_classes: int = 11  # 0=invalid, 1-5=left fingers, 6-10=right fingers
    
    # Special tokens
    PAD_TOKEN: int = 0
    SOS_TOKEN: int = 11  # Start of sequence
    EOS_TOKEN: int = 12  # End of sequence
    
    def __post_init__(self):
        """Convert string paths to Path objects"""
        for attr in ['project_root', 'dataset_path', 'fingering_path', 
                     'fingering_edited_path', 'meshes_path', 'output_dir']:
            val = getattr(self, attr)
            if isinstance(val, str):
                setattr(self, attr, Path(val))
    
    def save(self, path: Path):
        """Save config to JSON file"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
            else:
                config_dict[key] = value
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'Config':
        """Load config from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    @staticmethod
    def hand_finger_to_class(hand: str, finger: int) -> int:
        """
        Convert (hand, finger) to class index
        0 = invalid/padding
        1-5 = left fingers (1=thumb, 5=pinky)
        6-10 = right fingers
        """
        if hand == 'left':
            return finger  # 1-5
        else:
            return finger + 5  # 6-10
    
    @staticmethod
    def class_to_hand_finger(class_idx: int) -> tuple:
        """Convert class index to (hand, finger)"""
        if class_idx <= 0 or class_idx > 10:
            return None, None
        if class_idx <= 5:
            return 'left', class_idx
        else:
            return 'right', class_idx - 5
    
    @property
    def vocab_size(self) -> int:
        """Vocabulary size including special tokens"""
        return 13  # 0=pad, 1-10=fingerings, 11=SOS, 12=EOS
