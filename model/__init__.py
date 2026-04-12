"""
Fingering Correction Model (Transformer with Note-level Sequence)

Deep learning-based piano fingering auto-correction model
- Note-level sampling (15x more efficient than frame-level)
- Simultaneous note (chord) modeling
- Sequence-to-Sequence Transformer
"""

from .config import Config
from .features import FeatureExtractor, NoteEvent, NoteGroup
from .dataset import (
    FingeringSequenceDataset, 
    DatasetBuilder, 
    create_dataloaders,
    save_samples,
    load_samples
)
from .model import FingeringCorrectionTransformer, CorrectionLoss, create_model
from .train import Trainer

__all__ = [
    'Config',
    'FeatureExtractor',
    'NoteEvent',
    'NoteGroup',
    'FingeringSequenceDataset',
    'DatasetBuilder',
    'create_dataloaders',
    'save_samples',
    'load_samples',
    'FingeringTransformer',
    'FingeringLoss',
    'create_model',
    'Trainer',
]
