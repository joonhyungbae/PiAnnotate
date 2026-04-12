"""
Utility functions for Fingering Correction Model
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_motion_data(dataset_path: Path, piece_id: int) -> Dict:
    """Load motion data for a piece"""
    motion_path = dataset_path / f"{piece_id:03d}" / "motion.pkl"
    with open(motion_path, 'rb') as f:
        return pickle.load(f)


def load_fingering_data(fingering_path: Path, piece_id: int) -> List:
    """Load fingering data for a piece"""
    path = fingering_path / f"{piece_id:03d}.pkl"
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_fingering_data(fingering_data: List, output_path: Path):
    """Save fingering data"""
    with open(output_path, 'wb') as f:
        pickle.dump(fingering_data, f)


def compute_class_weights(samples: List[Dict]) -> torch.Tensor:
    """Compute class weights for imbalanced data"""
    from collections import Counter
    
    class_counts = Counter()
    for s in samples:
        if s['is_valid'] > 0.5:
            class_counts[s['target_class']] += 1
    
    total = sum(class_counts.values())
    num_classes = 10
    
    weights = []
    for cls in range(num_classes):
        count = class_counts.get(cls, 1)
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def analyze_dataset_statistics(samples: List[Dict]) -> Dict:
    """Analyze dataset statistics"""
    from collections import Counter
    
    total = len(samples)
    valid_count = sum(1 for s in samples if s['is_valid'] > 0.5)
    
    # Class distribution
    class_counts = Counter()
    for s in samples:
        if s['is_valid'] > 0.5:
            class_counts[s['target_class']] += 1
    
    # Per-piece statistics
    piece_counts = Counter()
    for s in samples:
        piece_counts[s['piece_id']] += 1
    
    return {
        'total_samples': total,
        'valid_samples': valid_count,
        'invalid_samples': total - valid_count,
        'class_distribution': dict(class_counts),
        'num_pieces': len(piece_counts),
        'samples_per_piece': {
            'mean': np.mean(list(piece_counts.values())),
            'min': min(piece_counts.values()),
            'max': max(piece_counts.values())
        }
    }


def export_predictions_to_json(
    predictions: List[List[Dict]],
    output_path: Path,
    piece_id: int
):
    """Export predictions to JSON format for visualization"""
    output = {
        'piece_id': piece_id,
        'num_frames': len(predictions),
        'frames': []
    }
    
    for frame_idx, frame_data in enumerate(predictions):
        if frame_data:
            frame_entry = {
                'frame_idx': frame_idx,
                'fingerings': []
            }
            for entry in frame_data:
                frame_entry['fingerings'].append({
                    'key_index': entry['key_index'],
                    'hand': entry['hand'],
                    'finger': entry['finger'],
                    'finger_name': entry.get('finger_name', ''),
                    'confidence': entry.get('confidence', 1.0)
                })
            output['frames'].append(frame_entry)
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)


def compare_fingering_files(
    original_path: Path,
    corrected_path: Path,
    edited_path: Optional[Path] = None
) -> Dict:
    """
    Compare fingering files
    
    Args:
        original_path: Path to original (rule-based) fingering
        corrected_path: Path to model-corrected fingering
        edited_path: Optional path to human-edited ground truth
    
    Returns:
        Comparison statistics
    """
    with open(original_path, 'rb') as f:
        original = pickle.load(f)
    
    with open(corrected_path, 'rb') as f:
        corrected = pickle.load(f)
    
    ground_truth = None
    if edited_path and edited_path.exists():
        with open(edited_path, 'rb') as f:
            ground_truth = pickle.load(f)
    
    num_frames = min(len(original), len(corrected))
    
    stats = {
        'total_frames': num_frames,
        'original_entries': 0,
        'corrected_entries': 0,
        'changed_entries': 0,
        'added_entries': 0,
        'removed_entries': 0,
    }
    
    for frame_idx in range(num_frames):
        orig_dict = {e['key_index']: e for e in original[frame_idx] 
                    if isinstance(e, dict) and 'key_index' in e}
        corr_dict = {e['key_index']: e for e in corrected[frame_idx]
                    if isinstance(e, dict) and 'key_index' in e}
        
        stats['original_entries'] += len(orig_dict)
        stats['corrected_entries'] += len(corr_dict)
        
        # Added
        stats['added_entries'] += len(set(corr_dict.keys()) - set(orig_dict.keys()))
        
        # Removed
        stats['removed_entries'] += len(set(orig_dict.keys()) - set(corr_dict.keys()))
        
        # Changed
        for key_idx in set(orig_dict.keys()) & set(corr_dict.keys()):
            if (orig_dict[key_idx].get('finger') != corr_dict[key_idx].get('finger') or
                orig_dict[key_idx].get('hand') != corr_dict[key_idx].get('hand')):
                stats['changed_entries'] += 1
    
    # Compare with ground truth if available
    if ground_truth:
        orig_correct = 0
        corr_correct = 0
        total_gt = 0
        
        for frame_idx in range(min(num_frames, len(ground_truth))):
            gt_dict = {e['key_index']: e for e in ground_truth[frame_idx]
                      if isinstance(e, dict) and 'key_index' in e}
            orig_dict = {e['key_index']: e for e in original[frame_idx]
                        if isinstance(e, dict) and 'key_index' in e}
            corr_dict = {e['key_index']: e for e in corrected[frame_idx]
                        if isinstance(e, dict) and 'key_index' in e}
            
            for key_idx, gt_entry in gt_dict.items():
                total_gt += 1
                
                # Check original
                if key_idx in orig_dict:
                    orig_e = orig_dict[key_idx]
                    if (orig_e.get('hand') == gt_entry.get('hand') and
                        orig_e.get('finger') == gt_entry.get('finger')):
                        orig_correct += 1
                
                # Check corrected
                if key_idx in corr_dict:
                    corr_e = corr_dict[key_idx]
                    if (corr_e.get('hand') == gt_entry.get('hand') and
                        corr_e.get('finger') == gt_entry.get('finger')):
                        corr_correct += 1
        
        stats['ground_truth_comparison'] = {
            'total': total_gt,
            'original_accuracy': orig_correct / total_gt if total_gt > 0 else 0,
            'corrected_accuracy': corr_correct / total_gt if total_gt > 0 else 0,
            'improvement': (corr_correct - orig_correct) / total_gt if total_gt > 0 else 0
        }
    
    return stats


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
