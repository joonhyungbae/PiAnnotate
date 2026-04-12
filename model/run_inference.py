#!/usr/bin/env python3
"""
Fingering Correction Inference (Fingering-based)

Usage:
    # Single piece inference
    python model/run_inference.py --piece_id 0 --checkpoint checkpoints/best.pt

    # Direct inference from fingering file
    python model/run_inference.py \
        --fingering path/to/fingering.pkl \
        --motion path/to/motion.pkl \
        --checkpoint checkpoints/best.pt
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

# Support both direct execution and module import
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.config import Config
    from model.model import FingeringCorrectionTransformer, create_model
    from model.dataset import InferenceDatasetBuilder
else:
    from .config import Config
    from .model import FingeringCorrectionTransformer, create_model
    from .dataset import InferenceDatasetBuilder


def load_model(checkpoint_path: Path, config: Config, device: torch.device):
    """Load model from checkpoint"""
    model = create_model(config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best val acc: {checkpoint.get('best_val_acc', 'N/A'):.4f}")
    
    return model


def run_inference(
    model: FingeringCorrectionTransformer,
    samples: List[Dict],
    device: torch.device,
    correction_threshold: float = 0.5
) -> List[Dict]:
    """
    Run inference.

    Returns:
        List of results (predictions for each note)
    """
    all_results = []
    
    model.eval()
    with torch.no_grad():
        for sample in samples:
            # Prepare input
            features = torch.tensor(sample['features'], dtype=torch.float32).unsqueeze(0).to(device)
            original_classes = torch.tensor(sample['original_classes'], dtype=torch.long).unsqueeze(0).to(device)
            note_mask = torch.tensor(sample['note_mask'], dtype=torch.bool).unsqueeze(0).to(device)
            seq_len = features.shape[1]
            seq_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
            
            # Run model
            final_preds, needs_correction, correction_probs = model.predict(
                features, note_mask, seq_mask, original_classes,
                correction_threshold=correction_threshold
            )
            
            # Convert to results
            final_preds = final_preds[0].cpu().numpy()
            needs_correction = needs_correction[0].cpu().numpy()
            correction_probs = correction_probs[0].cpu().numpy()
            note_mask_np = sample['note_mask']
            
            for group_idx, note_infos in enumerate(sample['note_infos']):
                for note_idx, note_info in enumerate(note_infos):
                    if not note_mask_np[group_idx, note_idx]:
                        continue
                    
                    pred_class = int(final_preds[group_idx, note_idx])
                    orig_class = int(sample['original_classes'][group_idx, note_idx])
                    corr_prob = float(correction_probs[group_idx, note_idx])
                    
                    # was_corrected: True only when prediction differs from original
                    # (model's correction_prob is always high, so it is not used)
                    was_corrected = (pred_class != orig_class) and (pred_class > 0)
                    
                    # Class to hand/finger
                    if pred_class > 0:
                        hand, finger = Config.class_to_hand_finger(pred_class)
                    else:
                        hand, finger = None, None
                    
                    # Original hand/finger
                    if orig_class > 0:
                        orig_hand, orig_finger = Config.class_to_hand_finger(orig_class)
                    else:
                        orig_hand, orig_finger = None, None
                    
                    result = {
                        'onset_frame': note_info['onset_frame'],
                        'offset_frame': note_info.get('offset_frame', note_info['onset_frame']),  # block end frame
                        'key_idx': note_info['key_idx'],
                        'predicted_hand': hand,
                        'predicted_finger': finger,
                        'predicted_class': pred_class,
                        'original_hand': orig_hand,
                        'original_finger': orig_finger,
                        'original_class': orig_class,
                        'was_corrected': was_corrected,
                        'correction_prob': corr_prob,
                    }
                    all_results.append(result)
    
    return all_results


def results_to_fingering_format(
    results: List[Dict],
    num_frames: int
) -> List[List[Dict]]:
    """
    Convert results to fingering format.

    Store fingering for the entire onset~offset range per fingering block.

    Returns:
        List[List[Dict]] - per-frame fingering list
    """
    # Group by frame
    fingering_by_frame = {i: [] for i in range(num_frames)}
    
    # Add each result to the entire onset~offset range
    for result in results:
        onset_frame = result['onset_frame']
        offset_frame = result.get('offset_frame', onset_frame)
        
        # Use only onset if offset is missing or out of range
        if offset_frame is None:
            offset_frame = onset_frame
        
        # Range validation
        onset_frame = max(0, min(onset_frame, num_frames - 1))
        offset_frame = max(onset_frame, min(offset_frame, num_frames - 1))
        
        if result['predicted_hand'] and result['predicted_finger']:
            finger_names = {1: 'thumb', 2: 'index', 3: 'middle', 4: 'ring', 5: 'pinky'}
            
            fingering_entry = {
                'key_index': result['key_idx'],
                'hand': result['predicted_hand'],
                'finger': result['predicted_finger'],
                'finger_name': finger_names.get(result['predicted_finger'], ''),
                'was_corrected': result['was_corrected'],
                'correction_prob': result['correction_prob'],
                'is_missing': result.get('is_missing', False),  # whether this is a missing note
            }
            
            # Add fingering to all frames from onset to offset
            for frame_idx in range(onset_frame, offset_frame + 1):
                if frame_idx < num_frames:
                    fingering_by_frame[frame_idx].append(fingering_entry.copy())
    
    return [fingering_by_frame[i] for i in range(num_frames)]


def print_statistics(results: List[Dict]):
    """Print result statistics"""
    total = len(results)
    corrected = sum(1 for r in results if r['was_corrected'])
    unchanged = total - corrected
    
    print(f"\n=== Inference Results ===")
    print(f"Total notes: {total}")
    print(f"Corrected: {corrected} ({100*corrected/total:.1f}%)")
    print(f"Unchanged: {unchanged} ({100*unchanged/total:.1f}%)")
    
    # Correction details
    if corrected > 0:
        print(f"\nCorrection details:")
        
        hand_changes = 0
        finger_only_changes = 0
        
        for r in results:
            if r['was_corrected']:
                if r['original_hand'] != r['predicted_hand']:
                    hand_changes += 1
                else:
                    finger_only_changes += 1
        
        print(f"  Hand changes: {hand_changes}")
        print(f"  Finger-only changes: {finger_only_changes}")


def main():
    parser = argparse.ArgumentParser(description="Fingering Correction Inference")
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--piece_id", type=int, help="Piece ID from dataset")
    group.add_argument("--fingering", type=str, help="Path to original fingering.pkl")
    
    # Additional inputs for --fingering mode
    parser.add_argument("--motion", type=str, help="Path to motion.pkl (required with --fingering)")
    
    # Model options
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Correction threshold (0-1)")
    
    # Output options
    parser.add_argument("--output", type=str, help="Output path for corrected fingering")
    parser.add_argument("--verbose", action="store_true", help="Print detailed results")
    
    args = parser.parse_args()
    
    # Setup
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    if args.piece_id is not None:
        # Dataset mode
        piece_id = args.piece_id
        motion_path = config.dataset_path / f"{piece_id:03d}" / "motion.pkl"
        fingering_path = config.fingering_path / f"{piece_id:03d}.pkl"
        
        if not motion_path.exists():
            print(f"Error: Motion file not found: {motion_path}")
            return
        if not fingering_path.exists():
            print(f"Error: Fingering file not found: {fingering_path}")
            return
        
        with open(motion_path, 'rb') as f:
            motion = pickle.load(f)
        with open(fingering_path, 'rb') as f:
            original_fingering = pickle.load(f)
            
        print(f"\nProcessing piece {piece_id}")
    else:
        # Direct file mode
        fingering_path = Path(args.fingering)
        
        if not args.motion:
            print("Error: --motion is required with --fingering")
            return
        
        motion_path = Path(args.motion)
        
        with open(motion_path, 'rb') as f:
            motion = pickle.load(f)
        with open(fingering_path, 'rb') as f:
            original_fingering = pickle.load(f)
        
        print(f"\nProcessing: {fingering_path}")
    
    # Build inference samples
    builder = InferenceDatasetBuilder(config)
    samples = builder.build_samples_from_fingering(
        original_fingering=original_fingering,
        motion=motion,
        max_seq_len=128
    )
    
    # Statistics
    stats = builder.get_inference_statistics(samples)
    print(f"  Total notes: {stats['total_notes']}")
    print(f"  Total groups: {stats['total_groups']}")
    print(f"  Avg notes per group: {stats['avg_notes_per_group']:.2f}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    model = load_model(checkpoint_path, config, device)
    
    # Run inference
    print("\nRunning inference...")
    results = run_inference(model, samples, device, args.threshold)
    
    # Print statistics
    print_statistics(results)
    
    # Verbose output
    if args.verbose:
        print("\nDetailed corrections:")
        for r in results:
            if r['was_corrected']:
                print(f"  Frame {r['onset_frame']:5d}, Key {r['key_idx']:2d}: "
                      f"{r['original_hand'] or 'N/A':5s} {r['original_finger'] or '-'} → "
                      f"{r['predicted_hand']:5s} {r['predicted_finger']} "
                      f"(prob: {r['correction_prob']:.3f})")
    
    # Save output
    if args.output:
        # Get frame count
        left_joints = motion.get('left', {}).get('joints', None)
        right_joints = motion.get('right', {}).get('joints', None)
        num_frames = max(
            left_joints.shape[0] if left_joints is not None else 0,
            right_joints.shape[0] if right_joints is not None else 0
        )
        
        fingering = results_to_fingering_format(results, num_frames)
        
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(fingering, f)
        
        print(f"\nSaved corrected fingering to: {output_path}")


if __name__ == "__main__":
    main()
