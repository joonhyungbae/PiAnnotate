#!/usr/bin/env python3
"""
Batch Inference for Unannotated Pieces

Perform AI annotation on pieces that have not yet been annotated by humans.
Results are saved to the fingering_edited_ai/ folder.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional

import torch
import numpy as np
from tqdm import tqdm

# Support both direct execution and module import
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from model.config import Config
    from model.model import create_model
    from model.dataset import InferenceDatasetBuilder
    from model.run_inference import run_inference, results_to_fingering_format
    from model.missing_notes import process_missing_notes_with_model, merge_results_with_missing
else:
    from .config import Config
    from .model import create_model
    from .dataset import InferenceDatasetBuilder
    from .run_inference import run_inference, results_to_fingering_format
    from .missing_notes import process_missing_notes_with_model, merge_results_with_missing


def get_annotated_ids(fingering_edited_path: Path) -> Set[int]:
    """Set of piece IDs annotated by humans (files exist in fingering_edited)"""
    ids = set()
    for f in fingering_edited_path.glob("*.pkl"):
        try:
            piece_id = int(f.stem)
            ids.add(piece_id)
        except ValueError:
            continue
    return ids


def get_human_completed_ids(status_path: Path) -> Set[int]:
    """Set of piece IDs with completed human annotation (review1 is not null)"""
    ids = set()
    if not status_path.exists():
        return ids
    
    for f in status_path.glob("*.json"):
        try:
            piece_id = int(f.stem)
            with open(f, 'r') as fp:
                status = json.load(fp)
            # If review1 is not null, human annotation is complete
            if status.get('review1') is not None:
                ids.add(piece_id)
        except (ValueError, json.JSONDecodeError):
            continue
    return ids


def get_all_ids(fingering_path: Path) -> Set[int]:
    """Set of all piece IDs"""
    ids = set()
    for f in fingering_path.glob("*.pkl"):
        try:
            piece_id = int(f.stem)
            ids.add(piece_id)
        except ValueError:
            continue
    return ids


def load_model(checkpoint_path: Path, config: Config, device: torch.device):
    """Load model from checkpoint"""
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def filter_corrections_by_finger(
    results: List[Dict],
    ignore_fingers: Optional[List[int]] = None
) -> List[Dict]:
    """
    Ignore corrections involving specific fingers and revert to original values.

    Args:
        results: List of results from run_inference
        ignore_fingers: List of finger numbers to ignore (e.g., [1, 2] -> thumb, index)
                       None means no filtering

    Returns:
        Filtered results list

    Ignore conditions:
        - Original finger is in ignore_fingers, or
        - Predicted finger is in ignore_fingers
        -> Keep original value without correction
    """
    if not ignore_fingers:
        return results
    
    ignore_set = set(ignore_fingers)
    filtered_results = []
    
    for r in results:
        new_r = r.copy()
        
        if r['was_corrected']:
            orig_finger = r['original_finger']
            pred_finger = r['predicted_finger']
            
            # Cancel correction if original or predicted finger is in the ignore set
            should_ignore = False
            if orig_finger in ignore_set or pred_finger in ignore_set:
                should_ignore = True
            
            if should_ignore:
                # Revert to original value
                new_r['predicted_hand'] = r['original_hand']
                new_r['predicted_finger'] = r['original_finger']
                new_r['predicted_class'] = r['original_class']
                new_r['was_corrected'] = False
                new_r['filtered_reason'] = f"ignored_finger_{orig_finger}_or_{pred_finger}"
        
        filtered_results.append(new_r)
    
    return filtered_results


def process_piece(
    piece_id: int,
    model,
    config: Config,
    device: torch.device,
    correction_threshold: float = 0.5,
    handle_missing: bool = True,
    ignore_fingers: Optional[List[int]] = None
) -> List[List[Dict]]:
    """Process a single piece (model inference + missing note handling)

    Args:
        ignore_fingers: List of finger numbers to ignore corrections for (e.g., [1, 2] -> ignore thumb, index)
    """
    
    # Load data
    motion_path = config.dataset_path / f"{piece_id:03d}" / "motion.pkl"
    fingering_path = config.fingering_path / f"{piece_id:03d}.pkl"
    
    if not motion_path.exists():
        raise FileNotFoundError(f"Motion not found: {motion_path}")
    if not fingering_path.exists():
        raise FileNotFoundError(f"Fingering not found: {fingering_path}")
    
    with open(motion_path, 'rb') as f:
        motion = pickle.load(f)
    with open(fingering_path, 'rb') as f:
        original_fingering = pickle.load(f)
    
    # Get frame count
    left_joints = motion.get('left', {}).get('joints', None)
    right_joints = motion.get('right', {}).get('joints', None)
    num_frames = max(
        left_joints.shape[0] if left_joints is not None else 0,
        right_joints.shape[0] if right_joints is not None else 0
    )
    
    # Build inference samples
    builder = InferenceDatasetBuilder(config)
    samples = builder.build_samples_from_fingering(
        original_fingering=original_fingering,
        motion=motion,
        max_seq_len=128
    )
    
    if not samples:
        # No valid samples, return original fingering
        return original_fingering, [], 0
    
    # Run model inference
    results = run_inference(model, samples, device, correction_threshold)
    
    # Handle missing notes (notes present in MIDI but missing from rule-based results)
    missing_count = 0
    if handle_missing:
        missing_results, missing_count = process_missing_notes_with_model(
            piece_id=piece_id,
            fingering=original_fingering,
            motion=motion,
            model=model,
            config=config,
            device=device,
            correction_threshold=correction_threshold,
            verbose=False
        )
        
        if missing_results:
            results = merge_results_with_missing(results, missing_results)
    
    # Filter corrections by finger (ignore corrections for specific fingers)
    if ignore_fingers:
        results = filter_corrections_by_finger(results, ignore_fingers)
    
    # Convert to fingering format
    fingering = results_to_fingering_format(results, num_frames)
    
    return fingering, results, missing_count


def main():
    parser = argparse.ArgumentParser(description="Batch AI Annotation for Unannotated Pieces")
    
    parser.add_argument("--checkpoint", type=str, default="model/checkpoints/best.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="data/fingering_edited_ai",
                        help="Output directory for AI annotations")
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="Correction threshold (0-1), higher means fewer corrections")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing AI annotations")
    parser.add_argument("--include-annotated", action="store_true",
                        help="Also process pieces that have human annotations (for comparison)")
    parser.add_argument("--ignore-fingers", type=str, default=None,
                        help="Comma-separated finger numbers to ignore corrections (e.g., '1,2' for thumb,index)")
    
    args = parser.parse_args()
    
    # Parse ignore_fingers
    ignore_fingers = None
    if args.ignore_fingers:
        try:
            ignore_fingers = [int(x.strip()) for x in args.ignore_fingers.split(',')]
            print(f"Ignoring corrections involving fingers: {ignore_fingers}")
        except ValueError:
            print(f"Error: Invalid --ignore-fingers format: {args.ignore_fingers}")
            print("  Expected comma-separated integers, e.g., '1,2'")
            return
    
    # Setup
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get piece IDs
    all_ids = get_all_ids(config.fingering_path)
    
    # Check pieces with completed review1 from status files (human completion criterion)
    status_path = Path("data/metadata/annotation/status")
    human_completed_ids = get_human_completed_ids(status_path)
    
    if args.include_annotated:
        target_ids = all_ids
        print(f"\nProcessing ALL pieces (including human-completed)")
    else:
        target_ids = all_ids - human_completed_ids
        print(f"\nProcessing pieces without human annotation (review1=null)")
    
    print(f"  Total pieces: {len(all_ids)}")
    print(f"  Human-completed (review1 != null): {len(human_completed_ids)}")
    print(f"  To process: {len(target_ids)}")
    
    # Filter by existing motion data
    valid_ids = []
    for piece_id in sorted(target_ids):
        motion_path = config.dataset_path / f"{piece_id:03d}" / "motion.pkl"
        if motion_path.exists():
            valid_ids.append(piece_id)
    
    print(f"  With motion data: {len(valid_ids)}")
    
    if not valid_ids:
        print("\nNo pieces to process!")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check existing
    if not args.force:
        existing = [pid for pid in valid_ids if (output_dir / f"{pid:03d}.pkl").exists()]
        if existing:
            print(f"  Already processed: {len(existing)} (use --force to overwrite)")
            valid_ids = [pid for pid in valid_ids if pid not in existing]
    
    print(f"  Final to process: {len(valid_ids)}")
    
    if not valid_ids:
        print("\nNothing to process!")
        return
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\nLoading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, config, device)
    
    # Process pieces
    print(f"\nProcessing {len(valid_ids)} pieces...")
    
    stats = {
        'total_notes': 0,
        'corrected_notes': 0,
        'missing_notes': 0,
        'failed_pieces': [],
    }
    
    for piece_id in tqdm(valid_ids, desc="AI Annotation"):
        try:
            fingering, results, missing_count = process_piece(
                piece_id, model, config, device, args.threshold,
                handle_missing=True,
                ignore_fingers=ignore_fingers
            )
            
            # Save
            output_path = output_dir / f"{piece_id:03d}.pkl"
            with open(output_path, 'wb') as f:
                pickle.dump(fingering, f)
            
            # Stats
            stats['total_notes'] += len(results)
            stats['corrected_notes'] += sum(1 for r in results if r['was_corrected'])
            stats['missing_notes'] += missing_count
            
        except Exception as e:
            stats['failed_pieces'].append((piece_id, str(e)))
            tqdm.write(f"  Failed piece {piece_id}: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"Processed: {len(valid_ids) - len(stats['failed_pieces'])} pieces")
    print(f"Failed: {len(stats['failed_pieces'])} pieces")
    print(f"Total notes: {stats['total_notes']}")
    print(f"Corrected: {stats['corrected_notes']} ({100*stats['corrected_notes']/max(1,stats['total_notes']):.1f}%)")
    print(f"Missing notes found: {stats['missing_notes']}")
    print(f"Output: {output_dir}/")
    
    if stats['failed_pieces']:
        print(f"\nFailed pieces:")
        for pid, err in stats['failed_pieces']:
            print(f"  {pid:03d}: {err}")


if __name__ == "__main__":
    main()
