# Fingering Correction Transformer

Deep learning-based piano fingering **correction** model

## Key Features

### Explicit Correction Model
Previous models tried to "always predict the correct answer," but this model:
- **If the original is correct -> keep as is**
- **If the original is wrong -> output corrected value**
- **If the original is missing -> predict new value (fill in missing)**

### Fingering-Based Input
```
Both training and inference:
  Input = [notes from fingering data]

Advantages:
  - Same frame basis as motion data
  - No frame drift issues
  - Consistent processing approach
```

### Three-Task Learning
| Task | Description |
|------|-------------|
| **Needs Correction** | Does this fingering need to be corrected? |
| **Fingering Correction** | If so, what fingering should it be? |
| **Missing Fill** | If there is no prediction, what fingering? |

## Model Architecture

```
Input:
  - note_features: (batch, seq, notes, 77)
  - original_classes: (batch, seq, notes)
       |
+------------------------------------------+
| Note Encoder + Original Class Embedding  |
|   - Combines note features with original |
|     prediction                           |
|   - original_class=0 -> missing note     |
+------------------------------------------+
       |
+------------------------------------------+
| Transformer Encoder                       |
|   - Causal attention (sequence context)  |
|   - Learns previous fingering patterns   |
+------------------------------------------+
       |
+------------------------------------------+
| Dual Prediction Heads                     |
|   - correction_head -> needs correction  |
|   - classifier -> corrected fingering    |
+------------------------------------------+
       |
Output:
  - needs_correction: (batch, seq, notes)
  - corrected_class: (batch, seq, notes, 11)
```

## Inference Logic

```python
# Final prediction logic
if original_class == 0:  # Missing note
    final = model_prediction   # Always predict new value
elif needs_correction:        # Needs correction
    final = model_prediction   # Corrected value
else:                          # No correction needed
    final = original_class     # Keep original
```

## Usage

### Training

#### Training Data
- **Only R1 (Review 1) checked data** is used for training
- R1 check status is determined by the `review1.completed_at` field in `data/metadata/annotation/status/{piece_id}.json`
- R1-checked data is automatically filtered (`dataset.py`'s `_find_common_piece_ids()`)

#### Running Training

```bash
# Method 1: Use train.sh (recommended)
./train.sh

# Method 2: Run directly
python model/run_training.py --rebuild-samples --epochs 100

# Background execution
nohup ./train.sh > training.log 2>&1 &
```

#### Main Options

| Option | Description | Default |
|--------|-------------|---------|
| `--rebuild-samples` | Rebuild sample cache (required when data changes) | - |
| `--epochs` | Maximum number of epochs | 100 |
| `--batch-size` | Batch size | 32 |
| `--patience` | Early stopping patience | 15 |
| `--lr` | Learning rate | 1e-4 |

```bash
# View all options
python model/run_training.py --help
```

#### Training Output
- Checkpoint: `model/checkpoints/best.pt`
- Training logs: `model/checkpoints/logs/` (TensorBoard)
- Sample cache: `model/checkpoints/samples.pkl`

---

### Batch Inference (AI Annotation)

Performs AI annotation on data without R1 checks and saves results to `data/fingering_edited_ai/`.

```bash
# Run AI inference on data without R1 checks
python model/batch_inference.py --threshold 0.9

# Overwrite existing results (after model retraining)
python model/batch_inference.py --threshold 0.9 --force

# Process all data (including R1-checked, for comparison)
python model/batch_inference.py --threshold 0.9 --include-annotated --force
```

#### Main Options

| Option | Description | Default |
|--------|-------------|---------|
| `--threshold` | Correction threshold (higher = more conservative) | 0.9 |
| `--force` | Overwrite existing results | - |
| `--include-annotated` | Include R1-checked pieces | - |
| `--checkpoint` | Model checkpoint path | model/checkpoints/best.pt |
| `--output-dir` | Output directory | data/fingering_edited_ai |

---

### Single Piece Inference

```bash
# Run inference on a specific piece from the dataset
python model/run_inference.py \
    --piece_id 0 \
    --checkpoint model/checkpoints/best.pt \
    --output output/corrected_000.pkl

# Run inference directly with fingering files
python model/run_inference.py \
    --fingering path/to/original.pkl \
    --motion path/to/motion.pkl \
    --checkpoint model/checkpoints/best.pt

# Verbose output
python model/run_inference.py --piece_id 0 --verbose
```

### Inference Result Example

```
=== Inference Results ===
Total notes: 4565
Corrected: 156 (3.4%)   <- Incorrect predictions corrected
Unchanged: 4409 (96.6%) <- Original kept
```

## File Structure

```
model/
├── __init__.py          # Module initialization
├── config.py            # Configuration
├── features.py          # NoteEvent, NoteGroup, FeatureExtractor
├── dataset.py           # Training/inference datasets
│   ├── DatasetBuilder          # For training (R1-checked data only)
│   └── InferenceDatasetBuilder # For inference
├── model.py             # FingeringCorrectionTransformer
│   ├── correction_head  # Predicts whether correction is needed
│   └── classifier       # Predicts corrected fingering
├── train.py             # Training logic
├── run_training.py      # Training execution script
├── run_inference.py     # Single piece inference
├── batch_inference.py   # Batch AI annotation
├── missing_notes.py     # Missing note handling
├── utils.py             # Utilities
├── checkpoints/         # Model checkpoints
│   ├── best.pt          # Best performing model
│   ├── final.pt         # Final model
│   ├── samples.pkl      # Sample cache
│   └── logs/            # TensorBoard logs
└── README.md            # This file

data/
├── fingering/           # Rule-based fingering (input)
├── fingering_edited/    # Human-corrected fingering (training ground truth)
├── fingering_edited_ai/ # AI-generated fingering (inference results)
└── metadata/annotation/status/
    └── {piece_id}.json  # Annotation status (R1 check status)
```

## Data Flow

### During Training
```
fingering_edited (ground truth)
        |
        +-> Extract note onsets
        |
        v
+---------------------------+
| For each note:            |
| - target = edited value   |
| - original = fingering    |
|   value (0 if missing)    |
+---------------------------+
```

### During Inference
```
fingering (rule-based)
        |
        +-> Extract note onsets
        |
        v
+---------------------------+
| For each note:            |
| - original = fingering    |
|   value                   |
+---------------------------+
        |
        v
    Model inference
        |
        v
+---------------------------+
| needs_correction -> fix   |
| else -> keep original     |
+---------------------------+
```

## Why This Approach?

### Problem 1: Previous Approach (Prediction Instead of Correction)
```
Predict from scratch for every note
-> 95% are already correct, why re-predict?
-> Even correct ones might become wrong
```

### Solution: Explicit Correction Model
```
Input = notes from fingering data
-> Model determines whether correction is needed (needs_correction)
-> If correction needed, predict new value; otherwise keep original
-> Correct ones stay, only wrong ones get fixed
```

### Why Fingering-Based? (Instead of MIDI)
```
MIDI-Motion frame drift discovered:
  - First 5000 frames: 99.6% matching
  - After that: 10-30% matching (synchronization breaks)

Fingering-based:
  - Same frame basis as motion
  - 100% synchronization
```

## Overall Workflow

```
+-------------------------------------------------------------+
| 1. Data Preparation                                          |
|    - Humans annotate in fingering_edited/                    |
|    - When R1 check is complete, record review1 in            |
|      status/{piece_id}.json                                  |
+-------------------------------------------------------------+
                            |
+-------------------------------------------------------------+
| 2. Model Training                                            |
|    $ ./train.sh                                              |
|    - Automatically selects only R1-checked data              |
|    - Output: model/checkpoints/best.pt                       |
+-------------------------------------------------------------+
                            |
+-------------------------------------------------------------+
| 3. AI Annotation (data without R1 check)                     |
|    $ python model/batch_inference.py --threshold 0.9 --force |
|    - Run inference on data without R1 check                  |
|    - Output: data/fingering_edited_ai/                       |
+-------------------------------------------------------------+
                            |
+-------------------------------------------------------------+
| 4. Iterate                                                   |
|    - Humans R1-check more data -> retrain -> update AI       |
|      inference                                               |
+-------------------------------------------------------------+
```

## License

MIT License
