# PiAnnotate

A web-based annotation tool and diagnostic probe for piano fingering, built on the [FurElise](https://arxiv.org/abs/2410.05791) dataset (Wang et al., SIGGRAPH Asia 2024).

PiAnnotate adds expert fingering labels to the 153 piano performances in FurElise. The tool renders the FurElise 3D hand-mesh tracks on a top-down piano view, lets an annotator assign finger numbers frame by frame, and tracks a multi-pass review workflow (R1/R2/R3). A small Transformer probe trained on the resulting annotations serves as a diagnostic instrument to verify that the corpus carries learnable signal.

## Prerequisites

- [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Node.js](https://nodejs.org/) (v18+)
- The FurElise dataset (see [Download FurElise](#download-furelise) below)

## Download FurElise

PiAnnotate requires the FurElise dataset for the source recordings, hand meshes, and MIDI data. Download it from Hugging Face:

```bash
# Install huggingface_hub if needed
pip install huggingface_hub

# Download to for_elise/
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='rcwang/for_elise', repo_type='dataset', local_dir='for_elise')
"
```

Or clone directly:
```bash
git lfs install
git clone https://huggingface.co/datasets/rcwang/for_elise for_elise
```

After download, your directory should contain `for_elise/dataset/` with per-piece subdirectories.

## Download MANO

The 3D hand mesh rendering requires the MANO hand model (v1.2):

1. Register and download from https://mano.is.tue.mpg.de/
2. Place the model files in `web/mano_v1_2/`

## Quick Start

### Installation

**Linux / macOS:**
```bash
bash setup.sh
```

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

This will:
1. Create the conda environment
2. Install Python and Node.js dependencies
3. Download MANO hand models and static resources

### Run

**Linux / macOS:**
```bash
bash start.sh
```

**Windows (PowerShell):**
```powershell
.\start.ps1
```

Then open http://localhost:3000.

- Frontend: http://localhost:3000
- Backend API: http://localhost:8080

### Manual Start

Terminal 1 (Flask backend):
```bash
cd web
conda activate tipianotation
python server.py
```

Terminal 2 (React frontend):
```bash
cd web
npm run dev
```

## Annotation Tool

The annotator sees a top-down piano with the FurElise hand meshes rendered in 3D for the current frame. Pressed keys are highlighted and overlaid with finger numbers (colour-coded by hand). A side panel allows assigning fingerings by key, hand, and finger number. A bottom timeline shows audio waveform with onset markers.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Play / Pause |
| `<-` `->` | Previous / Next Fingering |
| `1`--`5` | Assign finger number |
| `F` | Toggle Fingering Display |
| `ESC` | Deselect Fingering |
| `?` | Show Help |

### Review Workflow

Each piece tracks three review stages in a status JSON (`data/metadata/annotation/status/{piece_id}.json`):

- **R1**: First expert review of rule-based fingering
- **R2**: Second review (optionally with AI suggestions)
- **R3**: Third review pass

Only R1-checked pieces are used for probe training.

## Project Structure

```
PiAnnotate/
├── web/                          # Annotation tool (React + Flask)
│   ├── server.py                 #   Flask backend
│   └── src/                      #   React frontend
├── model/                        # Diagnostic probe
│   ├── model.py                  #   FingeringCorrectionTransformer
│   ├── dataset.py                #   Dataset builder
│   ├── train.py                  #   Training loop
│   ├── run_training.py           #   Training entry point
│   ├── run_inference.py          #   Single-piece inference
│   └── batch_inference.py        #   Batch AI annotation
├── experiments/                  # Analysis scripts for the paper
│   ├── holdout_train.py          #   Held-out training
│   ├── holdout_infer.py          #   Held-out inference
│   ├── analyze.py                #   Evaluation + triage metrics
│   ├── aggregate_seeds.py        #   Seed variance + bootstrap CI
│   ├── baseline_gbdt.py          #   GBDT non-sequence baseline
│   └── seed_runs/                #   Result JSONs
├── src/                          # Preprocessing scripts
├── data/                         # Created at runtime by the tool
├── for_elise/                    # FurElise dataset (not included, download separately)
├── setup.sh / setup.ps1          # Installation scripts
├── start.sh                      # Launch script
└── train.sh                      # Model training script
```

## Diagnostic Probe

A small Transformer trained on (rule, edited) fingering pairs to test whether the corpus carries learnable signal. The probe is **not** a deployable corrector -- it is a diagnostic instrument.

### Training

```bash
./train.sh
```

Or directly:
```bash
python model/run_training.py --rebuild-samples --epochs 100
```

| Option | Description | Default |
|--------|-------------|---------|
| `--rebuild-samples` | Rebuild sample cache (required after data changes) | - |
| `--epochs` | Maximum epochs | 100 |
| `--batch-size` | Batch size | 32 |
| `--patience` | Early stopping patience | 15 |

### Batch Inference

```bash
python model/batch_inference.py --threshold 0.9
python model/batch_inference.py --threshold 0.9 --force          # overwrite
python model/batch_inference.py --threshold 0.9 --include-annotated --force  # include R1-checked
```

### Pipeline

```
Rule-based fingering (from FurElise hand meshes)
        |
        v
  PiAnnotate Web Tool --> R1 human-edited fingering
        |                         |
        |                         v
        |                   Probe training
        |                         |
        |                         v
        +<-- R2/R3 review <-- Probe output
```

## Held-Out Evaluation (Paper Results)

The `experiments/` directory contains all scripts to reproduce the paper's held-out evaluation:

```bash
# Train 5 seeds on 91 non-R2 pieces, evaluate on 62 R2 held-out
python experiments/holdout_train.py --seed 0 --no-rule-embed
python experiments/holdout_infer.py --checkpoint model/checkpoints_holdout_norule_seed0/best.pt --output-dir data/fingering_edited_ai_holdout_norule_seed0
PIANNOTATE_AI_DIR=data/fingering_edited_ai_holdout_norule_seed0 python experiments/analyze.py --split r2

# Aggregate across seeds
python experiments/aggregate_seeds.py
```

## Citation

If you use PiAnnotate in your research, please cite:

```bibtex
@inproceedings{piannotate2026,
  title     = {PiAnnotate: A Web Annotation Tool and {ML}-Ready Corpus
               for Piano Fingering, with a Diagnostic Probe},
  author    = {Anonymous},
  booktitle = {Proc. of the 1st Korean Society for Music Informatics Conf.},
  year      = {2026}
}
```

The underlying performance data is from:

```bibtex
@inproceedings{wang2024furelise,
  author    = {Wang, Ruocheng and Xu, Pei and Shi, Haochen
               and Schumann, Elizabeth and Liu, C. Karen},
  title     = {{F\"urElise}: Capturing and Physically Synthesizing
               Hand Motions of Piano Performance},
  booktitle = {SIGGRAPH Asia 2024 Conference Papers},
  year      = {2024}
}
```

## License

- Annotation tool, probe code, analysis scripts: MIT
- Data (fingering tracks, status JSONs, etc.) are generated by the tool at runtime and not included in this repo
- FurElise recordings are not redistributed; download from [Hugging Face](https://huggingface.co/datasets/rcwang/for_elise)
