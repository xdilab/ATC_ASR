# ✈️ Enhancing Voice-to-Text Transcription for Air Traffic Communications Using AI

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Build](https://img.shields.io/badge/build-passing-brightgreen)

---

## 📄 Overview

Air Traffic Control (ATC) communications pose significant challenges for automatic speech recognition due to noisy radio environments, overlapping speech, accent variations, and aviation-specific jargon. This project introduces a **self-supervised error-correction framework** using advanced **sequence-to-sequence (seq2seq)** models like **BART** to enhance transcription accuracy for ATC scenarios.

The framework builds upon public voice-to-text (V2T) models, applying iterative correction mechanisms and domain-specific preprocessing to improve recognition quality, reduce word error rates, and handle ATC-specific terminology effectively.

---

## 🧪 Key Contributions

- ✅ Evaluation of multiple **pretrained V2T models** (Wav2Vec variants) on ATC datasets.
- ✅ Implementation of **BART-based sequence correction** for transcription refinement.
- ✅ Development of a **self-supervised iterative feedback loop** for continuous performance improvement.
- ✅ Integration of **noise reduction, audio normalization, and snippet processing** to handle real-world ATC conditions.

---

## 🖼️ System Architecture

```
[ Raw ATC Audio (WAV) ]
          |
  +--------------------+
  |  Preprocessing     | → Noise Reduction, Normalization, Slicing
  +--------------------+
          |
[ Voice-to-Text Model (Wav2Vec, XLS-R) ]
          |
  +---------------------------+
  |  Seq2Seq Error Correction | → BART, T5, RoBERTa, Pegasus
  +---------------------------+
          |
     [ Final Transcription ]
          ↓
    Evaluation & Analysis
```

<p align="center">
  <img src="img/framework.png" alt="Self-Supervised Framework for V2T-to-T2T" width="600">
</p>

---

## 🗄️ Repository Structure

```
CYCLE_PIPELINE/
├── datasets/                     # ATCOSIM, UWB-ATCC samples and transcriptions
├── models/                       # Pretrained models and checkpoints
├── Results/                      # Evaluation outputs and logs
├── create_bart.py                # Creates and configures the BART model
├── download_v2t_model.py         # Downloads HuggingFace V2T models
├── evaluate_bart.py              # Evaluates BART correction performance
├── evaluation_generator.py       # Generates evaluation datasets
├── WAV-to-TEXT.py                # Initial V2T transcription from WAV files
├── finetune_bart.py              # Fine-tuning BART on ATC-specific data
├── huggingface_model_downloader.py # HuggingFace model utility
├── manual_audio_snippets.py      # Manual processing for snippet extraction
├── snippet_normalization.py      # Normalizes audio segments
├── total_snippet_time.py         # Calculates total dataset audio length
├── README.md                     # You're here!
```

---

## 🎯 Datasets and Models

### 📁 Datasets:
- **atcosim_corpus** — 1,900 samples  
- **uwb_atcc** — 2,820 samples  
- **KGSO annotated frequencies** — 467 samples (~1 hr 21 min)

### 🤖 V2T Models Evaluated:
- `Common_voice_accents` (95M params)
- `Large-xlsr-53` (317M params)
- `Large-960h-uwb` (317M params)
- `Large-960h-atcosim` (317M params)
- `Xls-r-300m` (300M params)

### 🛠️ Seq2Seq Models for Correction:
- **BART** (primary focus)
- RoBERTa
- T5
- Pegasus

---

## 🛠️ Preprocessing Pipeline

- 📦 **Noise Reduction:** Stationary noise threshold = 1.5 std dev (`noisereduce`)
- 🎚️ **Volume Normalization:** `pydub`
- ✂️ **Snippet Slicing:** Configurable time windows

---

## 🧩 Experimental Setup

| Parameter               | Value              |
|-------------------------|--------------------|
| Dataset Split           | 80% Train / 20% Test |
| Training Batch Size     | 32                 |
| Evaluation Batch Size   | 64                 |
| Epochs                  | 10                 |
| Learning Rate Scheduler | Cosine with restarts |
| Early Stopping          | Based on cosine similarity |

---

## 📊 Results Snapshot

| Model               | Word Error Rate (WER) | Character Error Rate (CER) |
|---------------------|----------------------|---------------------------|
| Large-xlsr-53       | XX%                  | XX%                       |
| Large-960h-atcosim  | XX%                  | XX%                       |
| BART Correction     | **↓ Significant Reduction** | **↓ Improved Accuracy** |

> 📁 Full results and tables available in the `/Results` directory.

---

## 🚀 Quick Start

### Clone the Repository:
```bash
git clone https://github.com/yourusername/CYCLE_PIPELINE.git
cd CYCLE_PIPELINE
```

### Set up Environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Download Pretrained Models:
```bash
python download_v2t_model.py
```

### Run Initial Evaluation:
```bash
python evaluate_bart.py --config configs/eval_config.json
```

---

## 📌 Acknowledgments

This research was supported by **Boeing Company** and the **NNSA U.S. Department of Energy (TRACS – DE-NA0004189)**.

---

## 📬 Contact

For questions or collaborations, contact:  
**Everett-Alan Hood** — elhood@aggies.ncat.edu

---

## ⚖️ License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@inproceedings{hood2025v2t,
  title={Enhancing Voice-to-Text Transcription for Air Traffic Communications Using AI},
  author={Hood, Everett-Alan and LastName, SecondName},
  booktitle={Proceedings of the IEEE Conference},
  year={2025}
}
```

---

*Designed to improve transcription accuracy where it matters most — in the skies above us.* ✈️

