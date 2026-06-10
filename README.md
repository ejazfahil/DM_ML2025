# Thermal Image Detection — Industrial Hotspot & Fault Detection

> Detecting hotspots and thermal faults in industrial thermal/infrared imagery, combining classical image processing with deep-learning classification.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![Status](https://img.shields.io/badge/status-early%20prototype-orange)

---

**Status:** Early-stage prototype. This repository currently contains the preprocessing core and a documented methodology. The full CNN training pipeline and a reproducible evaluation harness are planned (see [Roadmap](#roadmap)). The numbers in `docs/notes.md` are design-stage targets, not yet reproduced by code in this repo.

---

## Overview

Thermography is widely used for predictive maintenance in industrial settings — electrical panels, motors, bearings, and PV installations all reveal incipient faults as localized heat signatures ("hotspots") before they fail. This project explores an automated pipeline that:

1. **Enhances** raw thermal frames for consistent dynamic range.
2. **Segments** candidate hotspots with a fast statistical threshold.
3. **Classifies** regions as fault / no-fault with a CNN, reducing false alarms from purely threshold-based detection.

## Methodology

| Stage | Technique | Status |
|-------|-----------|--------|
| Normalization | Min–max intensity normalization to `[0, 1]` | ✅ Implemented (`src/preprocess.py`) |
| Contrast enhancement | CLAHE (Contrast Limited Adaptive Histogram Equalization) | 📋 Planned |
| Hotspot segmentation | Percentile thresholding (default: top 5% of pixel intensities) | ✅ Implemented |
| Classification | CNN with a ResNet-50 backbone | 📋 Planned |

The currently implemented preprocessing primitives are intentionally dependency-light:

```python
# src/preprocess.py
normalize(img)                 # rescale thermal intensities to [0, 1]
threshold_hotspots(img, pct=95)  # binary mask of the hottest pixels
```

## Tech Stack

- **Python 3.10+**
- **NumPy** — array math and percentile thresholding
- (Planned) **OpenCV** for CLAHE, **PyTorch** for the ResNet-50 classifier

## Dataset

Per the project notes, the intended data sources are the **FLIR ADAS thermal dataset** plus **custom industrial thermal samples**. Raw imagery is not committed to this repository.

## Project Structure

```
DM_ML2025/
├── docs/
│   └── notes.md          # methodology notes & design-stage target metrics
├── src/
│   └── preprocess.py     # normalization + percentile hotspot segmentation
└── README.md
```

## Roadmap

- [ ] Add CLAHE enhancement step
- [ ] Build a labeled hotspot/fault dataset loader (FLIR ADAS + custom samples)
- [ ] Implement and train the ResNet-50 classifier
- [ ] Add an evaluation harness reporting precision / recall / F1 with held-out test data
- [ ] Commit reproducible benchmark results and confusion matrices

## Getting Started

```bash
git clone https://github.com/ejazfahil/DM_ML2025.git
cd DM_ML2025

python -c "import numpy as np; from src.preprocess import normalize, threshold_hotspots; \
img = np.random.rand(64, 64); mask = threshold_hotspots(normalize(img), pct=95); \
print('hotspot pixels:', int(mask.sum()))"
```

## Conclusion

This is a focused proof of concept for thermal fault detection. The preprocessing foundation is in place; the next milestone is closing the loop with a trained classifier and an honest, reproducible evaluation.
