# ThinCloudCorrection

Python implementation of a single-scene thin-cloud correction framework for dynamic macroalgal blooms.

---

# Overview

This repository provides the Python implementation of a single-scene thin-cloud correction method for dynamic macroalgal blooms.

The method estimates cloud-induced scattering and attenuation effects using neighboring water pixels and a MODTRAN-based lookup table (LUT), and retrieves clear-sky reflectance for thin-cloud-contaminated macroalgal pixels.

Two Python scripts are provided and should be run sequentially.

---

# Abstract

Thin clouds substantially attenuate spectral signals in satellite imagery, leading to systematic underestimation of macroalgal bloom coverage. Existing thin-cloud correction methods have critical limitations for dynamically drifting blooms: they typically require paired cloudy-clear observations or stable surface conditions, which are rarely available, or inadequately account for cloud-induced attenuation, particularly in the near-infrared band that is essential for algal quantification.

This study proposes a single-scene thin-cloud correction framework for dynamic macroalgal blooms that removes both additive scattering and multiplicative attenuation from visible to NIR bands. By exploiting local atmospheric consistency between algal patches and adjacent water, the method infers localized cloud effects from neighboring water pixels. Cloud-free and cloud-affected water pixels are first identified to estimate clear-sky and cloudy top-of-atmosphere reflectance. A linear conversion model derived from radiative transfer theory then relates these two reflectances through coefficients C<sub>T</sub> and C<sub>R</sub>, whose relationship is constrained by a MODTRAN-based lookup table (LUT). The coefficients are solved by combining the conversion model with a quadratic polynomial relationship between C<sub>T</sub> and C<sub>R</sub> established from the LUT.

Validation using GaoFen-1 and Landsat-8 imagery against near-concurrent Sentinel-2 observations over the Southern Yellow Sea demonstrates effective near-infrared restoration. Using two complementary validation strategies—macroalgal area estimation from real images and reflectance evaluation from simulated cloudy data—the proposed method achieves consistently higher accuracy than representative state-of-the-art methods. Application to golden tide events in the East China Sea further confirms its robustness and transferability.

The method provides a practical, physics-guided solution for monitoring dynamic macroalgal blooms under thin-cloud contamination.

---

# Keywords

Lookup table (LUT) • Macroalgal blooms • MODTRAN • Thin-cloud correction

---

# Repository Structure

```
ThinCloudCorrection/
├── 01_water_class.py
├── 02_thin_cloud_cor.py
├── README.md
```

---

# Example Data

Example data used in this study are publicly available on Zenodo:

**https://doi.org/10.5281/zenodo.23067084**

After downloading the dataset from Zenodo, place the downloaded **data** folder in the root directory of this repository.

The local directory should be organized as follows:

```text
ThinCloudCorrection/
├── 01_water_class.py
├── 02_thin_cloud_cor.py
├── README.md
└── data/
    ├── Uncorrected image/
    ├── Uncorrected image_xml/
    ├── Uncorrected image_hdr/
    ├── algae_mask image/
    ├── LUT/
    ├── water_class image/
    ├── corrected_TOA/
    └── corrected_TOA_to_radiance/
```

---

# Code

The workflow consists of two sequential processing steps.

## Step 1. Water turbidity classification

Run

```bash
python 01_water_class.py
```

### Purpose

This script classifies water pixels into different turbidity classes using K-means clustering based on spectral characteristics.

### Input

```
data/
└── Uncorrected image/
```

### Output

```
data/
└── water_class image/
```

The generated water turbidity classification is required by the thin-cloud correction algorithm.

---

## Step 2. Thin-cloud correction

Run

```bash
python 02_thin_cloud_cor.py
```

### Purpose

This script performs thin-cloud correction for macroalgal bloom imagery using

- water turbidity classification;
- MODTRAN lookup table;
- GF-1 metadata;
- macroalgal mask;
- ENVI header information.

The algorithm estimates cloud attenuation and cloud scattering coefficients from neighboring water pixels and reconstructs cloud-free TOA reflectance.

### Required inputs

```
data/
├── Uncorrected image/
├── Uncorrected image_xml/
├── Uncorrected image_hdr/
├── algae_mask image/
├── LUT/
└── water_class image/
```

### Outputs

```
data/
├── corrected_TOA/
└── corrected_TOA_to_radiance/
```

where

**corrected_TOA**

contains the recovered cloud-free TOA reflectance.

**corrected_TOA_to_radiance**

contains the corrected radiance used for subsequent atmospheric correction.

---

# Citation

If you use this code or the example dataset in your research, please cite:

**Paper**

F. Zhang, S. Chen, H. Su, K. Liu, C. Wang, and W. Wang.

*Single-scene thin-cloud correction for dynamic macroalgal blooms via water-referenced radiative transfer.*

(The complete citation will be updated after publication.)


---

# Contact

If you have any questions about the code or dataset, please open an Issue in this repository.
