# C-OLR
Python implementation of a Constrained Ordinal Logistic Regression (C-OLR) framework for atmospheric corrosivity classification.


# Constrained Ordinal Logistic Regression for Atmospheric Corrosivity Classification (C-OLR)

## Abstract
This repository contains the custom Python implementation of the Constrained Ordinal Logistic Regression (C-OLR) framework developed for the classification of atmospheric corrosivity based on environmental exposure parameters.

The framework uses ordinal logistic regression with non-negative constraints on the regression coefficients to preserve physically meaningful relationships between environmental exposure conditions and corrosion severity. The repository provides the model implementation, datasets used for model development and independent evaluation, and supporting information required to reproduce the analyses reported in the associated publication (DOI: xxx).

---

## Data Sources
The dataset used in this study was compiled from the ISOCORRAG, MICAT, and other published atmospheric corrosion sources (external validation) cited in the associated publication. The Excel workbook contains separate worksheets for the training dataset, testing dataset, and data dictionary/source information.

The training dataset contains 320 observations used for model development, while the independent testing dataset contains 67 observations used for model evaluation.

The data used in this study were compiled from the ISOCORRAG and MICAT corrosion databases and from additional published atmospheric corrosion sources cited in the associated publication.

Third-party datasets and source materials remain subject to any applicable restrictions imposed by their original providers or publishers.

---

## Instructions
Install the required Python packages and run the C-OLR Python script.

The code loads the training and testing datasets, estimates the constrained ordinal logistic regression coefficients and latent thresholds, assigns predicted corrosivity categories, and calculates the model-performance metrics reported in the associated publication.

Required packages:

numpy
pandas
scipy
scikit-learn
tabulate

---

## Repository Contents
- `C_OLR.py` — C-OLR model implementation and evaluation code 
- `datasets.xlsx` — Model training dataset  
- `README.md` — Documentation and usage instructions  

---

## Methodology
The C-OLR framework represents atmospheric corrosivity as an ordinal outcome using environmental exposure parameters. Non-negative constraints are imposed on the regression coefficients, while ordered latent thresholds define the boundaries between corrosivity categories.
- The model is estimated using constrained numerical optimization and evaluated using independent testing observations.
-A detailed description of the methodology and theoretical formulation is provided in the associated publication.

---

## Versioning
- **v1.0.0** — Initial release associated with the published study  
Future updates may include improvements, recalibration, and extended functionality.

---

## Citation
If you use this tool, please cite the associated publication and archived version:

> Rockey A., Hurlebaus S., *Constrained Ordinal Learning for ISO~9223 Atmospheric Corrosivity Assessment of Steel Infrastructure*, Journal Name, Year.  
> DOI: xxx  

Zenodo Archive:
> DOI: 10.5281/zenodo.xxxxxxx

---

## License
This repository contains original code developed for the C-OLR framework. The code is distributed under the MIT License.

Third-party datasets and published source materials are not covered by this license and remain subject to the terms of their respective providers and publishers.

---

## Disclaimer
This repository is provided for research and educational purposes only. The C-OLR framework is intended to support atmospheric corrosivity research and should not be considered a substitute for engineering judgment, applicable standards, inspection, or site-specific corrosion assessment.
