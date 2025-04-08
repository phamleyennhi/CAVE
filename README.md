# Escaping Plato's Cave: Robust Conceptual Reasoning through Interpretable 3D Neural Object Volumes

<a href="https://phamleyennhi.github.io/">Nhi Pham</a><sup>1</sup>,
<a href="https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/people/bernt-schiele">Bernt Schiele</a><sup>1</sup>,
<a href="https://genintel.mpi-inf.mpg.de/">Adam Kortylewski</a><sup>*1, 2</sup>,
<a href="https://explainablemachines.com/members/jonas-fischer.html">Jonas Fischer</a><sup>*1</sup>

<sup>\*</sup>Equal senior advisorship

<sup>1</sup>Max Planck Institute for Informatics, Saarland Informatics Campus, Germany

<sup>2</sup>University of Freiburg, Germany

[![arXiv](https://img.shields.io/badge/arXiv-2403.16292-b31b1b.svg)](https://arxiv.org/abs/2503.13429)
[![Project Website](https://img.shields.io/badge/Website-Visit%20Here-006c66)](https://phamleyennhi.github.io/cave/)

## 📣 News

- [25-09-04] 👀 Release of [arXiv](https://arxiv.org/abs/2503.13429) paper and [project website](https://phamleyennhi.github.io/cave/). Code is available soon, stay tuned!

## Contents

- [🌐 Project Website](https://phamleyennhi.github.io/cave/)
- [📓 Abstract](#-abstract)
- [🛠️ Installation](#️-installation)
- [💾 Datasets & Checkpoints](#-datasets--checkpoints)
- [📣 Usage](#-usage)
- [📘 Citation](#-citation)

## 📓 Abstract

With the rise of neural networks, especially in high-stakes applications, these networks need two properties (i) robustness and (ii) interpretability to ensure their safety. Recent advances in classifiers with 3D volumetric object representations have demonstrated greatly enhanced robustness in out-of-distribution data. However, these 3D-aware classifiers have not been studied from the perspective of interpretability. We introduce CAVE - Concept Aware Volumes for Explanations - a new direction that unifies interpretability and robustness in image classification. We design an inherently-interpretable and robust classifier by extending existing 3D-aware classifiers with concepts extracted from their volumetric representations for classification. In an array of quantitative metrics for interpretability, we compare against different concept-based approaches across the explainable AI literature and show that CAVE discovers well-grounded concepts that are used consistently across images, while achieving superior robustness.

## 🛠️ Installation

To get started, create a virtual environment using Python 3.12+:

```bash
python3.12 -m venv cave
source cave/bin/activate
pip install -r requirements.txt
```

## 💾 Datasets & Checkpoints

### Datasets

### Checkpoints

## 📣 Usage

### CAVE : A 3D-Aware Inherently Interpretable Classifier

### LRP with Conservation for CAVE

### Evaluation

## 📘 Citation

When using this code in your project, consider citing our work as follows:

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <pre><code>@inproceedings{pham25cave,
    title     = {Escaping Plato's Cave: Robust Conceptual Reasoning through Interpretable 3D Neural Object Volumes},
    author    = {Pham, Nhi and Schiele, Bernt and Kortylewski, Adam and Fischer, Jonas},
    booktitle = {arXiv},
    year      = {2025},
}</code></pre>
  </div>
</section>

## Acknowledgements

Nhi Pham was funded by the International Max Planck Research School on Trustworthy Computing (IMPRS-TRUST) program. We thank Christopher Wewer for his support with the NOVUM codebase, insightful discussions on 3D consistency evaluation, and careful proofreading of our paper. We also thank Artur Jesslen for his help with NOVUM codebase issues.
