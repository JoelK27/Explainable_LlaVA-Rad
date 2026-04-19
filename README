# Explainable LLaVA-Rad: Bayesian Evaluation Pipeline

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Supported-orange)
![PyMC](https://img.shields.io/badge/PyMC-Bayesian_Inference-lightgrey)
![Status](https://img.shields.io/badge/Status-Research_Prototype-success)

This repository contains the official code and evaluation pipeline for the Bachelor's thesis evaluating the clinical explainability and hallucination rates of **LLaVA-Rad**, a multimodal large language model for chest X-ray interpretation.

Instead of relying on basic n-gram metrics (like BLEU/ROUGE), this pipeline utilizes **RadGraph** to extract clinical entities and relations, applying a **Bayesian Latent Variable Model (PyMC)** to calculate a structurally grounded Explainability Score ($E_i$).

## 🗂️ Repository Structure

```text
.
├── code/
│   └── scripts/               # Data preparation and subset building scripts
├── data/
│   ├── lists/                 # Filtered image filenames (e.g., 1500 subset)
│   └── queries/               # JSON configuration for inference queries
├── notebooks_v2/              # Core evaluation pipeline (Jupyter Notebooks)
│   ├── 01_validate_pilot.ipynb
│   ├── 02_align_subset_to_inference.ipynb
│   ├── 03_run_radgraph_extraction_from_aligned_input.ipynb
│   ├── 04_analyze_radgraph_results.ipynb
│   ├── 05.1_create_template_for_manual_review.ipynb
│   ├── 05.2_create_hallucination-subset.ipynb
│   └── 05.3_pymc_explainability_final.ipynb
├── outputs/                   # Processed Bayesian models, CSVs, and diagnostics
└── README.md
```

> **Note on external data:** The images, references, and large output directories are ignored via `.gitignore`. The **MIMIC-CXR-JPG** dataset requires credentialed access via [PhysioNet](https://physionet.org/). The underlying LLaVA-Rad repository and weights are also fetched independently to avoid Git submodule conflicts.

---

## ⚙️ Setup & Installation

Due to the heavy computational requirements of LLaVA-Rad, a dedicated GPU environment (e.g., **RunPod** with RTX 3090 / 4090 / A6000) is highly recommended.

### 1. Clone this Repository
```bash
git clone <your-repository-url>
cd <your-repository-name>
```

### 2. Setup External LLaVA-Rad Repository
This project relies on the official LLaVA-Rad architecture for the inference step. Clone it directly into the `code` directory (it is git-ignored by this repository to keep the workspace clean).
```bash
cd code
git clone https://github.com/Stanford-AIMI/LLaVA-Rad.git
cd ..
```

### 3. Environment & Dependencies
Create a virtual environment and install both the LLaVA-Rad requirements and the Bayesian evaluation stack (PyMC, PyTensor, Pandas, Jupyter, etc.):
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install requirements (make sure you have your requirements.txt updated)
pip install --upgrade pip
pip install -r requirements.txt

# Alternatively, install core components manually:
pip install jupyterlab pandas numpy scikit-learn pymc pytensor transformers torch networkx
```

---

## 🚀 Execution Pipeline

The workflow strictly enforces a modular separation of concerns. Do not proceed to the evaluation notebooks until the inference is complete.

### Stage 1: Data Preparation
Identify the necessary chest X-ray images based on the structured query subsets (e.g., 1500 cases). Ensure that the MIMIC-CXR-JPG dataset is downloaded and placed in the appropriate directory (e.g., `data/mimic-cxr-jpg/`).

### Stage 2: Model Inference (LLaVA-Rad)
Execute the generative model to produce raw clinical findings. This uses the official evaluation scripts from the cloned LLaVA-Rad repository.
```bash
# Example inference command (adapt paths according to your setup)
cd code/LLaVA-Rad
bash scripts/eval.sh --query_file ../../data/queries/subset_1500.json --image_folder ../../data/mimic-cxr-jpg/
cd ../..
```

### Stage 3: Evaluation & Bayesian Modeling
Once the raw inference file is generated, the remainder of the pipeline is executed sequentially via the provided Jupyter Notebooks in `notebooks_v2/`.

| Notebook | Description |
|----------|-------------|
| `01_validate_pilot` | Initial validation of the extracted queries and data schemas. |
| `02_align_subset_to_inference` | Aligns study IDs, drops empty text outputs, and handles data conservation. Exports matched and unmatched tracking artifacts. |
| `03_run_radgraph_extraction...`| Transforms the paired texts (Reference vs. Generated) into structured clinical graphs (entities and relations). |
| `04_analyze_radgraph_results` | Computes precision, recall, and mismatch rates between the generated and reference reports at the observation level. |
| `05.*_pymc_explainability...` | **The core contribution.** Runs the PyMC MCMC sampler to derive the latent explainability score ($E_i$) and conducts downstream regressions against hallucination burden. |

---

## 📊 Results & Diagnostics

All final outputs are automatically saved to the `outputs/` directory when running the PyMC notebooks. This includes:
* **Highest-Density Intervals (HDI)**
* **MCMC Trace Plots**
* **Divergence Checks:** Verified zero-divergence runs.
* **Convergence Metrics:** Maximum $\hat{R}$ metrics ensuring model convergence ($\hat{R} \leq 1.02$).
* **Posterior Predictive Checks (PPC):** Verifying the model's generative adequacy against observed mismatch structures.