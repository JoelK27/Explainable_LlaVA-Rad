# Explainable LLaVA-Rad: Bayesian Evaluation Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Supported-orange)
![PyMC](https://img.shields.io/badge/PyMC-Bayesian_Inference-lightgrey)
![Status](https://img.shields.io/badge/Status-Research_Prototype-success)

This repository contains the official code and evaluation pipeline for the Bachelor's thesis evaluating the clinical explainability and hallucination rates of **LLaVA-Rad** (introduced in *"A clinically accessible small multimodal radiology model and evaluation metric for chest X-ray findings"*, Nature Communications 2025). 

While the official LLaVA-Rad model generates fluent radiological findings from frontal chest X-rays, this thesis introduces a rigorous post-hoc evaluation framework. Instead of relying on basic n-gram metrics (like BLEU/ROUGE), this pipeline utilizes **RadGraph** to extract clinical entities and relations, applying a **Bayesian Latent Variable Model (PyMC)** to calculate a structurally grounded Explainability Score ($E_i$).

## 🗂️ Repository Structure

```text
.
├── code/
│   ├── LLaVA-Rad/             # Official LLaVA-Rad codebase (cloned during setup)
│   └── scripts/               # Data preparation and subset building scripts
├── data/
│   ├── lists/                 # Filtered image filenames (e.g., 1500 subset)
│   ├── images/                # Image directory (downloaded via PhysioNet)
│   ├── queries/               # JSON configuration for inference queries
│   └── refs/                  # LlaVA-Rad MIMIC-CXR Annotations 
├── notebooks_v2/              # Core evaluation pipeline (Jupyter Notebooks)
│   ├── 01_validate_pilot.ipynb
│   ├── 02_align_subset_to_inference.ipynb
│   ├── 03_run_radgraph_extraction_from_aligned_input.ipynb
│   ├── 04_analyze_radgraph_results.ipynb
│   ├── 05_pymc_explainability_model_A.ipynb
│   ├── 05.1_create_template_for_manual_review.ipynb
│   ├── 05.2_create_hallucination-subset.ipynb
│   └── 05.3_pymc_explainability_final.ipynb
├── outputs/                   # Processed Bayesian models, CSVs, and diagnostics
└── README.md
```

> **Note on external data:** The images, references, and large output directories are ignored via `.gitignore`. The underlying LLaVA-Rad repository and weights are fetched independently to avoid Git submodule conflicts.

---

## ⚙️ Setup & Installation

Due to the heavy computational requirements of LLaVA-Rad, a dedicated GPU environment (e.g., **RunPod** with RTX 3090 / 4090 / A100 / V100) and **CUDA 11.x+** is required. 

### 1. Clone this Repository
```bash
git clone https://github.com/JoelK27/Explainable_LlaVA-Rad.git
cd Explainable_LlaVA-Rad
```

### 2. Setup the Python Environment
We use Python 3.10 as required by LLaVA-Rad. Conda is highly recommended.
```bash
conda create -n llavarad python=3.10 -y
conda activate llavarad
pip install --upgrade pip
```

### 3. Install LLaVA-Rad
Clone the official LLaVA-Rad repository directly into the `code` directory and install its dependencies:
```bash
cd code
git clone https://github.com/microsoft/LLaVA-Rad.git
cd LLaVA-Rad
pip install -e .

# Install additional performance dependencies:
pip install ninja
pip install flash-attn --no-build-isolation
cd ../..
```

### 4. Install Bayesian Evaluation Stack
Install the required packages for the thesis evaluation pipeline:
```bash
pip install jupyterlab pandas numpy scikit-learn pymc pytensor transformers torch networkx
# Or use: pip install -r requirements.txt (if available in the root)
```

---

## 📥 Data & Model Weights Preparation

Before running inference, you must acquire the necessary clinical datasets and model weights. **Note: MIMIC-CXR requires a signed data use agreement on PhysioNet.**

1. **Images:** Download the [MIMIC-CXR-JPG images](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) and place them in `data/images/`.
2. **Text Annotations:** Download the [LLaVA-Rad MIMIC-CXR Annotations](https://physionet.org/content/llava-rad-mimic-cxr-annotation/1.0.0/) which include reports with extracted sections in LLaVA format and place them in `data/refs/`.
3. **Model Weights:** Download the pretrained model weights for BiomedCLIP-CXR and LLaVA-Rad from HuggingFace at [microsoft/llava-rad](https://huggingface.co/microsoft/llava-rad).

---

## 🚀 Execution Pipeline

This thesis treats LLaVA-Rad as a fixed generative backbone. Therefore, we **skip model training/fine-tuning** and proceed directly to data preparation, inference, and evaluation.

### Stage 1: Data Preparation & Subset Generation
The full MIMIC-CXR image dataset and LLaVA-Rad reference annotations are massive. To avoid downloading hundreds of gigabytes of data and to allow efficient evaluation on manageable subsets, the pipeline provides scripts to slice the reference data and identify only the specifically needed image files.

#### 1a. Build the JSON Subset
First, extract a specific baseline subset (e.g., 1500 cases) from the massive raw LLaVA-Rad PhysioNet annotation file.
```bash
# From the project root path
python code/scripts/build_json_subset.py
```

#### 1b. Identify Required Images
Next, scan the generated JSON subset to extract a text list of exactly which chest X-ray images are required for inference. You can then use this list to selectively download only those images from PhysioNet.
```bash
python code/scripts/build_needed_images_from_query.py \
    --query data/queries/subset_1500.json \
    --out data/lists/needed_images_1500.txt
```

#### 1c. Validate and Prepare Final Subset
Once the images are downloaded, use `prepare_subset.py` to ensure all referenced images exist on the disk. This script will filter out any missing images and generate the final randomized or ordered subset (e.g., for sizes like 20, 300, 600, or 1500) that guarantees a crash-free inference.
```bash
python code/scripts/prepare_subset.py \
    --input_jsonl data/queries/subset_1500.json \
    --image_folder data/images/mimic-cxr-jpg/ \
    --output_jsonl data/queries/final_inference_subset.jsonl \
    --n 1500 \
    --mode random
```

### Stage 2: Model Inference (LLaVA-Rad)
Execute the generative model to produce raw clinical findings. This uses the official `eval.sh` script from the cloned LLaVA-Rad repository.

<details>
<summary><b>Click here to see the required modifications to <code>scripts/eval.sh</code></b></summary>

To replicate the inference step for this thesis, the official `eval.sh` script needs to be pointed to your specific query file and image folder. The core execution logic remains untouched.

Update `code/LLaVA-Rad/scripts/eval.sh` as follows (uncomment and adjust paths):

```bash
#!/bin/bash
set -e
set -o pipefail

model_base=lmsys/vicuna-7b-v1.5
model_path=microsoft/llava-rad
run_name=llavarad_300_run

# --- MODIFIED PATHS FOR THESIS EVALUATION ---
query_file=/workspace/thesis/data/queries/<your_chosen_subset>.json
image_folder=/workspace/thesis/data/images/mimic-cxr-jpg/2.1.0/files/mimic
prediction_dir=/workspace/thesis/outputs_raw/main
prediction_prefix=${prediction_dir}/<your_chosen_subset>_run
# --------------------------------------------

loader="mimic_test_findings"
conv_mode="v1"

mkdir -p "${prediction_dir}"

CHUNKS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

for (( idx=0; idx<CHUNKS; idx++ ))
do
    CUDA_VISIBLE_DEVICES=$idx python -m llava.eval.model_mimic_cxr \
        --query_file "${query_file}" \
        --loader "${loader}" \
        --image_folder "${image_folder}" \
        --conv_mode "${conv_mode}" \
        --prediction_file "${prediction_prefix}_${idx}.jsonl" \
        --temperature 0 \
        --model_path "${model_path}" \
        --model_base "${model_base}" \
        --chunk_idx "${idx}" \
        --num_chunks "${CHUNKS}" \
        --batch_size 8 \
        --group_by_length &
done

wait

cat "${prediction_prefix}"_*.jsonl > "${prediction_prefix}.jsonl"
```
</details>

Once the script is updated, run it:
```bash
cd code/LLaVA-Rad
bash scripts/eval.sh
cd ../..
```

### Stage 3: Bayesian Evaluation Pipeline
Once the raw inference file is generated, the evaluation is executed sequentially via the Jupyter Notebooks in `notebooks_v2/`.

| Notebook | Description |
|----------|-------------|
| `01_validate_pilot` | Initial validation of the extracted queries and data schemas. |
| `02_align_subset_to_inference` | Aligns study IDs, drops empty text outputs, and handles data conservation. Exports matched and unmatched tracking artifacts. |
| `03_run_radgraph_extraction...`| Transforms the paired texts (Reference vs. Generated) into structured clinical graphs. |
| `04_analyze_radgraph_results` | Computes precision, recall, and mismatch rates between the generated and reference reports at the observation level. |
| `05.*_pymc_explainability...` | **The core contribution.** Runs the PyMC MCMC sampler to derive the latent explainability score ($E_i$) and conducts downstream regressions against hallucination burden. |

---

## 📊 Results & Diagnostics

All final evaluation outputs are automatically saved to the `outputs/` directory when running the PyMC notebooks. This includes:
* **Highest-Density Intervals (HDI)**
* **MCMC Trace Plots**
* **Divergence Checks:** Verified zero-divergence runs.
* **Convergence Metrics:** Maximum $\hat{R}$ metrics ensuring model convergence ($\hat{R} \leq 1.02$).
* **Posterior Predictive Checks (PPC):** Verifying the model's generative adequacy against observed mismatch structures.

---

## 📜 License and Usage Notices

The data, code, and model checkpoints are licensed and intended for **research use only**. They should not be used in direct clinical care or for any clinical decision-making purpose. The LLaVA-Rad code and model checkpoints are subject to additional restrictions as determined by the Terms of Use of LLaMA, Vicuna, and GPT-4 respectively.

## 🎓 Citation & Acknowledgements

If you use this evaluation pipeline or the underlying LLaVA-Rad model, please cite the original manuscript:

```bibtex
@Article{ZambranoChaves2025,
    author={Zambrano Chaves, Juan Manuel and Huang, Shih-Cheng and Xu, Yanbo and others},
    title={A clinically accessible small multimodal radiology model and evaluation metric for chest X-ray findings},
    journal={Nature Communications},
    year={2025},
    month={Apr},
    day={01},
    volume={16},
    number={1},
    pages={3108},
    doi={10.1038/s41467-025-58344-x},
    url={https://doi.org/10.1038/s41467-025-58344-x}
}
```

The LLaVA-Rad codebase heavily relies on [LLaVA v1.5](https://github.com/haotian-liu/LLaVA). Please consider citing them as well:

```bibtex
@misc{liu2023improvedllava,
      title={Improved Baselines with Visual Instruction Tuning}, 
      author={Liu, Haotian and Li, Chunyuan and Li, Yuheng and Lee, Yong Jae},
      publisher={arXiv:2310.03744},
      year={2023},
}
```