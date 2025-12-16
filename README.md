## 📂 Data Provenance & Training Source

The core training logic and feature extraction pipelines are encapsulated within the **`Xception-Training/`** directory.

The Jupyter notebook located in this folder relies on the **Fashion550k dataset** as the authoritative ground truth for model optimization and style attribute learning.

> **Note:** The dataset utilized in these scripts is sourced from the academic research presented at the **ICCV Workshops**.

If you use these training scripts or the underlying data logic, please acknowledge the original source:

<details>
<summary>📚 <strong>Click to view BibTeX Citation</strong></summary>

```bibtex
@InProceedings{TakagiICCVW2017,
  author    = {Moeko Takagi and Edgar Simo-Serra and Satoshi Iizuka and Hiroshi Ishikawa},
  title     = {{What Makes a Style: Experimental Analysis of Fashion Prediction}},
  booktitle = "Proceedings of the International Conference on Computer Vision Workshops (ICCVW)",
  year      = 2017,
}
</details>

## 🌐 Web Interface & Deployment (Daedalus)

The graphical user interface (GUI) and deployment logic are encapsulated within the **`Daedalus/`** directory.

This component is responsible for orchestrating the inference pipeline, allowing users to interact with the trained models via a web-based environment. It integrates the computer vision backends with a responsive frontend.

### ⚙️ Environment & Dependencies

 The `Daedalus` module requires a specific set of libraries to ensure compatibility across TensorFlow, PyTorch, and data processing utilities.

**Core Frameworks:**
* `tensorflow==2.20.0`
* `torch==2.9.1+cpu`
* `ultralytics==8.3.229` (YOLO)
* `polars==1.35.2`

For exact reproducibility, please ensure your environment matches the full dependency tree below:

<details>
<summary>📋 <strong>Click to view full <code>requirements.txt</code></strong></summary>

```text
absl-py==2.3.1
astunparse==1.6.3
certifi==2025.11.12
charset-normalizer==3.4.4
contourpy==1.3.2
cycler==0.12.1
filelock==3.19.1
flatbuffers==25.9.23
fonttools==4.61.0
fsspec==2025.9.0
gast==0.7.0
google-pasta==0.2.0
grpcio==1.76.0
h5py==3.15.1
idna==3.11
Jinja2==3.1.6
keras==3.10.0
kiwisolver==1.4.9
libclang==18.1.1
Markdown==3.10
markdown-it-py==4.0.0
MarkupSafe==3.0.3
matplotlib==3.10.7
mdurl==0.1.2
ml_dtypes==0.5.4
mpmath==1.3.0
namex==0.1.0
networkx==3.3
numpy==1.26.4
opencv-python==4.8.1.78
opt_einsum==3.4.0
optree==0.18.0
packaging==25.0
pillow==12.0.0
pip==25.3
polars==1.35.2
polars-runtime-32==1.35.2
protobuf==6.33.2
psutil==7.1.3
Pygments==2.19.2
pyparsing==3.2.5
python-dateutil==2.9.0.post0
PyYAML==6.0.3
requests==2.32.5
rich==14.2.0
scipy==1.15.1
setuptools==63.2.0
six==1.17.0
sympy==1.14.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
tensorflow==2.20.0
termcolor==3.2.0
torch==2.9.1+cpu
torchvision==0.24.1+cpu
typing_extensions==4.15.0
ultralytics==8.3.229
ultralytics-thop==2.0.18
urllib3==2.6.0
Werkzeug==3.1.4
wheel==0.45.1
wrapt==2.0.1

