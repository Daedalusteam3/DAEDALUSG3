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

