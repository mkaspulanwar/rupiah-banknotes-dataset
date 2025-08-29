# Rupiah Banknotes Dataset

**Short description:**
A curated image dataset of Indonesian Rupiah banknotes (Rp1,000 – Rp100,000) in **JPG format**, photographed by **M. Kaspul Anwar, Muhammad Lutfan, and Andri Rahmadani**, for Machine Learning and Computer Vision research and experiments.

---

## 🔍 What is this?

This repository contains a labeled collection of **images of Indonesian banknotes** organized by denomination. All images are in **JPG format** and were photographed by our dataset team. The dataset was prepared for tasks such as image classification, object detection, recognition, and other computer vision experiments using deep learning or classical ML.

**Denominations included:** `1k`, `2k`, `5k`, `10k`, `20k`, `50k`, `100k`, and a `mix` folder for miscellaneous or combined images.

**Primary goals:**

* Provide a simple, well-structured dataset for academic research and prototyping.
* Make replication and benchmarking of currency-recognition models easier.
* Encourage responsible use and clarity about dataset provenance and license.

---

## ✅ Use cases

* **Image classification**: recognize which denomination appears in an image.
* **Object detection**: locate banknotes in complex scenes.
* **Segmentation / OCR**: crop and extract text/serial numbers (if available and permitted).
* **Transfer learning & benchmarking**: experiments with CNNs, MobileNet, ResNet, EfficientNet, etc.
* **Education**: teaching CV pipelines end-to-end from data to model to deployment.

---

## 📁 Dataset structure (recommended)

```
rupiah-banknotes-dataset/
├── README.md
├── LICENSE
├── data/
│   ├── 1k/            # JPG images of Rp1,000
│   ├── 2k/            # JPG images of Rp2,000
│   ├── 5k/
│   ├── 10k/
│   ├── 20k/
│   ├── 50k/
│   ├── 100k/
│   └── mix/           # mixed/crowd JPG images or images containing multiple notes
├── scripts/
│   ├── split_dataset.py    # create train/val/test splits
│   └── stats.py            # compute per-class counts & basic stats
└── examples/
    ├── pytorch_example.ipynb
    └── tensorflow_example.ipynb
```

**File formats & naming:**

* Images are stored in **JPG format**.
* Filenames should be unique and, optionally, encode metadata (e.g. `20k_studio_001.jpg`).
* Avoid embedding personal data or location metadata in filenames.

---

## ⚙️ How to use this dataset

### 1) Clone the repository

```bash
git clone https://github.com/<your-username>/rupiah-banknotes-dataset.git
cd rupiah-banknotes-dataset
```

> If the dataset contains large files, consider hosting the images on a release, Zenodo, or cloud storage (Google Drive, S3) and keeping the repo lightweight. Use Git LFS for versioning large binaries if you intend to push images to GitHub.

### 2) Install common Python dependencies

```bash
python -m venv venv && source venv/bin/activate    # linux/mac
# windows: python -m venv venv && venv\Scripts\activate
pip install --upgrade pip
pip install numpy pillow matplotlib torchvision tensorflow
```

### 3) Load data with PyTorch (ImageFolder)

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

data = datasets.ImageFolder('data/train', transform=transform)
dataloader = DataLoader(data, batch_size=32, shuffle=True)

# class_to_idx: {'1k':0, '2k':1, ...}
print(data.class_to_idx)
```

### 4) Load data with TensorFlow (tf.data)

```python
import tensorflow as tf

dataset = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(224,224),
    batch_size=32,
)

for images, labels in dataset.take(1):
    print(images.shape, labels.shape)
```

### 5) Creating train / val / test splits

Use `scripts/split_dataset.py` (example):

```python
# simple script idea (not full code):
# - iterate through each class folder
# - shuffle file list
# - move 70%->train, 15%->val, 15%->test
```

(See `scripts/split_dataset.py` for a ready-to-run implementation.)

---

## 🧰 Example: quick preprocessing tips

* **Resize** to a fixed size (e.g., 224×224) or keep aspect ratio and pad.
* **Normalize** using ImageNet mean/std if using pre-trained models.
* **Data augmentation**: random crop, horizontal flip (careful: flipping may alter features), brightness/contrast jitter, rotation.
* **Class imbalance**: use class weights, oversampling, or augmentation for under-represented denominations.

---

## 📊 Dataset statistics

Include a script (`scripts/stats.py`) that computes:

* number of images per class
* total image count
* average resolution

> *Note*: If you share the dataset publicly, include an up-to-date table of per-class counts here.

---

## ⚖️ License & Terms

This dataset is released under the **MIT License**. By using the dataset you agree to follow the terms in `LICENSE`.

**If you prefer a different license for datasets (e.g., CC0 / CC BY 4.0)**, replace the LICENSE file accordingly. The README includes the MIT choice as the default for code and scripts in the repo.

---

## 🧾 Citation

If you use this dataset in academic work, please cite the repository. Suggested citation:

```
@misc{rupiah-banknotes-dataset,
  author = {M. Kaspul Anwar and Muhammad Lutfan and Andri Rahmadani},
  title = {Rupiah Banknotes Dataset},
  year = {2025},
  howpublished = {\url{https://github.com/<your-username>/rupiah-banknotes-dataset}},
}
```

Replace the URL with your repository link.

---

## 🛡️ Ethics & Responsible Use

Please follow these rules when using the dataset:

1. **Legality**: Ensure your use complies with local and national laws concerning currency images. Some jurisdictions limit reproduction of banknote images — check legal constraints before redistribution or commercial use.

2. **No counterfeiting**: This dataset **must not** be used for counterfeiting, fraud, or any illegal financial activity. Models trained on this dataset should not be used to produce realistic images or printed copies of currency.

3. **Privacy**: If any photo contains identifiable people, personal possessions, or private property, obtain consent or blur faces before sharing publicly. Do not include personal data or private information.

4. **Attribution**: If you publish results using this dataset, give proper credit to this repository (see Citation section above).

5. **Sensitive uses**: Avoid deploying models in high-stakes settings (e.g., automated financial systems) without thorough testing and safeguards.

6. **Respect licenses**: If you combine this dataset with other datasets, respect their licenses.

---

## 🤝 Contributing

Contributions are welcome! Please:

* Open an issue to discuss major changes.
* Fork the repo and submit a pull request for fixes or additions.
* Add clear metadata when you contribute new images (e.g., acquisition conditions, device, date).

When contributing images, ensure you have the right to share them and they comply with the ethics above.

---

## ✉️ Contact

For questions, issues, or requests, please open an issue in this repository or contact the maintainer listed on the GitHub profile.

---

## 📌 Acknowledgements

Dataset images were photographed by **M. Kaspul Anwar, Muhammad Lutfan, and Andri Rahmadani**. Thank you to everyone who helps collect, clean, and validate the images. If you used third-party resources to host data (e.g., Zenodo, Google Drive, S3), mention them here.

---

*Last updated: 2025-08-29*
