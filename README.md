# DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding

DetectiumFire is a large-scale, multi-modal dataset designed to advance fire understanding in both traditional computer vision and modern vision-language tasks. 
It provides high-quality real and synthetic fire data, detailed annotations, and human preference feedback for training and evaluating object detectors, diffusion models, and vision-language models (VLMs). 

The whole dataset can be found at https://www.kaggle.com/datasets/38b79c344bdfc55d1eed3d22fbaa9c31fad45e27edbbe9e3c529d6e5c4f93890.

The associated models, such as object detectors (e.g., YOLO families), diffusion models (e.g., Stable Diffusion) can be found at ?

This repo contains the codes for processing our dataset for training, information regarding training the related models and the meta data of the dataset.


We will iterative updates our dataset and estimate to introduce DetectiumFire-Plus, which will include more recent fire-related images. We welcome community feedback to grow DetectiumFire in both scale and impact!


The image folder contains

**Folder Structure**

```python
image/
├── image_caption_gemini.py    # Code to caption 
└── fire_prompts.json  # Metadata and fire descriptions

```

🧾 fire_prompts.json

This file provides detailed annotations and metadata for each fire image. Each entry in the JSON file follows this format:


```python
{
  "image": "msg5430551134-14167_jpg.rf.1fdbd3d8e3053b8b93f36844e46b8d71.jpg",
  "source": "iot_device_detectium",
  "fire_prompt": "A small flame is burning from a lighter held in a person's hand, indoors, with minor severity.",
  "fire_type": "Indoor_lighter_flame"
}
```

🔑 **Field Descriptions**:

- image: Filename of the fire image, located in real_fire/images/.

- source: The origin of the image, as categorized in Appendix C.1 of the paper. Possible values include: web_search, iot_device_detectium, FIRE, Forest Fire and FireNET.

- fire_prompt: Final edited, human-verified fire prompt used for text-to-image generation and fine-tuning diffusion models.

- fire_type: Detailed taxonomy label corresponding to the fire type, following the hierarchical categorization described in Appendix C.4.




The video folder contains



```python
image/
├── 
└── fire_prompts.json  # Metadata and fire descriptions

```



## 📚 Citation

If you find **DetectiumFire** helpful, please cite:

**Liu, Z., Khajavi, S. H., & Jiang, G. (2025).**  
*DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding.*  
NeurIPS Datasets and Benchmarks Track, 2025.  
[OpenReview](https://openreview.net/forum?id=vhHYTjMt9Z)

```bibtex
@inproceedings{liu2025detectiumfire,
  title     = {DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding},
  author    = {Zixuan Liu and Siavash H. Khajavi and Guangkai Jiang},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2025},
  url       = {https://openreview.net/forum?id=vhHYTjMt9Z}
}
```
