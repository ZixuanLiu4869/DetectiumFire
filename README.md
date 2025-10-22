# DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding

**DetectiumFire** is a large-scale, multi-modal dataset designed to advance fire understanding in both traditional computer vision and modern vision-language tasks. 
It provides high-quality real and synthetic fire data, detailed annotations, and human preference feedback for training and evaluating object detectors, diffusion models, and vision-language models (VLMs). 

➡️ Dataset: https://www.kaggle.com/datasets/38b79c344bdfc55d1eed3d22fbaa9c31fad45e27edbbe9e3c529d6e5c4f93890.

➡️ Associated models (YOLO families, Stable Diffusion, etc.): https://www.kaggle.com/models/yimengfuyao/detectiumfire-models.

➡️ OpenReview: https://openreview.net/forum?id=vhHYTjMt9Z

➡️ Detectium Startup page: https://detectium.io

This repository contains:

- Code for processing the dataset for training

- Information regarding training the related models

- Metadata for the dataset


We will iteratively update our dataset and plan to introduce **DetectiumFire-Plus**, which will include more recent fire-related images. We welcome community feedback to grow DetectiumFire in both scale and impact!

---

## 📁 Repository Structure

### `image/`

```python
image/
├── image_caption_gemini.py    # Code to caption images using LLMs
└── fire_prompts.json  # Metadata and fire descriptions

```
🧾image_caption_gemini.py

Code to caption images using LLMs (e.g., GPT-4o, Gemini).
Please replace lines 16–20 with your API key and your LLM query code.



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
 

### `video/`


```python
video/
├── caption_video_gemini.py   # Code to caption our video using LLMs
├── cut_to_10s.py    # Code to cut the original videos into 10 second clips
└── generate_train_val_test_split.py  # Code to generate train/val/test dataset

```

🧾 caption_video_gemini.py

Code for captioning videos using LLMs.
The results will be released in a future version.


🧾 cut_to_10s.py

Cuts original fire videos into 10-second clips for training with TimeSformer or VideoMamba.
Edit the paths and run:

```python

# In cut_to_10s.py
# line 58:
source_folder = "YOUR_VIDEO_PATH"
# line 59:
output_folder = "YOUR_OUTPUT_FOLDER"

python cut_to_10s.py

```


🧾 generate_train_val_test_split.py


Generates `train.csv`, `val.csv`, `test.csv` in the formats required by TimeSformer and VideoMamba.
Edit the directories and run:
```python
# In generate_train_val_test_split.py
# line 6:
fire_dir = "PATH_TO_FIRE_VIDEOS"
# line 7:
non_fire_dir = "PATH_TO_NON_FIRE_VIDEOS"

python generate_train_val_test_split.py

```

---


## 🏋️ Training Notes

### TimeSformer


Official implementation: https://github.com/facebookresearch/TimeSformer

After cloning their repo, place `train.csv`, `val.csv`, and `test.csv` in the TimeSformer directory, then run:


```python

python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR . \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8

```

**Common issue**:
You may see an import error like `cannot import name '_LinearWithBias' from 'torch.nn.modules.linear'`.

**Workaround**: comment out the line


```python
from torch.nn.modules.linear import _LinearWithBias

```

in the offending file.


### VideoMamba

Official implementation: https://github.com/OpenGVLab/VideoMamba

After cloning their repo, place `train.csv`, `val.csv`, and `test.csv` in the TimeSformer directory, then run:

```python
bash ./exp/k400/videomamba_middle_mask/run_f8x224.sh

```

Edit `./exp/k400/videomamba_middle_mask/run_f8x224.sh`:
- `PREFIX='YOUR_PATH_TO_VIDEOMAMBA'`
- `DATA_PATH='YOUR_PATH_TO_VIDEOMAMBA'`
- Adjust `GPUS`, `GPUS_PER_NODE` to match your hardware.

**Common issue**:
TypeError: Mamba.__init__() got an unexpected keyword argument 'bimamba'.

**Workaround**: follow the solution described here:
https://blog.csdn.net/qq_15557299/article/details/136973682

Locate where `mamba_ssm` is installed and replace it with the official `mamba_ssm` code from the VideoMamba repo.

---

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
