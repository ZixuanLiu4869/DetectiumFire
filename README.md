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
├── image_caption_gemini.py    # Code to caption our image using LLMs
└── fire_prompts.json  # Metadata and fire descriptions

```
🧾image_caption_gemini.py

This file contains the code to caption the image using LLMs (e.g., GPT-4o, Gemini).
Please replace lines 16-20 with your API key and code for querying your LLMs.



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
video/
├── caption_video_gemini.py   # Code to caption our video using LLMs
├── cut_to_10s.py    # Code to cut the original videos into 10 second clips
└── generate_train_val_test_split.py  # Code to generate train/val/test dataset

```

🧾 caption_video_gemini.py

This file contain the codes for caption our video using LLMs. The results will be released in the furture version.


🧾 cut_to_10s.py

This file contains the codes to cut our original fire videos into 10 seconds clips for training using either TimeSformer or VideoMamba. Please change line 58 source_folder = "Your video path" and line 59 output_folder = "your output folder" and run the code using python cut_to_10s.py.

🧾 generate_train_val_test_split.py


This file contains the codes to generate the train.csv/val.csv/test.csv that follows the format requirements by Timesformer and VideoMamba. Please change line 6, fire_dir="Your path to fire videos directory" and line 7 non_fire_dir="Your path to non fire videos directory" and run the code using python generate_train_val_test_split.py


We train the TimeSformer model using the official implemetation from https://github.com/facebookresearch/TimeSformer. After cloning their repo, please place the train.csv/val.csv/test.csv in the TimeSformer directory and run the code

```python
python tools/run_net.py \
  --cfg configs/Kinetics/TimeSformer_divST_8x32_224.yaml \
  DATA.PATH_TO_DATA_DIR . \
  NUM_GPUS 8 \
  TRAIN.BATCH_SIZE 8 \

```

You may encounter several errors regarding cannot import name '_linearwithbias' from 'torch.nn.modules.linear', Please simply comment out from torch.nn.modules.linear import _LinearWithBias, which should fix the bugs.



We train the VideoMamba model using the official implementation from https://github.com/OpenGVLab/VideoMamba.  After cloning their repo, please place the train.csv/val.csv/test.csv in the TimeSformer directory and run the code

```python
bash ./exp/k400/videomamba_middle_mask/run_f8x224.sh
```

Please change file ./exp/k400/videomamba_middle_mask/run_f8x224.sh ,line / PREFIX='your path to VideoMamba' and line 8 DATA_PATH='your path to VideoMamba' and change GPUS, GPUS_PER_NODE based on your gpu settings. You may encounter errors regarding TypeError: Mamba.__init__() got an unexpected keyword argument 'bimamba'. Please follow the solution from https://blog.csdn.net/qq_15557299/article/details/136973682, where you should find the path where your mamba_ssm is installed and replace that one with the official code of mamba_ssm in the VideoMamba repo. This will fix the bugs. 



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
