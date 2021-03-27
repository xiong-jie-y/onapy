# onapy
Onapy contains, you know, masturbator motion detector and sound player based on the motion.

![](main_image.gif)

Ona comes from Japanese name of the masturbator "Onahole".
Please see this [presentation](https://www.youtube.com/watch?v=W3vWto6AU9Y&t=5s) for detailed explanation of Open Onaho Project.

## Requirements
* GPU
* Depth camera from realsense series (D435i or something)
* masturbator

## Installation

```bash
# (Optional) Create python environment for onapy.
# Install miniconda if you don't have it.
# https://docs.conda.io/en/latest/miniconda.html
conda create -n py38_onapy python=3.8.5
conda activate py38_onapy

# Please install appropriate torch and mmcv version for your pc.
# For example, for cuda11, please install like this.
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.2.7 -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.0/index.html

pip install onapy

# Please put your favorite sounds into the folder.
# These sound will be chosen randomly and played,
# when masturbator is moved (estemated as inserted).
cp *.mp3 sounds/
```

## How to run it.
### Recognize waist motion attached with Realsense T265
```bash
recognize_waist_motion  --sound-dir sounds/
```
Just move your waist.

### Recognize masturbator motion
```bash
recognize_onaho_motion  --sound-dir sounds/
```

After that, just move the masturbator.