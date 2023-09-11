<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_emotion_fer_plus/main/icon/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_emotion_fer_plus</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_emotion_fer_plus">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_emotion_fer_plus">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_emotion_fer_plus/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_emotion_fer_plus.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Facial expression recognition algorithm.

![example](https://raw.githubusercontent.com/Ikomia-hub/infer_emotion_fer_plus/feat/new_readme/icon/result.jpg)

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

[Change the sample image URL to fit algorithm purpose]

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add face detection algorithm
face_detector = wf.add_task(name="infer_face_detection_kornia", auto_connect=True)

# Add emotion recognition algorithm
algo = wf.add_task(name="infer_emotion_fer_plus", auto_connect=True)

# Run on your image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_emotion_fer_plus/feat/new_readme/icon/example_face.jpg")

# Get graphics
graphics = algo.get_output(1)

# Display results
display(algo.get_output(0).get_image_with_graphics(graphics))
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).


## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add face detection algorithm
face_detector = wf.add_task(name="infer_face_detection_kornia", auto_connect=True)

# Add emotion recognition algorithm
algo = wf.add_task(name="infer_emotion_fer_plus", auto_connect=True)

# Run on your image
wf.run_on(url="https://raw.githubusercontent.com/Ikomia-hub/infer_emotion_fer_plus/feat/new_readme/icon/example_face.jpg")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```
