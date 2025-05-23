import os
import json
import torch
from torch import nn
from torchvision import models
from PIL import Image
import numpy as np

device = torch.device("cpu")

# Load class mapping
with open('classmap.json', 'r') as f:
    idx_to_class = json.load(f)

# Load model
model = models.efficientnet_b3(pretrained=False)

# Replace final FC layer
num_classes = len(idx_to_class)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

state_dict = torch.load("checkpoints/best_model.pth", map_location=device)
# Strip 'module.' prefix if present (from DataParallel)
if any(k.startswith('module.') for k in state_dict['model_state_dict'].keys()):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict['model_state_dict'].items():
        new_key = k.replace('module.', '', 1)  # remove only first 'module.'
        new_state_dict[new_key] = v
    state_dict = new_state_dict
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

bird_name_map = idx_to_class

input_shape = (1, 3, 224, 224)

img = Image.open('american-goldfinch.png')
img = img.convert("RGB")
img = img.resize((input_shape[2], input_shape[3]))

img = np.expand_dims(img, axis=0)
img = img.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
img = img.astype(np.float32) / 255.0

stats = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
mean = np.array(stats[0]).astype(np.float32)
std = np.array(stats[1]).astype(np.float32)
mean = mean.reshape(1, -1, 1, 1)
std = std.reshape(1, -1, 1, 1)
img = (img - mean) / std
img = img.astype(np.float32)

example_input = torch.tensor(img)
traced_model = torch.jit.trace(model.eval(), example_input, strict=True)
traced_model.eval()
output_data = traced_model(example_input)
o = output_data[0].softmax(dim=0)
result = torch.max(o, dim=0)
label = bird_name_map[str(result.indices.item())]
print(f"Predicted label: {label}")

model_config = {
    "input_shape": input_shape,
    "model": "resnet",
    "mean": stats[0],
    "std": stats[1],
    "files": [
    ],
    "labels": bird_name_map,
}

def export_openvino():
    import openvino as ov

    ov_model = ov.convert_model('models/onnx/model.onnx', example_input=img, input=[input_shape])

    path = "models/openvino"
    os.system(f"rm -rf {path}")

    ov.save_model(ov_model, f"{path}/model.xml")
    model_config["files"] = [
        f"model.xml",
        f"model.bin"
    ]

    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_coreml():
    import coremltools as ct

    path = "models/coreml"
    os.system(f"rm -rf {path}")
    
    model = ct.convert(
        traced_model,
        convert_to="mlprogram",
        inputs=[ct.TensorType(shape=input_shape)],
    )

    model.save(path + "/model.mlpackage")

    model_config["files"] = [
        f"model.mlpackage/Manifest.json",
        f"model.mlpackage/Data/com.apple.CoreML/weights/weight.bin",
        f"model.mlpackage/Data/com.apple.CoreML/model.mlmodel",
    ]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_onnx():
    path = "models/onnx"

    os.system(f"rm -rf {path}")
    os.system(f"mkdir -p {path}")

    torch.onnx.export(
        traced_model,
        example_input,
        f"{path}/model.onnx",
        verbose=False,
        input_names=["input"],
        opset_version=9,
    )

    model_config["files"] = [
        f"model.onnx",
    ]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

def export_ncnn():
    path = "models/ncnn"
    os.system(f"rm -rf {path}")
    os.system(f"mkdir -p {path}")
    traced_model.save(f"{path}/model.pt")
    input_shape_str = json.dumps(input_shape)
    os.system(f"pnnx {path}/model.pt 'inputshape={input_shape_str}'")


    model_config["files"] = [
        f"model.ncnn.param",
        f"model.ncnn.bin",
    ]
    with open(f"{path}/config.json", "w") as f:
        json.dump(model_config, f)

# comment/uncomment the ones you want to export.
# some exports may not work depending on the model or host operating system.
# openvino may require intel system.
# ncnn has limited model/op support.
export_onnx()
export_openvino()
export_coreml()
export_ncnn()
