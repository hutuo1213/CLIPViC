# CLIPViC (CLIP & PViC) for Human-Object Interaction Detection
> This is the official implementation code for CLIPViC.

## Status
Currently, we present the model's architecture and mAP. The relevant code is being cleaned up.

<img src="./assets/clip4hoi.png" align="center" height="300">

|Model|Dataset|Backbone|Default Settings|
|:-|:-:|:-:|:-:|
|Ours|HICO-DET|ResNet50+B/32|(`36.70`, `34.82`, `37.27`)|
|Ours|HICO-DET|ResNet50+B/16|(`37.55`, `35.38`, `38.20`)|
|Ours|HICO-DET|Swin-L+L/14@336|(`48.02`, `49.97`, `47.43`)|

|Model|Dataset|Backbone|Scenario 1|Scenario 2|
|:-|:-:|:-:|:-:|:-:|
|Ours|V-COCO|ResNet50+B/32|`60.5`|`66.2`|
|Ours|V-COCO|ResNet50+B/16|`61.0`|`66.7`|
|Ours|V-COCO|Swin-L+L/14@336|`63.0`|`69.4`|
