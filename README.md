# MultiLingual-OCR

<div style="text-align: justify">

This code is used for the competition [繁體中文場景文字辨識競賽－高階賽：複雜街景之文字定位與辨識](https://tbrain.trendmicro.com.tw/Competitions/Details/19). The competition is about scene text detection and recognition in Taiwan (Traditional Chinese). This code assembled [YOLOv5](https://github.com/ultralytics/yolov5), [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR.git) and [PREN](https://github.com/RuijieJ/pren.git) in one. YOLOv5 for text detection and EfficientNet-PyTorch, PaddleOCR and PREN for recognition. The dataset for text detection included [Google Landmarks Dataset V2](https://github.com/cvdfoundation/google-landmark) as background images; for EfficientNet included [Single_char_image_generator](https://github.com/rachellin0105/Single_char_image_generator) as character generator. Both the dataset also included the data given by the organiser. The dataset for PaddleOCR training included [SynthTiger](https://github.com/clovaai/synthtiger.git) as text image generator.

</div>

---
<div style="text-align: justify">
<font size = "3">

<b>NOTE :exclamation:</b> <br>
Some of the works are modified in this code. Therefore, directly clone from the master repo will not compatible with this code, or vice versa!
</font>
</div>

---

## Qucik Start
### Installation
[**Python**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/RabbitCrab/MultiLingual-OCR/main/requirements.txt) installed including

```
$ git clone https://github.com/RabbitCrab/MultiLingual-OCR.git
$ cd Chinese-Character-Traditional-Scene-Text
$ pip install -r requirements.txt
```

### Inference
Please download the weight 
|Weight|Size|Reference|
|------|----|---------|
|[best.pt](https://drive.google.com/file/d/1itEkrhMZl-BpsShADcUnfKPo_UjVxvty/view?usp=sharing)|1.05 GB|YOLOv5
<br>
Place the EfficientNet model under the [efficientNet_training](https://github.com/RabbitCrab/MultiLingual-OCR/tree/main/efficientNet_training)

```
