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
$ pip install -r requirements.txt
```

### Inference
Please download the weight 
|Weight|Size|Reference|
|------|----|---------|
|[best.pt](https://drive.google.com/file/d/1itEkrhMZl-BpsShADcUnfKPo_UjVxvty/view?usp=sharing)|1.05 GB|YOLOv5|
|[final_eff.pt](https://drive.google.com/file/d/1n8zuGWBGNYVW910y-qYr4yEfHjOO7fTN/view?usp=sharing)|294.1 MB|EfficientNet|
|[paddle.zip](https://drive.google.com/file/d/1GQOiu_7izVQ4413uzMHvv5fwT2ja1Ewf/view?usp=sharing)|89.9 MB|PaddleOCR|
|[pren.pth](https://drive.google.com/file/d/1ts2eDD52ZxY930_kCUpToQw0Qsv-OTtn/view?usp=sharing)|334.1 MB|PREN|

<br>

Place the models according to the following structure. <br>

```
Project
|---README.md
|---requirements.txt
|---efficientNet_training
|   |---final_eff.pt
|---PaddleOCR
|   |---paddle.zip
|---pren
|   |---models
|   |   |---pren.pth
|---yolov5
|   |---best.pt
```

Unzip the `paddle.zip` file under PaddleOCR folder. <br>

<div style="text-align: justify">

<br> `final_test.py` runs inference on a `path/` (directory) and output the result in `*.csv`. <br>
The `final_test.py` is designed to run for the competition [繁體中文場景文字辨識競賽－高階賽：複雜街景之文字定位與辨識](https://tbrain.trendmicro.com.tw/Competitions/Details/19). <br>
Line 287: you probably need to change the output file name. <br>
The public dataset from the competition provided can be download from [HERE](https://drive.google.com/file/d/1E09fzyjJLAtciDi7fInn-CsJyhVevhUw/view?usp=sharing). <br><br>

 ---
<div style="text-align: justify">
<font size = "3">

<b>NOTE :exclamation:</b> <br>
Please run the following command under the folder `yolov5`!

```
cd yolov5
```

</font>
</div>

---

Run the following commandd:

 ```
 python final_test.py --img 1536 --weight best.pt --augment --source ../private/ --rec_model_dir="../PaddleOCR/inference/" --rec_char_dict_path ../PaddleOCR/ppocr/utils/new_small_dict_ch_en_num.txt
 ```

</div>

When finished, the result will be saved and output as `final_output.csv`.

## Dataset
The following datasets used for the training. <br>

|Dataset|Train On|
|-------|--------|
|[contest]()|YOLOv5|
|[eff_img_label.zip](https://drive.google.com/file/d/1j1FfRjADFRfsdlqZ5Lz_PJK6daeExpQf/view?usp=sharing)|EfficientNet|
|[eff_img.zip](https://drive.google.com/file/d/17pm5ygJKLXt3jXkSlhvYZbzuCQVdS7T0/view?usp=sharing)|EfficientNet|
|[crop.zip](https://drive.google.com/file/d/1aEteeYScr7zV2UCsOHA9i7-scLOgrv1N/view?usp=sharing)|PaddleOCR|
|[new_tiger_images.zip](https://drive.google.com/file/d/1L9PAxISsTc9AaBRvCiGKOs9ntdT_deJs/view?usp=sharing)|PaddleOCR|

Place the datasets according to the following structure and unzip them. <br>

```
Project
|---README.md
|---crop.zip
|---contest.zip
|---synthtiger
|   |---new_tiger_images.zip
|---requirements.txt
|---efficientNet_training
|   |---final_eff.pt
|   |---eff_img_label.zip
|   |---eff_img.zip
|---PaddleOCR
|   |---paddle.zip
|---pren
|   |---models
|   |   |---pren.pth
|---yolov5
|   |---best.pt
```

