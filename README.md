# Lightweight_ECG_Beat_Classification

## Environments

```pip install -r requirements.txt```

## Dataset

Please download and uncompressed in the root of this repository
- Raw Dataset: [link](https://drive.google.com/file/d/16g5NXeenswcHerJCoPme28qeWImBYCaI/view?usp=sharing)
- TFRecord: [link](https://drive.google.com/file/d/1RMUVK0ZBvk2zM-3saZBCEni2bQgfzGGb/view?usp=sharing)

## Training

Keras: run ```python main_tf.py``` to train and testing inference, the checkpoints and results are saved in folder **logs**.

## Inference
Keras: run ```python main_tf_inference.py``` for inference. The results is printed on terminal and the confusion matrix is displayed in the **test_cm.png**.