import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import models

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,\
      recall_score, f1_score, ConfusionMatrixDisplay

import tqdm
import wfdb


class ECGDataset(tf.keras.utils.Sequence):
    def __init__(self, waveform_dir, dataset, batch_size):
        self.waveform_dir = waveform_dir
        self.dataset = dataset
        self.dict_label = {"N": 0, "L": 1, "R": 2, "V": 3, "A": 4}
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        batch_data = self.dataset[idx * self.batch_size:(idx + 1) * self.batch_size]
        waveforms, labels = [], []

        for _, row in batch_data.iterrows():
            raw_signal, _ = wfdb.rdsamp(self.waveform_dir + '/' + str(row['filename_lr']))
            channel_idx = int(row['channel'])
            start_idx = row['start']
            end_idx = row['end']
            raw_signal = raw_signal[start_idx:end_idx, channel_idx]

            if len(raw_signal) < 320:
                raw_signal = np.pad(raw_signal, (0, 320 - len(raw_signal)), 'constant')
            else:
                raw_signal = raw_signal[:320]

            waveform = np.nan_to_num(raw_signal)
            waveforms.append(waveform)

            label = row['Label'][0]
            target = self.dict_label[label]
            labels.append(target)

        waveforms = np.array(waveforms).reshape(len(batch_data), 320, 1)

        return np.array(waveforms), np.array(labels)


def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Specificity calculation
    specificity = []
    for i in range(len(cm)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity.append(tn / (tn + fp))
    specificity = sum(specificity) / len(specificity)

    metrics = {
        'acc': accuracy,
        'sen': recall,
        'spec': specificity,
        'ppv': precision,
        'f1': f1,
        'cm': cm,
    }

    return metrics


if __name__ == '__main__':
    # model = MiniInceptionTimeFunctional(input_shape=(320, 1), d_model=32, num_classes=5)
    # model.build((None, 320, 1))
    # model.summary()

    # for layer in model.layers:
    #     if layer.non_trainable_weights:
    #         print(f"Layer: {layer.name}")
    #         for weight in layer.non_trainable_weights:
    #             print(f"    Non-trainable weight: {weight.name}, shape: {weight.shape}")


    # configuration
    params = {
        "batch_size_test": 256,
        "test_labels_csv": "train_test_cp/test_data.csv",
        "data_dir": "dataset",
    }

    # get data
    df = pd.read_csv(params['test_labels_csv'])
    dataset = ECGDataset(params['data_dir'], df, params['batch_size_test'])
    num_batches = len(df) // params['batch_size_test']

    # load pretrained model
    checkpoint_path = 'logs/best_loss_checkpoint.h5'
    model = models.load_model(checkpoint_path)
    model.summary()

    # predict
    all_preds = []
    all_labels = []
    for waveforms, labels in tqdm.tqdm(dataset, total=num_batches, desc=f'Testing'):
        predictions = model(waveforms, training=False)
        all_preds.extend(np.argmax(predictions, axis=1))
        all_labels.extend(labels)

    # evaluate
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    print('ACC: {:.4f} | SEN: {:.4f} | SPEC: {:.4f} | PPV: {:.4f} | F1: {:.4f}'.format(
        metrics['acc'],
        metrics['sen'],
        metrics['spec'],
        metrics['ppv'],
        metrics['f1']
    ))

    # plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=metrics['cm'],
                                  display_labels=['Normal', 'LBBB', 'RBBB', 'PVC', 'APB'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f'test_cm.png')
