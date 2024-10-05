import os
import time
import pickle
import random
import csv
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

import tqdm
import wfdb

warnings.filterwarnings('ignore')
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Seed initialization
def initialize_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def initialize_log_directory():
    if not os.path.exists(os.path.join(os.getcwd(), 'logs')):
        os.mkdir(os.path.join(os.getcwd(), 'logs'))


def init_dir(save_model_path):
    if not os.path.exists(save_model_path):
        os.mkdir(save_model_path)
    if not os.path.exists(os.path.join(save_model_path, 'checkpoints')):
        os.mkdir(os.path.join(save_model_path, 'checkpoints'))


def initialization(seed=0):
    initialize_seed(seed)
    initialize_log_directory()


def parse_tfrecord(example_proto):
    feature_description = {
        'waveform': tf.io.FixedLenFeature([320], tf.float32),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def load_tfrecord(file_pattern, batch_size, training=False):
    dataset = tf.data.TFRecordDataset(file_pattern)
    dataset = dataset.map(parse_tfrecord)
    if training:
        dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


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


def BaseBlockFunctional(x, d_model):
    dim = d_model // 4

    # Bottleneck
    bottleneck = layers.Conv1D(dim, kernel_size=1, strides=1, use_bias=True)(x)

    # Max pooling path
    x1 = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(x)
    x1 = layers.Conv1D(dim, kernel_size=1, strides=1, use_bias=True)(x1)

    # Convolutional paths
    x2 = layers.Conv1D(dim, kernel_size=3, strides=1, padding='same', use_bias=True)(bottleneck)
    x3 = layers.Conv1D(dim, kernel_size=5, strides=1, padding='same', use_bias=True)(bottleneck)
    x4 = layers.Conv1D(dim, kernel_size=7, strides=1, padding='same', use_bias=True)(bottleneck)

    # Concatenate
    x_out = layers.Concatenate()([x1, x2, x3, x4])

    # Batch Normalization and ReLU
    # x_out = layers.BatchNormalization()(x_out)
    x_out = layers.ReLU()(x_out)

    # # Pooling
    # x_out = layers.MaxPooling1D(pool_size=2)(x_out)

    return x_out


def MiniInceptionTimeFunctional(input_shape, in_channel=1, d_model=64, num_classes=10):
    inputs = tf.keras.Input(shape=input_shape)

    # Projection layer
    x = layers.Conv1D(d_model, kernel_size=7, strides=1, padding='same', use_bias=True)(inputs)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2, 2)(x)

    # Shortcuts
    shortcut1 = x

    # BaseBlock 1 and 2
    x = BaseBlockFunctional(x, d_model)
    x = BaseBlockFunctional(x, d_model)

    # First addition with shortcut1
    x = layers.Add()([x, shortcut1])

    # Projection layer
    x = layers.Conv1D(2 * d_model, kernel_size=5, strides=1, padding='same', use_bias=True)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2, 2)(x)

    # Shortcuts
    shortcut2 = x

    # BaseBlock 3 and 4
    x = BaseBlockFunctional(x, 2 * d_model)
    x = BaseBlockFunctional(x, 2 * d_model)

    # Second addition with shortcut2
    x = layers.Add()([x, shortcut2])

    # Projection layer
    x = layers.Conv1D(4 * d_model, kernel_size=3, strides=1, padding='same', use_bias=True)(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(2, 2)(x)

    # Shortcuts
    shortcut3 = x

    # BaseBlock 5 and 6
    x = BaseBlockFunctional(x, 4 * d_model)
    x = BaseBlockFunctional(x, 4 * d_model)

    # Second addition with shortcut2
    x = layers.Add()([x, shortcut3])

    # Global Average Pooling and Fully Connected Layer
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model


def train_one_epoch(data_loader, model, criterion, optimizer, epoch, num_batches):
    training_loss = 0
    training_acc = 0

    # Wrap the data_loader with tqdm, providing the total number of batches
    for data in tqdm.tqdm(data_loader, total=num_batches, desc=f'Training Epoch {epoch + 1}'):
        waveforms, labels = data['waveform'], data['label']

        with tf.GradientTape() as tape:
            predictions = model(waveforms, training=True)
            loss = criterion(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        training_loss += loss
        training_acc += accuracy_score(labels, np.argmax(predictions, axis=1))

    return training_loss / num_batches, training_acc / num_batches


def evaluate(data_loader, model, criterion, epoch, num_batches):
    validation_loss = 0
    validation_acc = 0
    all_preds = []
    all_labels = []

    # Wrap the data_loader with tqdm, providing the total number of batches
    for data in tqdm.tqdm(data_loader, total=num_batches, desc=f'Validating Epoch {epoch + 1}'):
        waveforms, labels = data['waveform'], data['label']

        predictions = model(waveforms, training=False)
        loss = criterion(labels, predictions)

        validation_loss += loss
        validation_acc += accuracy_score(labels, np.argmax(predictions, axis=1))
        all_preds.extend(np.argmax(predictions, axis=1))
        all_labels.extend(labels)

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
    return validation_loss / num_batches, validation_acc / num_batches, metrics


# Assuming you have access to the dataset size
def train(params, save_model_path, device='cpu'):
    print(f'Using device: {device}')

    train_loader = load_tfrecord('train_test_cp/tfrecords/train_1.tfrecord', batch_size=20, training=True)
    val_loader = load_tfrecord('train_test_cp/tfrecords/val_1.tfrecord', batch_size=1024)

    # Calculate number of batches for tqdm
    # train_num_batches = tf.data.experimental.cardinality(train_loader).numpy()
    # val_num_batches = tf.data.experimental.cardinality(val_loader).numpy()
    train_num_batches = 70042 // 20
    val_num_batches = 10510 // 1024

    input_shape = (params['in_length'], params['in_channels'])
    model = MiniInceptionTimeFunctional(input_shape=(320, 1), d_model=8, num_classes=5)
    model.build((None, 320, 1))
    model.summary()

    optimizer = optimizers.Adam(learning_rate=params['lr'])
    # criterion = losses.CategoricalCrossentropy(from_logits=True)
    criterion = losses.SparseCategoricalCrossentropy(from_logits=True)
    # criterion = sparse_focal_loss()


    num_epochs = params['epochs']
    best_loss = float('inf')
    best_score = 0
    train_loss_log = []
    validation_loss_log = []

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        # if (epoch + 1) >= 40:
        #     optimizer.learning_rate.assign(0.00001)
        print(f"Learning rate changed to {optimizer.learning_rate.numpy()}")

        train_loss, train_acc = train_one_epoch(train_loader, model, criterion, optimizer, epoch, train_num_batches)
        val_loss, val_acc, metrics = evaluate(val_loader, model, criterion, epoch, val_num_batches)

        print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}')
        print('ACC: {:.4f} | SEN: {:.4f} | SPEC: {:.4f} | PPV: {:.4f} | F1: {:.4f}'.format(
            metrics['acc'],
            metrics['sen'],
            metrics['spec'],
            metrics['ppv'],
            metrics['f1']
        ))

        train_loss_log.append(train_loss)
        validation_loss_log.append(val_loss)

        # Save model checkpoint if validation loss decreases
        if val_loss < best_loss:
            best_loss = val_loss
            model.save(os.path.join(save_model_path, 'checkpoints/best_loss_checkpoint.h5'))

        # Save model checkpoint if F1 score improves
        if metrics['f1'] > best_score:
            best_score = metrics['f1']
            model.save(os.path.join(save_model_path, 'checkpoints/best_metrics_checkpoint.h5'))

    loss_history = {'train': train_loss_log, 'validation': validation_loss_log}
    with open(f'{save_model_path}/history.pkl', 'wb') as f:
        pickle.dump(loss_history, f)



def test(params, save_model_path, mode='metric', device='cpu'):
    print(f'Using device: {device}')

    # _, _, test_loader = get_loaders(params)
    test_loader = load_tfrecord('train_test_cp/tfrecords/test_1.tfrecord', batch_size=1024)
    num_batches = 10510 // 1024

    if mode == 'metric':
        model = models.load_model(os.path.join(save_model_path, 'checkpoints/best_metrics_checkpoint.h5'))
    else:
        model = models.load_model(os.path.join(save_model_path, 'checkpoints/best_loss_checkpoint.h5'))

    all_preds = []
    all_labels = []

    for data in tqdm.tqdm(test_loader, total=num_batches, desc=f'Testing'):
        waveforms, labels = data['waveform'], data['label']
        # labels = to_categorical(labels, num_classes=5)
        predictions = model(waveforms, training=False)
        all_preds.extend(np.argmax(predictions, axis=1))
        all_labels.extend(labels)

    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))

    with open(f'{save_model_path}/test_{mode}_checkpoint_metrics.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'F1-score'])
        writer.writerow([metrics['acc'], metrics['sen'], metrics['spec'], metrics['ppv'], metrics['f1']])

    disp = ConfusionMatrixDisplay(confusion_matrix=metrics['cm'],
                                  display_labels=['Normal', 'LBBB', 'RBBB', 'PVC', 'APB'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f'{save_model_path}/test_{mode}_cm.png')


if __name__ == '__main__':
    params = {
        "in_length": 1,
        "in_channels": 1,
        "first_width": 4,
        "num_classes": 5,
        "batch_size_train": 20,
        "batch_size_val": 1024,
        "batch_size_test": 1024,
        "epochs": 100,
        "lr": 0.0001,
        "pre_train_model": "",
        "train_labels_csv": "/usr/diem/Documents/quoc.tn/ECG_Beat_Classification/train_test_cp/train_data.csv",
        "test_labels_csv": "/usr/diem/Documents/quoc.tn/ECG_Beat_Classification/train_test_cp/test_data.csv",
        "val_labels_csv": "/usr/diem/Documents/quoc.tn/ECG_Beat_Classification/train_test_cp/val_data.csv",
        "data_dir": "/usr/diem/Documents/quoc.tn/ECG_Beat_Classification/dataset",
        "dict_label": {"N": 0, "L": 1, "R": 2, "V": 3, "A": 4},
        "fs": 100,
    }

    initialization(42)
    current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    model_name = 'MiniInceptionTime_TF'
    save_model_path = f'logs/{current_time}_{model_name}'
    init_dir(save_model_path)

    # device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
    device = 'cpu'
    train(params, save_model_path, device)
    test(params, save_model_path, 'loss', device)
    test(params, save_model_path, 'metric', device)
