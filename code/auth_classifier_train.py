import torch
from torchvision.transforms import transforms

import numpy as np
import matplotlib.pyplot as plt

import cnnclassifier as cnn
from yolov5model import getDevice

device = torch.device(getDevice())

# TRAINING

"""
optimizer = [
    'Adam', 'RAdam', 'SGD',
    'ASGD', 'Adagrad', 'Adadelta',
    'AdamW', 'Adamax', 'RMSProp'
]

ASGD, SGD and Adadelta are the best optimizers.
"""

train_path = '../data/train'
val_path = '../data/val'
epochs = 210
batch_size = 8
lr = 0.09
wd = 0.001
optimizer = 'ASGD'

model = cnn.CNNClassifier(
    num_classes = 4,
    batch_size = batch_size,
    img_width = 64,
    img_height = 64,
    img_norm_mean = [0.548886846995688, 0.3514836046738479, 0.3088542879257915],
    img_norm_std = [0.17404524141334393, 0.1598939667331403, 0.1512979199750062]
).to(device)

val_acc, loss = cnn.train(
    model = model,
    model_name = 'auth_model.pt',
    train_path = train_path,
    val_path = val_path,
    epochs = epochs,
    optim = optimizer,
    momentum = 0.7,
    alpha = 0.75,
    lambd = 0.0001,
    t0 = 1000000.0,
    learning_rate = lr,
    weight_decay = wd,
)

plt.plot(val_acc)
plt.xlabel('epochs')
plt.ylabel('validation accuracy')
plt.show()

plt.plot(loss)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# PREDICTING
"""
pred_path = '../data/gender/pred/131452.jpg.jpg'
transformer = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean = [0.65625078, 0.48664141, 0.40608295],
        std = [0.20471508, 0.17793475, 0.16603905]
    )
])

model = cnn.loadModel("best.pt").to(device)
pred = cnn.predict(model, pred_path, transformer, "classes.txt")

for (key, value) in pred.items():
    print(f"{key}\t: {value}")
"""
