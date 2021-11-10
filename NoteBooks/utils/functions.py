import pandas as pd
import os
from utils.SamplePreprocessor import preprocess
import cv2
import numpy as np
import io
import re
import unicodedata
import string
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU, Dropout
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Lambda
import sys
import matplotlib.pyplot as plt
from PIL import Image

# Functions file
#import utils.functions as functions



def plot(img, title):
    
    img = np.rot90(img)
    plt.imshow(img, cmap='gray', origin='lower')
    plt.title(title);
    
def loss(labels, logits):
    return tf.reduce_mean(
            tf.nn.ctc_loss(
                labels = labels,
                logits = logits,
                logit_length = [logits.shape[1]]*logits.shape[0],
                label_length = None,
                logits_time_major = False,
                blank_index=-1
            )
        );

def encode_labels(labels, charList):
    # Hash Table
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            charList,
            np.arange(len(charList)),
            value_dtype=tf.int32
        ),
        -1,
        name='char2id'
    )
    return table.lookup(
    tf.compat.v1.string_split(labels, delimiter=''));

def train_op(model, inputs, targets, optimizer):
    with tf.GradientTape() as tape:
        # Prédiction de notre modèle
        y_pred = model(inputs, training=True)
        # Calcule de l'erreur de notre modèle
        loss_value = tf.reduce_mean(loss(targets, y_pred))
       
    # Calculer le gradient de la fonction de perte
    grads = tape.gradient(loss_value, model.trainable_variables)
    # Descente de gradient
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    # Retourner la valeur de la fonction de perte
    return loss_value.numpy();

def decode_codes(codes, charList):
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            np.arange(len(charList)),
            charList,
            key_dtype=tf.int32
        ),
        '',
        name='id2char'
    )
    return table.lookup(codes);

def greedy_decoder(logits, charList):
    # ctc beam search decoder
    predicted_codes, _ = tf.nn.ctc_greedy_decoder(
        # shape of tensor [max_time x batch_size x num_classes] 
        tf.transpose(logits, (1, 0, 2)),
        [logits.shape[1]]*logits.shape[0]
    )
    
    # convert to int32
    codes = tf.cast(predicted_codes[0], tf.int32)
    
    # Decode the index of caracter
    text = decode_codes(codes, charList)
    
    # Convert a SparseTensor to string
    text = tf.sparse.to_dense(text).numpy().astype(str)
    
    return list(map(lambda x: ''.join(x), text));

# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.strip())
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z]", " ", w)
    w = w.rstrip().strip()
    return w

#----------------------------------------------------------

def preprocess_forms(img):
    for i in range(700):
        for j in range(2479):
            img[i,j] = 255
        
    for i in range(2820,3542):
        for j in range(2479):
            img[i,j] = 255
    return img

def print_resultat ( img,text , x, y):
    
    labelSize=cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, 1, 1)
    org = (x,y)
    _x1 = x
    _y1 = y#+int(labelSize[0][1]/2)
    _x2 = x+labelSize[0][0]
    _y2 = y-int(labelSize[0][1])
    cv2.rectangle(img,(_x1,_y1),(_x2,_y2),(0,255,0),cv2.FILLED)
    cv2.putText(img, text, org, fontFace = cv2.FONT_HERSHEY_TRIPLEX, fontScale = 1, color = (0,0,0))
    
    return img