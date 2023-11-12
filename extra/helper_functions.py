import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np



def view_random_images(target_dir, class_names, pashto_dict):
    """
    view images that are randomlly selected from each folder

    Args:
    -------------
    target_dir (str): The path of main folder that contain images folder
    class_names (list): Names of classes aka names of folders that contain images
    pashto_dict: (dictionary): dictionary of pashto digits and characters

    """

    random_images = {}
    for class_name in class_names:
        dir_path = target_dir + "/" + class_name
        img_name = random.choice(os.listdir(dir_path))
        random_images[class_name] = dir_path + "/" + img_name

    count = 1
    plt.figure(figsize=(15, 10))
    for class_name, img_path in random_images.items():
        target_img = mpimg.imread(img_path)
        plt.subplot(7, 15, count)
        plt.imshow(target_img, cmap="gray")
        plt.title(pashto_dict[int(class_name)])
        plt.axis(False);
        count += 1



def plot_loss_curves(history):
    """
    Returns separate loss curves for training and validation metrics.

    Args:
    -----------
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """ 
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(loss))

    # plot loss
    plt.plot(epochs, loss, label='loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()



def load_and_prep_image(img_path, img_shape=65, scale=True):
    """
    Reads in an image from img_path, turns it into a tensor, convert to graysacle if 
    it is not already grayscale and reshapes into (65, 65, 3).

    Parameters
    ----------
    img_path (str): string path of target image
    img_shape (int): size to resize target image to, default 65
    scale (bool): whether to scale pixel values to range(0, 1), default True
    """
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img)

    if img.shape[2] > 1:
        img = tf.image.rgb_to_grayscale(img)

    img = tf.image.resize(img, [img_shape, img_shape])

    if scale:
        return img/255.
    else:
        return img
    


def pred_and_plot(model, img_path, class_names, pashto_dict):
    """
    Imports an image located at image_path, makes a prediction on it with
    a trained model and plots the image with the predicted class as the title.

    parameters
    -------------
    model: Your trained model
    img_path (str): string path of target image
    class_names (list): Names of classes aka names of folders that contain images
    pashto_dict: (dictionary): dictionary of pashto digits and characters
    """
    img = load_and_prep_image(img_path)

    pred_prob = model.predict(tf.expand_dims(img, axis=0))
    pred_index = pred_prob.argmax()
    pred_class = class_names[pred_index]
    pred_character = pashto_dict[int(pred_class)]

    print(f"Pred_character: {pred_character}")

    plt.imshow(img, cmap='gray')
    plt.title(f"pred_character: {pred_character} - prob: {pred_prob.max()}")
    plt.axis(False)



def plot_confusion_matrix(model, test_data, class_names, fig_size=100):
    """
    Makes a labelled confusion matrix comparing predictions and ground truth labels.

    Args:
    ----------
    model: Your trained model
    class_names: list of class labels.
    figsize: Size of output figure (default=(100, 100)).
    test_data: The portion of data for testing the model.
    """
    pred_probs = model.predict(test_data)
    pred_classes = pred_probs.argmax(axis=1)
    y_labels = test_data.labels

    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(confusion_matrix(y_true=y_labels, y_pred=pred_classes),
                cmap='crest',
                annot=True,
                xticklabels=class_names,
                yticklabels=class_names,
                fmt='.0f')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('y_pred', fontsize=40)
    plt.ylabel('y_true', fontsize=40);



def pred_and_plot_images(model, pashto_dict, target_dir, class_names, fig_size=(15, 27)):
  """
  Randomlly select an image from each class and makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.

  parameters:

  model: Your trained model
  target_dir (str): directory that contains images folders
  pashto_dict (dictionary): dictionary of digits
  class_names (list): Names of classes aka names of folders that contain images
  fig_size (tuple): size of the figure defualt (15, 27)
  """

  random_images = {}
  for class_name in class_names:
      dir_path = target_dir + "/" +class_name
      img_name = random.choice(os.listdir(dir_path))
      random_images[class_name] = dir_path + "/" + img_name

  plt.figure(figsize=fig_size)
  plt.subplots_adjust(hspace=2.5)
  count = 1
  for label, img_path in random_images.items():
    img = load_and_prep_image(img_path)
    pred_prob = model.predict(tf.expand_dims(img, axis=0), verbose=0)
    pred_index = pred_prob.argmax()
    pred_class = class_names[pred_index]
    pred_char = pashto_dict[int(pred_class)]
    if (pred_char == pashto_dict[int(label)]):
      color = 'g'
    else:
      color = 'r'
    plt.subplot(15, 10, count)
    plt.imshow(img, cmap='gray')
    plt.title(f"label: {pashto_dict[int(label)]}\npred: {pred_char}\nprob: {pred_prob.max():.2f}", c=color)
    plt.axis(False)
    count += 1