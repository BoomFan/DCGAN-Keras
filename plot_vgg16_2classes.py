import os
import json
import codecs
import matplotlib.pyplot as plt


def saveHist(path, history):  # Used to save trining history
    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(history, f, separators=(',', ':'), sort_keys=True, indent=4)


def loadHist(path):  # Used to load trining history
    n = {}  # set history to empty
    if os.path.exists(path):  # reload history if it exists
        with codecs.open(path, 'r', encoding='utf-8') as f:
            n = json.loads(f.read())
    return n


def appendHist(h1, h2):
    if h1 == {}:
        return h2
    else:
        dest = {}
        for key, value in h1.items():
            dest[key] = value + h2[key]
        return dest


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_factor = 0.  # For plotting the accuracy and loss

# The path that we store our checkpoints
output_dir = '../data/output/VGG16_2classes_all_new'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_filename = output_dir + '/foodsafety_VGG16_2classes.h5'
history_filename = output_dir + '/model-history.json'

# Plot accuracy and loss
historical_history = loadHist(history_filename)  # load history from current training
print('Plot accuracy and loss ...')
# acc = vgg16_history.history['acc']
# val_acc = vgg16_history.history['val_acc']
# loss = vgg16_history.history['loss']
# val_loss = vgg16_history.history['val_loss']
acc = historical_history['acc']
acc = smooth_curve(acc, factor=smooth_factor)
val_acc = historical_history['val_acc']
val_acc = smooth_curve(val_acc, factor=smooth_factor)
loss = historical_history['loss']
loss = smooth_curve(loss, factor=smooth_factor)
val_loss = historical_history['val_loss']
val_loss = smooth_curve(val_loss, factor=smooth_factor)

epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'b-', label='Training acc')
plt.plot(epochs, acc, 'b*', label='Training acc')
plt.plot(epochs, val_acc, 'r--', label='Validation acc')
plt.plot(epochs, val_acc, 'ro', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
# plt.show()
image_name = output_dir + '/accuracy_smooth_'+str(smooth_factor)+'.png'
plt.savefig(image_name)
plt.close()


plt.figure()
plt.plot(epochs, loss, 'b-', label='Training loss')
plt.plot(epochs, loss, 'b*', label='Training loss')
plt.plot(epochs, val_loss, 'r--', label='Validation loss')
plt.plot(epochs, val_loss, 'ro', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

image_name = output_dir + '/loss_smooth_'+str(smooth_factor)+'.png'
plt.savefig(image_name)
plt.close()
