from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt

modelPath = "./model_VGG_SP_NOFC_keras/best_model.h5"
train_data_dir = "./MIT_small_train_1/train"
test_data_dir = "./MIT_small_train_1/test"

configs = dict(
    dataset = 'MIT_small_train_1',
    n_class = 8,
    image_width = 256,
    image_height = 256,
    batch_size = 32,
    model_name = 'VGG_SP_NOFC_keras',
    epochs = 100,
    init_learning_rate = 0.01,
    optimizer = 'nadam',
    loss_fn = 'categorical_crossentropy',
    metrics = ['accuracy'],
    weight_init = "glorot_normal",
    activation = "relu",
    regularizer = "l2",
    reg_coef = 0.01,
    # Data augmentation
    width_shift_range = 0,
    height_shift_range = 0,
    horizontal_flip = False,
    vertical_flip = False,
    rotation_range = 0,
    brightness_range = [0.8, 1.2],
    zoom_range = 0.15,
    shear_range = 0

)

model = load_model(modelPath)

datagenTest = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	rescale=1./255.,
    rotation_range=0.,
    width_shift_range=0.,
    height_shift_range=0.,
    shear_range=0.,
    zoom_range=0.,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=False,
    vertical_flip=False)


test_generator = datagenTest.flow_from_directory(test_data_dir,
        target_size=(configs["image_height"], configs["image_width"]),
        batch_size=2288,
        class_mode='categorical')


labels = ["Opencountry", "coast", "forest", "highway", "inside_city", "mountain", "street", "tallbuilding"]
X, y_test = test_generator.next()
y_pred = model.predict(X)

y_test_1 = np.argmax(y_test, axis=1)
y_pred_1 = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test_1, y_pred_1, normalize='true')
cm = np.around(cm, decimals = 2)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.show()

