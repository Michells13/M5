from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers, regularizers, Input, activations, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, SeparableConv2D, Dense, Activation, MaxPooling2D, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# For MLOps
import wandb
from wandb.keras import WandbCallback



def createModel(configs):
    
    activations = {"relu": relu}
    regularizer = regularizers.l2(configs["reg_coef"])
    
    # Create model
    img_inputs = Input(shape=(configs["image_height"], configs["image_width"], 3))
    conv1 = SeparableConv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=configs["weight_init"])(img_inputs)
    act1 = Activation(activations[configs["activation"]])(conv1)
    batch1 = BatchNormalization()(act1)
    mPool1 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch1)

    conv2 = SeparableConv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=configs["weight_init"])(mPool1)
    act2 = Activation(activations[configs["activation"]])(conv2)
    batch2 = BatchNormalization()(act2)
    mPool2 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch2)


    conv3 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=configs["weight_init"])(mPool2)
    act3 = Activation(activations[configs["activation"]])(conv3)
    batch3 = BatchNormalization()(act3)
    mPool3 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch3)

    conv4 = SeparableConv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=configs["weight_init"])(mPool3)
    act4 = Activation(activations[configs["activation"]])(conv4)
    batch4 = BatchNormalization()(act4)
    mPool4 = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="same")(batch4)

    conv5 = SeparableConv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="SAME", kernel_initializer=configs["weight_init"])(mPool4)
    act5 = Activation(activations[configs["activation"]])(conv5)
    batch5 = BatchNormalization()(act5)
    
    global_pool = GlobalAveragePooling2D()(batch5)

    fc = Dense(configs["n_class"], activation='softmax', name='predictions', kernel_regularizer = regularizer)(global_pool)

    model = Model(inputs=img_inputs, outputs=fc)

    return model

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

train_data_dir='./MIT_small_train_1/train'
test_data_dir = "./MIT_small_train_1/test"
validation_samples = 2288

folder = "./model_VGG_SP_NOFC_keras/"
MODEL_FNAME = folder + "best_model.h5"

# create the base pre-trained model
model = createModel(configs)

opt = optimizers.Nadam(learning_rate = configs["init_learning_rate"])
model.compile(loss=configs["loss_fn"],optimizer=opt, metrics=configs["metrics"])




#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
	rescale=1./255.,
    rotation_range=configs["rotation_range"],
    width_shift_range=configs["width_shift_range"],
    height_shift_range=configs["height_shift_range"],
    shear_range=configs["shear_range"],
    brightness_range = configs["brightness_range"],
    zoom_range=configs["zoom_range"],
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=configs["horizontal_flip"],
    vertical_flip=configs["vertical_flip"])

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

train_generator = datagen.flow_from_directory(train_data_dir,
        target_size=(configs["image_height"], configs["image_width"]),
        batch_size=configs["batch_size"],
        class_mode='categorical')

test_generator = datagenTest.flow_from_directory(test_data_dir,
        target_size=(configs["image_height"], configs["image_width"]),
        batch_size=configs["batch_size"],
        class_mode='categorical')


# Init wand
run = wandb.init(project='VGG_SP_NOFC', config=configs, job_type='train')
wandb.run.name = "KERAS"
config = wandb.config


# Define WandbCallback for experiment tracking
wandb_callback = WandbCallback(monitor='val_accuracy',
                               log_weights=True,
                               log_evaluation=True,
                               validation_steps=np.ceil(validation_samples/configs["batch_size"]))

# To save the best model
checkpointer = ModelCheckpoint(filepath=MODEL_FNAME, verbose=1, save_best_only=True, 
                               monitor='val_accuracy')


history=model.fit(train_generator,
        steps_per_epoch=(np.ceil(400/configs["batch_size"])),
        epochs=config.epochs,
        validation_data=test_generator,
        validation_steps= (np.ceil(validation_samples/configs["batch_size"])), callbacks=[checkpointer, wandb_callback])

# Finish
wandb.finish()

model = load_model(MODEL_FNAME)

result = model.evaluate(test_generator)
print( result)

# list all data in history
num_p = model.count_params()

file  = open(folder + "results.txt", "w")
file.write(f"Score: {result}\n")
file.write(f"Num param: {num_p}\n")
ratio = result[1]/(num_p/100000)
file.write(f"Ratio: {ratio}\n")
file.close()

# summarize history for accuracy
fig1, ax1 = plt.subplots()
ax1.plot(history.history['accuracy'])
ax1.plot(history.history['val_accuracy'])
ax1.set_title('model accuracy')
ax1.set_ylabel('accuracy')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
fig1.savefig(folder + 'accuracy.jpg')
plt.close(fig1)
  # summarize history for loss
fig1, ax1 = plt.subplots()
ax1.plot(history.history['loss'])
ax1.plot(history.history['val_loss'])
ax1.set_title('model loss')
ax1.set_ylabel('loss')
ax1.set_xlabel('epoch')
ax1.legend(['train', 'validation'], loc='upper left')
fig1.savefig(folder + 'loss.jpg')
