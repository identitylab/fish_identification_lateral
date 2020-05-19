import tensorflow as tf
from  tensorflow import keras
import datetime
import argparse
from VGG16 import VGG16Dilated
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img, save_img
from utils import *
import os, random
from cross_validation_data_generation_tent_n_aqua import *
import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#0 = all messages are logged (default behavior)
#1 = INFO messages are not printed
#2 = INFO and WARNING messages are not printed
#3 = INFO, WARNING, and ERROR messages are not printed

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

parser = argparse.ArgumentParser(description='Split the data and generate the train and test set')
parser.add_argument('BATCH_SIZE', help='the BATCH_SIZE', nargs='?',default=8, type=int)

args = parser.parse_args()
BATCH_SIZE = args.BATCH_SIZE
File_log_name='logs/vgg16_dilated_multistage_Ids15_tent_vote.log'
def aug_data(orig_path,SAVE_PATH):
    alllist = getfilelist(orig_path)
    num_imgs = len(alllist)
    print('total number of images:', num_imgs)

    num_aug_per_img = 5
    train_datagen = ImageDataGenerator(brightness_range=[0.5, 1.5],
                                       rotation_range=5,
                                       width_shift_range=0.01,
                                       height_shift_range=0.00,
                                       shear_range=0.2,
                                       zoom_range=0.3,
                                       channel_shift_range=10,
                                       horizontal_flip=False,
                                       fill_mode='nearest')

    for file in tqdm.tqdm(alllist):
        img = load_img(file)

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        path = os.path.normpath(file)
        parts = path.split(os.sep)
        # print('processing:' + parts[-1])
        

        check_folder(SAVE_PATH + '/' + parts[-2])
        save_img(SAVE_PATH + '/' + parts[-2] + '/' + parts[-1], img)
        for batch in train_datagen.flow(x, batch_size=16, save_to_dir=SAVE_PATH + '/' + parts[-2],
                                        save_prefix=parts[-2],
                                        save_format='png'):
            i += 1
            if parts[-2].isnumeric():
                if i > num_aug_per_img:
                    break
            else:
                if i > 50:
                    break

            
def aug_data_sess1(orig_path,k,SAVE_PATH): # use k images from testing dataset as gallery and train the model into a classfication model for this ten IDs
    subfolders = [f.path for f in os.scandir(orig_path) if f.is_dir()]
    Filelist = []
    for dirs in subfolders:
        filename = random.choices(os.listdir(dirs), k=1)  # change dir name to whatever
        print(filename)
        for file in filename:
            Filelist.append(os.path.join(dirs, file))
    selected_Filelist = Filelist
    num_imgs = len(selected_Filelist)
    print('total number of images:', num_imgs)

    num_aug_per_img = 20
    train_datagen = ImageDataGenerator(brightness_range=[0.5, 1.5],
                                       rotation_range=5,
                                       width_shift_range=0.01,
                                       height_shift_range=0.00,
                                       shear_range=0.2,
                                       zoom_range=0.3,
                                       channel_shift_range=10,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    for file in tqdm.tqdm(selected_Filelist):
        img = load_img(file)

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        path = os.path.normpath(file)
        parts = path.split(os.sep)
        # print('processing:' + parts[-1])
        check_folder(SAVE_PATH + '/' + parts[-2])
        save_img(SAVE_PATH + '/' + parts[-2] + '/' + parts[-1], img)
        for batch in train_datagen.flow(x, batch_size=1, save_to_dir=SAVE_PATH + '/' + parts[-2],
                                        save_prefix=parts[-2],
                                        save_format='png'):
            i += 1
            if i > num_aug_per_img:
                break

def aug_data_sess(orig_path,k,SAVE_PATH):
    subfolders = [f.path for f in os.scandir(orig_path) if f.is_dir()]
    Filelist = []
    for dirs in subfolders:
        filename = random.choices(os.listdir(dirs), k=1)  # change dir name to whatever
        print(filename)
        for file in filename:
            Filelist.append(os.path.join(dirs, file))
    selected_Filelist = random.choices(Filelist, k=k)
    #selected_Filelist = Filelist
    num_imgs = len(selected_Filelist)
    print('total number of images:', num_imgs)

    num_aug_per_img = 20
    train_datagen = ImageDataGenerator(brightness_range=[0.5, 1.5],
                                       rotation_range=5,
                                       width_shift_range=0.01,
                                       height_shift_range=0.00,
                                       shear_range=0.2,
                                       zoom_range=0.3,
                                       channel_shift_range=10,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    for file in tqdm.tqdm(selected_Filelist):
        img = load_img(file)

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        path = os.path.normpath(file)
        parts = path.split(os.sep)
        # print('processing:' + parts[-1])
        check_folder(SAVE_PATH + '/' + parts[-2])
        save_img(SAVE_PATH + '/' + parts[-2] + '/' + parts[-1], img)
        for batch in train_datagen.flow(x, batch_size=1, save_to_dir=SAVE_PATH + '/' + parts[-2],
                                        save_prefix=parts[-2],
                                        save_format='png'):
            i += 1
            if i > num_aug_per_img:
                break

def reportAccu(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES,model_2ed):
    test_data_dir = './tmp_tent/test/SESSION1_LT'
    testloadData = LoadFishDataUtil(test_data_dir, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)
    sess1_test_dataset, sess1_class_num = testloadData.loadTestFishData()
    scores_session1 =getAccByvote(test_data_dir,sess1_class_num, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)

    test_data_dir = './tmp_tent/test/SESSION2'
    scores_session2 =getAccByvote(test_data_dir,sess1_class_num, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)

    test_data_dir = './tmp_tent/test/SESSION3'
    scores_session3 = getAccByvote(test_data_dir,sess1_class_num, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)

    test_data_dir = './tmp_tent/test/SESSION4'
    scores_session4 = getAccByvote(test_data_dir,sess1_class_num, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)

    return scores_session1,scores_session2,scores_session3,scores_session4

def getAccByvote(test_data_dir,sess1_class_num, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES):
    testloadData = LoadFishDataUtil(test_data_dir, BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES)
    sess1_test_dataset, sess1_class_num = testloadData.loadTestFishDataWithname()
    ds_it = iter(sess1_test_dataset)

    result = {1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 0: []}

    num_batch = 0
    for batch in sess1_test_dataset:
        imgs, label = next(ds_it)
        output = model_2ed.predict(imgs)
        output = tf.argmax(tf.transpose(output))
        for i in range(output.shape[0]):
            mylabel = label[i].numpy()[0][0]
            result[mylabel].append(int(output[i]))

    print(result)
    final = {}
    correct = 0
    for i in range(sess1_class_num):
        lst = result[i]
        modeval = [x for x in set(lst) if lst.count(x) > 1]
        if len(modeval)>0:
            modeval = modeval[0]
            final[i] = modeval
        else:
            modeval = -1
            final[i] = modeval
        if i == modeval:
            correct = correct + 1
    return correct/sess1_class_num

orig_path = './tmp_tent/train/'  # stage 1 train set dir
SAVE_PATH = './tmp_tent/SESSION1_ST_AUG0111'  #stage 1 train set dir after augmentation
# IMG_WIDTH=320
# IMG_HEIGHT=75
IMG_WIDTH=256
IMG_HEIGHT=60

generateDataset(byIDorByImages=True,train_weight=0.67) # half as train and half as test
orig_path = 'tmp_tent/train/'
SAVE_PATH = './tmp_tent/SESSION1_ST_AUG0111'

aug_data(orig_path,SAVE_PATH)
##############################first stage TRAIN_SAVE_PATH = './tmp_tent/SESSION1_LT_AUG1231'
data_dir = SAVE_PATH
data_dir_path = pathlib.Path(data_dir)
image_count = len(list(data_dir_path.glob('*/*.png')))
print('total images:',image_count)


CLASS_NAMES=None
SPLIT_WEIGHTS=(0.9, 0.1, 0.0)# train cv val vs test
myloadData = LoadFishDataUtil(data_dir,BATCH_SIZE,IMG_WIDTH,IMG_HEIGHT,CLASS_NAMES,SPLIT_WEIGHTS)
train_dataset,val_dataset,test_dataset,STEPS_PER_EPOCH, CLASS_NAMES,class_num = myloadData.loadFishData()
input_shape=(IMG_WIDTH,IMG_HEIGHT, 3)
print(f'total class:{class_num},batch size {BATCH_SIZE}')

train_dataset_num_elements = tf.data.experimental.cardinality(train_dataset).numpy()
val_dataset_num_elements = tf.data.experimental.cardinality(val_dataset).numpy()
test_dataset_num_elements = tf.data.experimental.cardinality(test_dataset).numpy()
print(f"train_dataset_num_elements {train_dataset_num_elements}, val_dataset_num_elements {val_dataset_num_elements}, test_dataset_num_elements {test_dataset_num_elements}")

model = VGG16Dilated(input_shape=input_shape,class_num=class_num,useAMSoftmax=False,name='1st_stage_VGG16')


# build model and optimizer
model.compile(optimizer=keras.optimizers.Adam(0.001),
              #loss=tfa.losses.TripletSemiHardLoss(),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model.summary())

epochs = 10
validation_steps = 20
# 在文件名中包含 epoch (使用 `str.format`)
timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = "model/"+timestamp+"vgg6_dilated_multistage_tent_ep-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
logdir = os.path.join("logs/vgg6_dilated_multistage", timestamp)
check_folder(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# This callback will stop the training when there is no improvement in
# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    period=10)

# the validation loss for three consecutive epochs.
# # train
history=model.fit(train_dataset, epochs=epochs,
                  callbacks=[cp_callback,tensorboard_callback], #TestCallback(timestamp,period=10,layername='batch_normalization_13'),
          validation_data=val_dataset, verbose=1, validation_steps=validation_steps)
# model.load_weights('model/20200514-182443vgg6_dilated_multistage_tent_ep-0010.ckpt')


##############2ed stage#################################################################################################

epochs = 20
TRAIN_SAVE_PATH = './tmp_tent/test/SESSION_LT_AUG0111'
aug_data_sess1('./tmp_tent/test/SESSION1_LT',0,TRAIN_SAVE_PATH) # augmentation
data_dir_path = pathlib.Path(TRAIN_SAVE_PATH)
image_count = len(list(data_dir_path.glob('*/*.png')))
print('total images:',image_count)

CLASS_NAMES=None
SPLIT_WEIGHTS=(0.85, 0.15, 0.0)# train cv val vs test
myloadData = LoadFishDataUtil(TRAIN_SAVE_PATH,BATCH_SIZE,IMG_WIDTH,IMG_HEIGHT,CLASS_NAMES,SPLIT_WEIGHTS)
train_dataset,val_dataset,test_dataset,STEPS_PER_EPOCH, CLASS_NAMES,class_num = myloadData.loadFishData() # !!!important the classname here shuold be reused in the folloing step
print(f'total class:{class_num},batch size {BATCH_SIZE}')

train_dataset_num_elements = tf.data.experimental.cardinality(train_dataset).numpy()
val_dataset_num_elements = tf.data.experimental.cardinality(val_dataset).numpy()
test_dataset_num_elements = tf.data.experimental.cardinality(test_dataset).numpy()
print(f"train_dataset_num_elements {train_dataset_num_elements}, val_dataset_num_elements {val_dataset_num_elements}, test_dataset_num_elements {test_dataset_num_elements}")

model_2ed = VGG16Dilated(input_shape=input_shape,class_num=class_num,useAMSoftmax=False,name='2ed_stage_VGG16')
# Grabbing the weights from the trained network
for layer_target, layer_source in zip(model_2ed.layers, model.layers):
    if layer_source.name !='dense_1':
        weights = layer_source.get_weights()
        layer_target.set_weights(weights)
        del weights

print("********************trainable*********************************")
FREEZE_LAYERS = 28
for layer in model_2ed.layers[:FREEZE_LAYERS]:
    layer.trainable = False
for layer in model_2ed.layers[FREEZE_LAYERS:]:
    print(layer.name)
    layer.trainable = True

model_2ed.compile(optimizer=keras.optimizers.Adam(0.001),
              loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
print(model_2ed.summary())

# 在文件名中包含 epoch (使用 `str.format`)
timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = "model/"+timestamp+"vgg6_stage2_sess1_dilated_multistage_tent_ep-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
logdir = os.path.join("logs/vgg6_stage2_dilated_tent_multistage", timestamp)
check_folder(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# This callback will stop the training when there is no improvement in
# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    period=10)
#
history=model_2ed.fit(train_dataset, epochs=epochs,
                 callbacks=[cp_callback,tensorboard_callback], #TestCallback(timestamp,period=10,layername='batch_normalization_13'),
          validation_data=val_dataset, verbose=1, validation_steps=validation_steps)

###################################
# evaluate on test set
# model_2ed.load_weights('model/20200514-214208vgg6_stage2_sess1_dilated_multistage_tent_ep-0020.ckpt')

scores_session1,scores_session2,scores_session3,scores_session4 = reportAccu(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES,model_2ed)
scores_session1_aqua,scores_session2_aqua,scores_session3_aqua,scores_session4_aqua =  0, 0,0,0
printstr=f"{scores_session1}  {scores_session2}  {scores_session3}  {scores_session4}\n"

with open(File_log_name, encoding="utf-8",mode="a") as data:
    data.write(printstr)


######################3rd stage########################################################################################
#TRAIN_SAVE_PATH = './tmp_tent/SESSION2_LT_AUG1231'
aug_data_sess('./tmp_tent/test/SESSION2',15,TRAIN_SAVE_PATH)
myloadData = LoadFishDataUtil(TRAIN_SAVE_PATH,BATCH_SIZE,IMG_WIDTH,IMG_HEIGHT,CLASS_NAMES,SPLIT_WEIGHTS)
train_dataset,val_dataset,test_dataset,STEPS_PER_EPOCH, CLASS_NAMES,class_num = myloadData.loadFishData()

# 在文件名中包含 epoch (使用 `str.format`)
timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = "model/"+timestamp+"vgg6_stage2_sess1n2_dilated_tent_multistage_ep-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
logdir = os.path.join("logs/vgg6_stage2_dilated_tent_multistage", timestamp)
check_folder(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# This callback will stop the training when there is no improvement in
# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    period=10)

history=model_2ed.fit(train_dataset, epochs=epochs,
                  callbacks=[cp_callback,tensorboard_callback], #TestCallback(timestamp,period=10,layername='batch_normalization_13'),
          validation_data=val_dataset, verbose=1, validation_steps=validation_steps)

#model_2ed.load_weights('model/20200103-184402vgg6_stage2_sess1n2_dilated_multistage_ep-0010.ckpt')

###################################
# evaluate on test set
scores_session1,scores_session2,scores_session3,scores_session4 = reportAccu(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES,model_2ed)
scores_session1_aqua,scores_session2_aqua,scores_session3_aqua,scores_session4_aqua =  0, 0,0,0
printstr=f"{scores_session1}  {scores_session2}  {scores_session3}  {scores_session4}\n"

with open(File_log_name, encoding="utf-8",mode="a") as data:
    data.write(printstr)
'''
######################4th stage########################################################################################
#TRAIN_SAVE_PATH = './tmp_tent/SESSION3_LT_AUG1231'
aug_data_sess('./tmp_tent/test/SESSION3',15,TRAIN_SAVE_PATH)
myloadData = LoadFishDataUtil(TRAIN_SAVE_PATH,BATCH_SIZE,IMG_WIDTH,IMG_HEIGHT,CLASS_NAMES,SPLIT_WEIGHTS)
train_dataset,val_dataset,test_dataset,STEPS_PER_EPOCH, CLASS_NAMES,class_num = myloadData.loadFishData()

# 在文件名中包含 epoch (使用 `str.format`)
timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = "model/"+timestamp+"vgg6_stage2_sess1n2n3_dilated_tent_multistage_ep-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
logdir = os.path.join("logs/vgg6_stage2_dilated_tent_multistage", timestamp)
check_folder(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# This callback will stop the training when there is no improvement in
# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    period=10)

history=model_2ed.fit(train_dataset, epochs=epochs,
                  callbacks=[cp_callback,tensorboard_callback], #TestCallback(timestamp,period=10,layername='batch_normalization_13'),
          validation_data=val_dataset, verbose=1, validation_steps=validation_steps)

###################################
# evaluate on test set
scores_session1,scores_session2,scores_session3,scores_session4 = reportAccu(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES,model_2ed)
scores_session1_aqua,scores_session2_aqua,scores_session3_aqua,scores_session4_aqua =  0, 0,0,0
printstr=f"{scores_session1}  {scores_session2}  {scores_session3}  {scores_session4}\n "


with open(File_log_name, encoding="utf-8",mode="a") as data:
    data.write(printstr)

######################5th stage########################################################################################
#TRAIN_SAVE_PATH = './tmp_tent/SESSION3_LT_AUG1231'
aug_data_sess('./tmp_tent/test/SESSION4',15,TRAIN_SAVE_PATH)
myloadData = LoadFishDataUtil(TRAIN_SAVE_PATH,BATCH_SIZE,IMG_WIDTH,IMG_HEIGHT,CLASS_NAMES,SPLIT_WEIGHTS)
train_dataset,val_dataset,test_dataset,STEPS_PER_EPOCH, CLASS_NAMES,class_num = myloadData.loadFishData()

# 在文件名中包含 epoch (使用 `str.format`)
timestamp=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_path = "model/"+timestamp+"vgg6_stage2_sess1n2n3n4_dilated_tent_multistage_ep-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
logdir = os.path.join("logs/vgg6_stage2_dilated_tent_multistage", timestamp)
check_folder(logdir)
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
# This callback will stop the training when there is no improvement in
# 创建一个回调，每 5 个 epochs 保存模型的权重
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    period=10)

history=model_2ed.fit(train_dataset, epochs=epochs,
                  callbacks=[cp_callback,tensorboard_callback], #TestCallback(timestamp,period=10,layername='batch_normalization_13'),
          validation_data=val_dataset, verbose=1, validation_steps=validation_steps)

###################################
# evaluate on test set
scores_session1,scores_session2,scores_session3,scores_session4 = reportAccu(BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, CLASS_NAMES,model_2ed)
scores_session1_aqua,scores_session2_aqua,scores_session3_aqua,scores_session4_aqua =  0, 0,0,0
printstr=f"{scores_session1}  {scores_session2}  {scores_session3}  {scores_session4}\n "


with open(File_log_name, encoding="utf-8",mode="a") as data:
    data.write(printstr)
'''