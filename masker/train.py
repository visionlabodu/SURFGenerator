import segmentation_models as sm
import tensorflow as tf
import glob
import numpy as np

from natsort import natsorted

sm.set_framework("tf.keras")


temp = tf.zeros([4, 32, 32, 3])
 
p = sm.get_preprocessing('efficientnetb0')
temp = p(temp)

#see paper for details
class preprocess:

    def returnFirstImage(self, path):
        

        path = str(path)

        #print(path)

        if('dry/Bright/' in path):
            if '000.png' in path:
                return 'dry/Bright/0600000.png'
            elif '001.png' in path:
                return 'dry/Bright/0600001.png'
            elif '002.png' in path:
                return 'dry/Bright/0600002.png'
            elif '003.png' in path:
                return 'dry/Bright/0600003.png'
            elif '004.png' in path:
                return 'dry/Bright/0600004.png'
            elif '005.png' in path:
                return 'dry/Bright/0600005.png'
            elif '006.png' in path:
                return 'dry/Bright/0600006.png'
            elif '007.png' in path:
                return 'dry/Bright/0600007.png'
            else:
                return path + "error"
            
        elif('dry/Dark/' in path):       
            if '000.png' in path:
                return 'dry/Dark/0600000.png'
            elif '001.png' in path:
                return 'dry/Dark/0600001.png'
            elif '002.png' in path:
                return 'dry/Dark/0600002.png'
            elif '003.png' in path:
                return 'dry/Dark/0600003.png'
            elif '004.png' in path:
                return 'dry/Dark/0600004.png'
            elif '005.png' in path:
                return 'dry/Dark/0600005.png'
            elif '006.png' in path:
                return 'dry/Dark/0600006.png'
            elif '007.png' in path:
                return 'dry/Dark/0600007.png'
            else:
                return path + "error"
        elif('dry/Dim/' in path):
            if '000.png' in path:
                return 'dry/Dim/0600000.png'
            elif '001.png' in path:
                return 'dry/Dim/0600001.png'
            elif '002.png' in path:
                return 'dry/Dim/0600002.png'
            elif '003.png' in path:
                return 'dry/Dim/0600003.png'
            elif '004.png' in path:
                return 'dry/Dim/0600004.png'
            elif '005.png' in path:
                return 'dry/Dim/0600005.png'
            elif '006.png' in path:
                return 'dry/Dim/0600006.png'
            elif '007.png' in path:
                return 'dry/Dim/0600007.png'
            else:
                return path + "error"
        else:
            return path + "error"

    def returnFirstLabel(self, path):

        
        path = str(path)
        
        #print(path)

        if('/1/' in path):
            if '000.png' in path:
                return 'levels/levels/1/0600000.png'
            elif '001.png' in path:
                return 'levels/levels/1/0600001.png'
            elif '002.png' in path:
                return 'levels/levels/1/0600002.png'
            elif '003.png' in path:
                return 'levels/levels/1/0600003.png'
            elif '004.png' in path:
                return 'levels/levels/1/0600004.png'
            elif '005.png' in path:
                return 'levels/levels/1/0600005.png'
            elif '006.png' in path:
                return 'levels/levels/1/0600006.png'
            elif '007.png' in path:
                return 'levels/levels/1/0600007.png'
            else:
                return path + "error"
        elif('/2/' in path):       
            if '000.png' in path:
                return 'levels/levels/2/0600000.png'
            elif '001.png' in path:
                return 'levels/levels/2/0600001.png'
            elif '002.png' in path:
                return 'levels/levels/2/0600002.png'
            elif '003.png' in path:
                return 'levels/levels/2/0600003.png'
            elif '004.png' in path:
                return 'levels/levels/2/0600004.png'
            elif '005.png' in path:
                return 'levels/levels/2/0600005.png'
            elif '006.png' in path:
                return 'levels/levels/2/0600006.png'
            elif '007.png' in path:
                return 'levels/levels/2/0600007.png'
            else:
                return path + "error"
        elif('/3/' in path):
            if '000.png' in path:
                return 'levels/levels/3/0600000.png'
            elif '001.png' in path:
                return 'levels/levels/3/0600001.png'
            elif '002.png' in path:
                return 'levels/levels/3/0600002.png'
            elif '003.png' in path:
                return 'levels/levels/3/0600003.png'
            elif '004.png' in path:
                return 'levels/levels/3/0600004.png'
            elif '005.png' in path:
                return 'levels/levels/3/0600005.png'
            elif '006.png' in path:
                return 'levels/levels/3/0600006.png'
            elif '007.png' in path:
                return 'levels/levels/3/0600007.png'
            else:
                return path + "error"
        elif('/4/' in path):
            if '000.png' in path:
                return 'levels/levels/4/0600000.png'
            elif '001.png' in path:
                return 'levels/levels/4/0600001.png'
            elif '002.png' in path:
                return 'levels/levels/4/0600002.png'
            elif '003.png' in path:
                return 'levels/levels/4/0600003.png'
            elif '004.png' in path:
                return 'levels/levels/4/0600004.png'
            elif '005.png' in path:
                return 'levels/levels/4/0600005.png'
            elif '006.png' in path:
                return 'levels/levels/4/0600006.png'
            elif '007.png' in path:
                return 'levels/levels/4/0600007.png'
            else:
                return path + "error"
        elif('/5/' in path):
            if '000.png' in path:
                return 'levels/levels/5/0600000.png'
            elif '001.png' in path:
                return 'levels/levels/5/0600001.png'
            elif '002.png' in path:
                return 'levels/levels/5/0600002.png'
            elif '003.png' in path:
                return 'levels/levels/5/0600003.png'
            elif '004.png' in path:
                return 'levels/levels/5/0600004.png'
            elif '005.png' in path:
                return 'levels/levels/5/0600005.png'
            elif '006.png' in path:
                return 'levels/levels/5/0600006.png'
            elif '007.png' in path:
                return 'levels/levels/5/0600007.png'
            else:
                return path + "error"
        elif('/6/' in path):
            if '000.png' in path:
                return 'levels/levels/6/0600000.png'
            elif '001.png' in path:
                return 'levels/levels/6/0600001.png'
            elif '002.png' in path:
                return 'levels/levels/6/0600002.png'
            elif '003.png' in path:
                return 'levels/levels/6/0600003.png'
            elif '004.png' in path:
                return 'levels/levels/6/0600004.png'
            elif '005.png' in path:
                return 'levels/levels/6/0600005.png'
            elif '006.png' in path:
                return 'levels/levels/6/0600006.png'
            elif '007.png' in path:
                return 'levels/levels/6/0600007.png'
            else:
                return path + "error"
        elif('/7/' in path):
            if '000.png' in path:
                return 'levels/levels/7/0600000.png'
            elif '001.png' in path:
                return 'levels/levels/7/0600001.png'
            elif '002.png' in path:
                return 'levels/levels/7/0600002.png'
            elif '003.png' in path:
                return 'levels/levels/7/0600003.png'
            elif '004.png' in path:
                return 'levels/levels/7/0600004.png'
            elif '005.png' in path:
                return 'levels/levels/7/0600005.png'
            elif '006.png' in path:
                return 'levels/levels/7/0600006.png'
            elif '007.png' in path:
                return 'levels/levels/7/0600007.png'
            else:
                return path + "error"
        elif('/8/' in path):
            if '000.png' in path:
                return 'levels/levels/8/0600000.png'
            elif '001.png' in path:
                return 'levels/levels/8/0600001.png'
            elif '002.png' in path:
                return 'levels/levels/8/0600002.png'
            elif '003.png' in path:
                return 'levels/levels/8/0600003.png'
            elif '004.png' in path:
                return 'levels/levels/8/0600004.png'
            elif '005.png' in path:
                return 'levels/levels/8/0600005.png'
            elif '006.png' in path:
                return 'levels/levels/8/0600006.png'
            elif '007.png' in path:
                return 'levels/levels/8/0600007.png'
            else:
                return path + "error"
        elif('/9/' in path):
            if '000.png' in path:
                return 'levels/levels/9/0600000.png'
            elif '001.png' in path:
                return 'levels/levels/9/0600001.png'
            elif '002.png' in path:
                return 'levels/levels/9/0600002.png'
            elif '003.png' in path:
                return 'levels/levels/9/0600003.png'
            elif '004.png' in path:
                return 'levels/levels/9/0600004.png'
            elif '005.png' in path:
                return 'levels/levels/9/0600005.png'
            elif '006.png' in path:
                return 'levels/levels/9/0600006.png'
            elif '007.png' in path:
                return 'levels/levels/9/0600007.png'
            else:
                return path + "error"
        elif('/10/' in path):
            if '000.png' in path:
                return 'levels/levels/10/0600000.png'
            elif '001.png' in path:
                return 'levels/levels/10/0600001.png'
            elif '002.png' in path:
                return 'levels/levels/10/0600002.png'
            elif '003.png' in path:
                return 'levels/levels/10/0600003.png'
            elif '004.png' in path:
                return 'levels/levels/10/0600004.png'
            elif '005.png' in path:
                return 'levels/levels/10/0600005.png'
            elif '006.png' in path:
                return 'levels/levels/10/0600006.png'
            elif '007.png' in path:
                return 'levels/levels/10/0600007.png'
            else:
                return path + "error"
        else:
            return path + "error"



    def __init__(self, name):
        self.name = name
    


    def __call__(self, nameImage, nameLabel):
        
        pre = sm.get_preprocessing('efficientnetb0')

        image = tf.io.read_file(nameImage)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [256, 512], tf.image.ResizeMethod.BICUBIC)
        image = pre(image)
        
    
        nameImagePrev = tf.py_function(func=self.returnFirstImage,
        inp=[nameImage], Tout = tf.string)
        imagePrev = tf.io.read_file(nameImagePrev)
        imagePrev = tf.io.decode_jpeg(imagePrev, channels=3)
        imagePrev = tf.cast(imagePrev, tf.float32)
        imagePrev = tf.image.resize(imagePrev, [256, 512], tf.image.ResizeMethod.BICUBIC)
        imagePrev = pre(imagePrev)
        
        
        nameLabelPrev = tf.py_function(func=self.returnFirstLabel,
        inp=[nameLabel], Tout = tf.string)
        labelPrev = tf.io.read_file(nameLabelPrev)
        labelPrev = tf.io.decode_png(labelPrev, channels=1)
        labelPrev = tf.cast(labelPrev, tf.float32)
        labelPrev = tf.image.resize(labelPrev, [256, 512],
        tf.image.ResizeMethod.NEAREST_NEIGHBOR)



        label = tf.io.read_file(nameLabel)
        label = tf.io.decode_png(label, channels=1)
        label = tf.cast(label, tf.float32)
        label = tf.image.resize(label, [256, 512],
        tf.image.ResizeMethod.NEAREST_NEIGHBOR)


        input = tf.concat([image, imagePrev, labelPrev], 2)


        return input, label

np.random.seed(seed = 1234)
tf.random.set_seed(1234)

#this is something like getting these sequences in the right order
levels = np.empty((10, 3, 8, 300), dtype='U50')

images = np.empty((10, 3, 8, 300), dtype='U50')

for i in range(len(levels)):

    for j in range(len(levels[i][0])):

        levels[i, :, j] = natsorted(glob.glob('levels/levels/' + str(i+1) + '/'
        + '*' + str(j) + '.png'))[1:]

for j in range(len(images[0][0])):


    images[:, 0, j] = natsorted(glob.glob('dry/Bright/' + '*' + str(j) +
    '.png'))[1:]
    images[:, 1, j] = natsorted(glob.glob('dry/Dark/' + '*' + str(j) +
    '.png'))[1:]
    images[:, 2, j] = natsorted(glob.glob('dry/Dim/' + '*' + str(j) +
    '.png'))[1:]
    





levels = levels.flatten()

images = images.flatten()


indices = list(range(len(images)))
np.random.shuffle(indices)

images = [images[i] for i in indices]

levels = [levels[i] for i in indices]







strategy = tf.distribute.MirroredStrategy()


# [9871318894 4807696530]
# 



loss_fn = sm.losses.DiceLoss()

BACKBONE = 'efficientnetb0'




preprocessor = preprocess(BACKBONE)
    
print(images[0])
print(levels[0])
list_ds = tf.data.Dataset.from_tensor_slices((images, levels))
image_ds = list_ds.map(preprocessor)

num = int(len(images)*.15)
val_dataset = image_ds.take(num) 
train_dataset = image_ds.skip(num)



train_dataset = train_dataset.shuffle(128, reshuffle_each_iteration=True).batch(4)
val_dataset = val_dataset.shuffle(128, reshuffle_each_iteration=True).batch(4)



with strategy.scope():
    model = sm.Linknet(BACKBONE, input_shape=(256, 512, 7),
    encoder_weights=None, classes=1, activation='sigmoid')
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6),
        loss=loss_fn,
        metrics=[tf.keras.metrics.BinaryAccuracy(), sm.metrics.iou_score,
        sm.metrics.f1_score, sm.metrics.f2_score, sm.metrics.precision,
        sm.metrics.recall])
    
    
    
log_dir = "logs/fit/" + BACKBONE + "Linknet"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="training/cp.ckpt",
                                                save_weights_only=True,
                                                save_best_only=True,
                                                monitor='val_loss',
                                                verbose=1)

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8)




print("TRAINING HAS BEGUN")
print("Train size: " + str(tf.data.experimental.cardinality(train_dataset)))
print("Val size: " + str(tf.data.experimental.cardinality(val_dataset)))



model.fit(train_dataset, epochs=50, validation_data = val_dataset, callbacks=[tensorboard_callback, cp_callback, callback])

model.save(BACKBONE)
