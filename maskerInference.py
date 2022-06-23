import segmentation_models as sm
import tensorflow as tf
import glob
from natsort import natsorted
import cv2
import tqdm


#get input frames and assuming that the frames are named in a manner such as:
# Frame 1: frame1
# Frame 2: frame2
# and so on
# natsort is required so the frames are ordered in numerical order
inputSequence = natsorted(glob.glob("inputSequence/*"))

# as a reminder, the reference mask should be 0 where you do not want flooding and 1 where you want flooding
referenceMask = "outputMasks/frame0.jpg"
referenceImage = inputSequence[0]




sm.set_framework("tf.keras")
temp = tf.zeros([4, 32, 32, 3])
p = sm.get_preprocessing('efficientnetb0')
temp = p(temp)

class preprocess:
    def __call__(self, nameImage, nameImageReference, nameLabelReference):
        
        pre = sm.get_preprocessing('efficientnetb0')

        image = tf.io.read_file(nameImage)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, [256, 512], tf.image.ResizeMethod.BICUBIC)
        image = pre(image)
        
        imageReference = tf.io.read_file(nameImageReference)
        imageReference = tf.io.decode_jpeg(imageReference, channels=3)
        imageReference = tf.cast(imageReference, tf.float32)
        imageReference = tf.image.resize(imageReference, [256, 512], tf.image.ResizeMethod.BICUBIC)
        imageReference = pre(imageReference)
        
        labelReference = tf.io.read_file(nameLabelReference)
        labelReference = tf.io.decode_png(labelReference, channels=1)
        labelReference = tf.cast(labelReference, tf.float32)
        labelReference = tf.image.resize(labelReference, [256, 512],
        tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        input = tf.concat([image, imageReference, labelReference], 2)


        return input


preprocessor = preprocess()
loss_fn = sm.losses.DiceLoss()
BACKBONE = 'efficientnetb0'

masker = sm.Linknet(BACKBONE, input_shape=(256, 512, 7),
    encoder_weights=None, classes=1, activation='sigmoid')
masker.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-6),
        loss=loss_fn,
        metrics=[tf.keras.metrics.BinaryAccuracy(), sm.metrics.iou_score,
        sm.metrics.f1_score, sm.metrics.f2_score, sm.metrics.precision,
        sm.metrics.recall])
masker.load_weights("masker/training/cp.ckpt")


table = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant([0, 1]),
        values=tf.constant([1, 0]),
    ),
    default_value=tf.constant(-1),
    name="class_weight"
)


for i in tqdm.tqdm(range(len(inputSequence))):
    input = preprocessor(inputSequence[i], referenceImage, referenceMask)
    input = tf.expand_dims(input, 0)
    output = masker(input)
    output = tf.squeeze(output)
    output = tf.math.round(output)


    input = cv2.imread(inputSequence[i])
    input = cv2.resize(input, (512, 256))

    output = tf.cast(output, tf.int32)
    output = table.lookup(output)
    output = tf.cast(output, tf.float32)
    output = tf.expand_dims(output, -1)

    output = tf.concat([output, output, output], 2)
    #print(output.shape)
    #print(input.shape)
    
    output = output * input


    cv2.imwrite("outputMasks/frame" + str(i) + ".jpg", output.numpy())









