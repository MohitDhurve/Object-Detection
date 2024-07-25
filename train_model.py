import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Set parameters
input_shape = (224, 224, 3)
num_classes = 17  # Adjust based on the number of classes you have
batch_size = 32
epochs = 10

# Load MobileNetV3Small without the top layer
base_model = MobileNetV3Small(input_shape=input_shape, include_top=False, weights='imagenet')

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare data generators
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'path_to_dataset',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs
)

# Save the model to TensorFlow SavedModel format
saved_model_dir = 'saved_model'
model.save(saved_model_dir)

# Convert the model to GraphDef format
graph = tf.function(lambda x: model(x))
concrete_function = graph.get_concrete_function(tf.TensorSpec([None, 224, 224, 3], tf.float32))

# Serialize the model to .pbtxt
tf.io.write_graph(graph_or_graph_def=concrete_function.graph,
                  logdir='.',
                  name='mobilenet_v3_model.pbtxt',
                  as_text=True)
