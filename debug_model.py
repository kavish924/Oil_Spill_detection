import tensorflow as tf
from tensorflow.keras.models import load_model
import model_utils
import os

print("TF Version:", tf.__version__)

files = ["model.h5", "model_final.h5"]

for f in files:
    if os.path.exists(f):
        print(f"\nChecking {f}...")
        try:
            # Try loading as full model
            with tf.keras.utils.custom_object_scope({'precision': model_utils.precision, 'recall': model_utils.recall, 'f1_score': model_utils.f1_score}):
                model = load_model(f)
                print(f"  SUCCESS: Loaded as full model. Layers: {len(model.layers)}")
        except Exception as e:
            print(f"  FAILED to load as full model: {e}")
            
            # Try loading weights into architecture
            try:
                print("  Attempting to load weights into model_utils.xception()...")
                model_arch = model_utils.xception(num_classes=8, activation_function='softmax')
                print(f"  Architecture layers: {len(model_arch.layers)}")
                model_arch.load_weights(f)
                print("  SUCCESS: Loaded weights into architecture.")
            except Exception as e2:
                print(f"  FAILED to load weights: {e2}")
    else:
        print(f"File {f} not found.")
