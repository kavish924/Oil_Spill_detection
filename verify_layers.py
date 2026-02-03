import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import model_utils
import os

def count_layers(model):
    return len(model.layers)

print("--- Architecture Check ---")
try:
    arch_model = model_utils.xception(num_classes=8, activation_function='softmax')
    print(f"model_utils.xception() layers: {count_layers(arch_model)}")
except Exception as e:
    print(f"Error building architecture: {e}")

print("\n--- File Check ---")
files = ["model.h5", "model_final.h5"]
for f in files:
    if os.path.exists(f):
        print(f"Checking {f}...")
        try:
            # Try loading as full
            with tf.keras.utils.custom_object_scope({'precision': model_utils.precision, 'recall': model_utils.recall, 'f1_score': model_utils.f1_score}):
                m = load_model(f)
                print(f"  Load Full SUCCESS. Layers: {count_layers(m)}")
        except Exception as e:
            print(f"  Load Full FAILED: {e}")
            
            # Try exploring H5 content if possible
            try:
                import h5py
                with h5py.File(f, 'r') as h5f:
                     if 'model_config' not in h5f.attrs:
                         print("  No 'model_config' in H5 attributes (Confirms weights-only or new format).")
                     else:
                         print("  'model_config' found.")
                     
                     if 'layer_names' in h5f.attrs:
                         lnames = h5f.attrs['layer_names']
                         print(f"  Layer names in H5 attrs: {len(lnames)}")
            except Exception as h5e:
                print(f"  H5 inspection failed: {h5e}")

