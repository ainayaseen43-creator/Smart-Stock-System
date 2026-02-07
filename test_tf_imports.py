# test.py
import sys
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import tensorflow as tf
    print(f"\n✅ TensorFlow version: {tf.__version__}")
    
    # Test keras import
    from tensorflow import keras
    print("✅ Keras imported successfully")
    
    # Test model creation
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(1)
    ])
    print("✅ Model created successfully!")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
except Exception as e:
    print(f"\n❌ Error: {e}")