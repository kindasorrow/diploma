from UNETModel import UNet
import tensorflow as tf

print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.config.list_physical_devices("GPU") else "OFF" }')
unet = UNet()
unet.build_model()
unet.load_weights()

print(unet.model)

