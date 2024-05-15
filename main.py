from UNETModel import UNet
import tensorflow as tf
import cv2
import warnings


print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.config.list_physical_devices("GPU") else "OFF"}')
warnings.filterwarnings("ignore")

unet = UNet()
unet.build_model()
unet.load_weights()

print(unet.model)

cap = cv2.VideoCapture('http://192.168.0.6:8080/video')

while True:
    ret, frame = cap.read()
    cv2.imshow("Capturing", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
# parse sensors json
# https://192.168.0.6:8080/sensors.json?from=1715756092410&sense=accel,mag,gyro,light,proximity,motion,gravity,motion_event,lin_accel,motion_active,rot_vector
