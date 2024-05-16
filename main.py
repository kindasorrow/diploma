from UNETModel import UNet
import tensorflow as tf
import cv2
import warnings
import numpy as np
from skimage import measure
from skimage.draw import polygon_perimeter
from skimage.morphology import dilation, disk

print(f'Tensorflow version {tf.__version__}')
print(f'GPU is {"ON" if tf.config.list_physical_devices("GPU") else "OFF"}')
warnings.filterwarnings("ignore")

rgb_colors = [
    (0,   0,   0),
    (255, 0,   0),
    (0,   255, 0),
    (0,   0,   255),
]

unet = UNet()
unet.build_model()
unet.load_weights()

print(unet.model)

cap = cv2.VideoCapture('http://192.168.0.10:8080/video')

while True:
    ret, frame = cap.read()
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # разворот кадра в случае вертикальной съемки
    sample = cv2.resize(frame, unet.sample_size)

    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    predict = unet.predict(sample)
    scale = frame.shape[0] / unet.sample_size[0], frame.shape[1] / unet.sample_size[1]
    frame = (frame / 1.5).astype(np.uint8)
    for channel in range(1, unet.classes):
        contour_overlay = np.zeros((frame.shape[0], frame.shape[1]))
        contours = measure.find_contours(np.array(predict[:, :, channel]))
        try:
            for contour in contours:
                rr, cc = polygon_perimeter(contour[:, 0] * scale[0],
                                           contour[:, 1] * scale[1],
                                           shape=contour_overlay.shape)
                contour_overlay[rr, cc] = 1
            contour_overlay = dilation(contour_overlay, disk(1))
            frame[contour_overlay == 1] = rgb_colors[channel]
        except:
            pass

    cv2.imshow("Capturing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print(len(cv2.imread('C://Work//diploma//frames//$image051.jpg')))
cv2.imshow("Capturing", cv2.imread('/frames/$image051.jpg'))
cap.release()
cv2.destroyAllWindows()
# parse sensors json
# https://192.168.0.6:8080/sensors.json?from=1715756092410&sense=accel,mag,gyro,light,proximity,motion,gravity,motion_event,lin_accel,motion_active,rot_vector
