import os
from math import *
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression


def find_mask_boundaries(mask, radius=5):
    """
    :param mask: numpy.array, входное изображение маски.
    :param radius: int, радиус для операции дилатации (по умолчанию 5).
    :return: list, список координат границ маски.
        - boundary_points_1: список, точки границы для маски 1.
        - boundary_points_2: список, точки границы для маски 2.
    """
    mask_1 = (mask == 1).astype(np.uint8)
    mask_2 = (mask == 2).astype(np.uint8)

    contours_1, _ = cv2.findContours(mask_1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_2, _ = cv2.findContours(mask_2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boundary_points_1 = []
    boundary_points_2 = []

    expanded_mask_1 = cv2.dilate(mask_1, np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8))
    expanded_mask_2 = cv2.dilate(mask_2, np.ones((2 * radius + 1, 2 * radius + 1), dtype=np.uint8))

    for contour in contours_1:
        for point in contour[:, 0]:
            x, y = point[0], point[1]
            if expanded_mask_2[y, x] == 1:
                boundary_points_1.append((x, y))

    for contour in contours_2:
        for point in contour[:, 0]:
            x, y = point[0], point[1]
            if expanded_mask_1[y, x] == 1:
                boundary_points_2.append((x, y))

    return boundary_points_1, boundary_points_2


def get_line_by_points(image, points):
    """
    Функция для нахождения линии по двум точкам.

    Параметры:
    :param image: numpy.array, изображение, для которого находится линия.
    :param points: np.array, массив точек для построения линии.

    Возвращает:
    :return tuple, содержащий координаты двух точек, образующих найденную линию.
    """

    regressor = LinearRegression()

    # Обучаем модели линейной регрессии для каждого кластера
    regressor.fit(points[:, 0].reshape(-1, 1), points[:, 1])

    # Получаем коэффициенты регрессии для каждого кластера
    slope = regressor.coef_[0]
    intercept = regressor.intercept_
    y1 = int(slope * 0 + intercept)
    y2 = int(slope * (image.shape[1] - 1) + intercept)
    return (0, y1), (image.shape[1] - 1, y2)


def kmeans_lines(image, points, k=2):
    """
   Функция, которая выполняет кластеризацию KMeans на точках для поиска линий в изображении.

   Параметры:
   :param image: numpy.array, входное изображение для поиска линий.
   :param points: np.array, массив точек для кластеризации.
   :param k: int, количество кластеров для KMeans (по умолчанию равно 2).

   Возвращает:
   :return lines:
   - список, содержащий кортежи координат двух точек, образующих каждую линию.
   """
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(points)
    labels = kmeans.labels_
    cluster = []
    lines = []
    for i in range(k):
        cluster.append(points[labels == i])
        lines.append(get_line_by_points(image, cluster[i]))
    return lines


def draw_lr(image, lines):
    """
    Функция для рисования двух линий на входном изображении.

    Parameters:
    :param image: numpy.array, входное изображение.
    :param lines: список кортежей, каждый из которых содержит координаты двух точек, определяющих линию.
    :return void
    """
    cv2.line(image, lines[0][0], lines[0][1], (255, 0, 0), 4)
    cv2.line(image, lines[1][0], lines[1][1], (0, 0, 255), 4)


def draw_params(image, a1, a2):
    """
    Функция для записи значений переменных на изображение.
    Параметры:
    :param image: numpy.array, входное изображение.
    :param a1: int, значение первого угла для отображения.
    :param a2: int, значение второго угла для отображения.
    Возвращает:
    :return image: numpy.array, изображение с добавленными значениями углов.
    """
    delta = radians(a1 - a2)
    cos_delta = round(abs(cos(delta)), 5)

    # запись значения переменных на изображение
    image = draw_number(image, "a1 = " + str(a1), font_scale=1, margin=220, color=(255, 0, 0))  # угол для первой прямой
    image = draw_number(image, "a2 = " + str(a2), font_scale=1, margin=180,
                        color=(127, 127, 255))  # угол для второй прямой
    if cos_delta > 0.99:
        image = draw_number(image, "|cos(a1-a2)| = " + str(cos_delta), font_scale=1, margin=140, color=(144, 238, 144))
    else:
        image = draw_number(image, "|cos(a1-a2)| = " + str(cos_delta), font_scale=1, margin=140)


def angle_to_horizontal(point1, point2):
    """
    Функция для определения угла прямой относительно горизонтали по двум точкам на изображении.

    Параметры:
    :param point1: tuple, координаты первой точки (x, y).
    :param point2: tuple, координаты второй точки (x, y).

    Возвращает:
    :return angle: float, угол прямой относительно горизонтали в градусах.
    """
    # Извлекаем координаты точек
    x1, y1 = point1
    x2, y2 = point2

    # Находим изменение координаты y
    dy = y2 - y1

    # Находим изменение координаты x
    dx = x2 - x1

    # Если dx равно нулю, это вертикальная прямая, возвращаем 90 градусов
    if dx == 0:
        return 90.0

    # Находим тангенс угла наклона прямой
    tangent = dy / dx

    # Находим угол в радианах
    angle_rad = atan(tangent)

    # Переводим угол радиан в градусы
    angle_deg = degrees(angle_rad)

    # Корректируем угол в диапазоне от -90 до 90 градусов
    if dx < 0:
        angle_deg += 180
    elif dx > 0 > dy:
        angle_deg += 360

    return round(angle_deg, 5)


def plot_in_image(frame, data, min_y=0, max_y=0):
    """
    Функция для отображения графика на изображении.

    Параметры:
    :param frame: numpy.array, изображение.
    :param data: list, список значений для графика.
    :param min_y: int, минимальное значение для графика.
    :param max_y: int, максимальное значение для графика.
    :return void
    """
    # Наложение графика на текущий кадр

    plt.figure(figsize=(6, 4))
    plt.plot(data)
    if min_y or max_y:
        plt.ylim(min_y, max_y)
    plt.grid(True)
    plt.savefig('temp_graph.png')
    graph = cv2.imread('temp_graph.png')
    graph = cv2.resize(graph, (600, 400))  # Уменьшение размера графика для лучшего помещения на кадр

    # Наложение графика на текущий кадр
    frame[10:410, 10:610] = graph  # Наложение графика в верхний левый угол

    # Удаление временного файла графика
    os.remove('temp_graph.png')


def plot_grad(frame, data1, data2):
    """
    Функция для отображения графика градиентов на изображении.

    Параметры:
    :param frame:
    :param data1:
    :param data2:
    :return:
    """
    data3 = [0] * len(data1)
    plt.figure(figsize=(6, 4))
    plt.plot(data1, color="red")
    plt.plot(data2, color="blue")
    plt.plot(data3, color="green")
    plt.grid(True)

    plt.savefig('temp_graph.png')
    graph = cv2.imread('temp_graph.png')

    graph = cv2.resize(graph, (600, 400))  # Уменьшение размера графика для лучшего помещения на кадр
    frame[10:410, 10:610] = graph  # Наложение графика в верхний левый угол
    os.remove('temp_graph.png')


def gradient(point1, point2):
    """
    Функция для вычисления градиента прямой по двум точкам.

    Параметры:
    :param point1: tuple - Кортеж с координатами первой точки (x1, y1).
    :param point2: tuple - Кортеж с координатами второй точки (x2, y2).

    Возвращает:
    :return float: Градиент (угловой коэффициент) прямой.
    """
    x1, y1 = point1
    x2, y2 = point2
    return (y2 - y1) / (x2 - x1)


def calculate_class_area(mask, class_value=1):
    """
    Функция для расчета площади класса на маске.

    Параметры:
    :param mask: numpy.array, маска, содержащая значения классов.
    :param class_value: int, значение класса, площадь которого нужно посчитать.

    Возвращает:
    :return area: int, площадь класса на маске.
    """
    # Подсчитываем количество пикселей с заданным значением класса
    area = np.sum(mask == class_value)
    return area


def visualize_mask(image, mask):
    """
    Функция для визуализации маски на изображении.

    Параметры:
    :param image: numpy.array, входное изображение.
    :param mask: numpy.array, маска, содержащая значения классов (0, 1, 2).

    Возвращает:
    :return numpy.array, изображение с визуализированной маской.
    """
    # Создание палитры для визуализации классов
    colormap = np.array([[0, 0, 0],  # Класс 0: Отсутствие класса - Черный
                         [255, 0, 0],  # Класс 1: Первый класс - Красный
                         [0, 255, 0]])  # Класс 2: Второй класс - Зеленый

    # Применение маски к изображению
    masked_image = colormap[mask]

    # Добавление маски к изображению
    result = cv2.addWeighted(image, 0.5, masked_image.astype(np.uint8), 0.5, 0)

    return result


def draw_number(image, number, color=(255, 255, 255), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2,
                margin=10):
    """
    Функция для отображения цифры на изображении в середине внизу.

    Параметры:
    :param image: numpy.array, входное изображение.
    :param number: int, число для отображения.
    :param color: tuple, цвет текста в формате BGR.
    :param font: int, шрифт OpenCV.
    :param font_scale: float, масштаб шрифта.
    :param thickness: int, толщина линии шрифта.
    :param margin: int, отступ от края изображения.

    Возвращает:
    :return image_with_number: numpy.array, изображение с добавленной цифрой.
    """
    # Получаем размер текста
    text_size, _ = cv2.getTextSize(str(number), font, font_scale, thickness)

    # Определяем позицию текста
    text_x = int((image.shape[1] - text_size[0]) / 2)
    text_y = image.shape[0] - margin

    # Рисуем текст на изображении
    image_with_number = cv2.putText(image, str(number), (text_x, text_y), font, font_scale, color, thickness)

    return image_with_number
