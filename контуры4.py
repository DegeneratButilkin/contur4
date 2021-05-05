import cv2
import numpy as np
import math

# функция поворота изображения
def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

# возвращает отсортированные по строкам таблицу ячеек
# [[((x,y),(w,h),angle)),((x,y),(w,h),angle)),...],[((x,y),(w,h),angle)),((x,y),(w,h),angle)),...],]
def сортировка_контуров(контуры):
    контуры2 = контуры.copy()
    таблицаЯчеек = []
    while len(контуры2) > 0:
        t = контуры2[0] # первый попавшийся контур(ячейка)
        y = контуры2[0][0][1] # смотрим кординату y
        строкаКонтуров = [] # здесь будут контрура принадлежащие одной строке
        for _ in контуры2[:]: # смотрим все ячейки
            nextY = _[0][1]
            if y+5 > nextY > y-5: # если y следующей ячейки находится в диапазоне +-5 пикселей
                y = nextY 
                строкаКонтуров.append(_) #добавляем эту ячейку в список
                контуры2.remove(_) # и удаляем из входного сиска
        таблицаЯчеек.append(строкаКонтуров)# получившиюся строку добавляем в выходной список
    return таблицаЯчеек # возвращаем влложенный список строк
            
# функция просто ищет в строке ячеек мин и макс координаты углов ячеек и на основе их вырезает нужный прямоугольник
def показать_строку(таблицаТочек,n):
    L = len(таблицаТочек)
    строка = таблицаТочек[L-n]

    maxY = 0
    for _ in строка:
        for __ in _:
            if __[1] > maxY:
                maxY = __[1]
    minY = строка[0][0][1]
    for _ in строка:
        for __ in _:
            if __[1] < minY:
                minY = __[1]

    maxX = 0
    for _ in строка:
        for __ in _:
            if __[0] > maxX:
                maxX = __[0]
    minX = строка[0][0][0]
    for _ in строка:
        for __ in _:
            if __[0] < minX:
                minX = __[0]

    #print(maxY,minY,maxX,minX)
    #input()



    crop_img = rotimg[int(minY):int(maxY), int(minX):int(maxX)]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
            
            
        
        

image_file = "bt.png"
img = cv2.imread(image_file) # чтение в img
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # конвертация в серый
ret, thresh = cv2.threshold(gray, 20, 255, 0, cv2.THRESH_BINARY) #всё что темнее 20 краситься в 0 иначе в 255
img_erode = cv2.erode(thresh, np.ones((3, 3), np.uint8), iterations=1) # чёрные пиксели красят вокрук все соседние, 1 проход

# поиск контуров (здесь нужен только самый большой контур) для определения угла поворота
contours, hierarchy = cv2.findContours(img_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()



# определение угла поворота по самой нижней линии контура
cnt = contours[1]  # самый большой контур копировать в cnt
rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
box = np.int0(box)  # округление координат
#print(box)
for _ in box:
    xy = tuple(_)
    output = cv2.circle(output, xy, 1, (0, 0, 255), 2)

# нахождение 2 самых нижних точек
t = -1
indt1 = -1
for i, _ in enumerate(box):
    if _[1] > t:
        t = _[1]
        indt1 = i

t = -1
indt2 = -1
for i, _ in enumerate(box):
    if indt1 != i:
        if _[1] > t:
            t = _[1]
            indt2 = i
t1 = box[indt1]
t2 = box[indt2]

edge1 = (t1[0] - t2[0])
edge2 = (t1[1] - t2[1])
#print(edge1, edge2)
# edge1 edge2 - катеты прямоугольного треугольника
angle = 180.0 / math.pi * math.atan(edge2 / edge1)
print("угол", angle)
# конец определения угла поворота

roterode = rotate_image(img_erode, angle) # поворот уже преобразованного изображения на нужный угол
rotimg = rotate_image(img, angle) #поворот оригинального изображения
cv2.imshow("rotate", rotimg)  # показывает повёрнутое изображение
cv2.waitKey(0) # ждать нажатия клавиши (иначе ничего не покажет)

# ещё раз поиск контуров но уже на повёрнутом изображении, все координаты точек в contours, а в hierarchy их иерархия (вложенность)
contours, hierarchy = cv2.findContours(roterode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# здесь функция преобразования беспорядочных координат отрезков контуров из contours
# с помощью информации об их иерархии преобразуется в ячейку вида ((x,y),(w,h),angle)) - прямоугольник
n = 0
ячейки = []
for idx, contour in enumerate(contours):
    #(x, y, w, h) = cv2.boundingRect(contour)
    # print("R", idx, x, y, w, h, cv2.contourArea(contour), hierarchy[0][idx])
    # hierarchy[i][0]: the index of the next contour of the same level
    # hierarchy[i][1]: the index of the previous contour of the same level
    # hierarchy[i][2]: the index of the first child
    # hierarchy[i][3]: the index of the parent
    if hierarchy[0][idx][3] == 1:
        n+=1
        rect = cv2.minAreaRect(contour) # выход:((x,y),(w,h),angle))
        ячейки.append(rect)
       
#
таблица = сортировка_контуров(ячейки) # сортировка прямоугольников по строкам
print(таблица[0])
print("-----")

# преобразование ячеек из x, y, ширина, высота, угол в просто координаты 4 углов.
таблицаТочек = []
for _ in таблица:
    строка = []
    for __ in _:
        #print(__)
        #input()
        строка.append(cv2.boxPoints(__))
    таблицаТочек.append(строка)
#-----------------------------------------------------

print(таблицаТочек[0])

# просто рисование красных кружков в каждой ячейке (нафиг не надо)
for _ in таблицаТочек:
    for __ in _:
        for ___ in __:
            #print(___)
            #input()
            cv2.circle(rotimg,(int(___[0]),int(___[1])) , 2, (0,0,255), 2)
#---------------------------------------------------------------

показать_строку(таблицаТочек,7) # показ строки


# просто рисование номера ячейки  в каждой ячейке (нафиг не надо)
n = 1
d = 1
for _ in таблица:
    for __ in _:
        #cv2.circle(rotimg,(int(__[0][0]),int(__[0][1])) , 2, (0,0,255), 2)
        cv2.putText(rotimg, str(n),(int(__[0][0]),int(__[0][1])),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2, cv2.LINE_AA)
        n+=1
    d+=1
#print(n)
        

        


#cv2.imshow("Input", img)
#cv2.imshow("gray", gray)
#cv2.imshow("thresh", thresh)
#cv2.imshow("Enlarged", img_erode)
#cv2.imshow("rotimg", rotimg)
#cv2.waitKey(0)
