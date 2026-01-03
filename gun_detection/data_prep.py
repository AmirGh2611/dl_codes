import cv2
from glob import glob


def show_gun(img, cordinates):
    for i in range(len(cordinates)):
        x1, y1, x2, y2 = cordinates[i]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("result", img)
    cv2.waitKey(0)




for image, label in zip(glob("Images/*.jpeg"), glob("Labels/*.txt")):
    img = cv2.imread(image)
    f = open(label, "r")
    a = int(f.readline().strip())
    points = []
    for _ in range(a):
        x1, y1, x2, y2 = map(int, f.readline().strip().split())
        points.append([x1, y1, x2, y2])
    show_gun(img, points)
    f.close()
cv2.destroyAllWindows()