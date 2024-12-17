import pygame
import numpy as np
import random

from sklearn.cluster import DBSCAN

from dbscan_class import CustomDBSCAN


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def dist(self, p):
        return np.sqrt((p.x - self.x) ** 2 + (p.y - self.y) ** 2)


def add_near_points(point):
    new_points = []
    d = 15
    for i in range(random.randint(1, 5)):
        x = point.x + random.randint(-d, d)
        y = point.y + random.randint(-d, d)
        new_points.append(Point(x, y))
    return new_points


def main():
    pygame.init()

    points = []
    colors = ['blue', 'green', 'purple', 'yellow', 'pink', 'cyan', 'magenta', 'brown', 'blue', 'green', 'purple', 'yellow', 'pink', 'cyan', 'magenta', 'brown','red']
    radius = 5

    screen = pygame.display.set_mode((600, 400), pygame.RESIZABLE)
    screen.fill("#FFFFFF")
    pygame.display.flip()  # перерисовка
    moving = False
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.WINDOWSIZECHANGED:
                screen.fill("#FFFFFF")
                for i in range(len(points)):
                    pygame.draw.circle(screen, "black", (points[i].x, points[i].y), radius)
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    moving = True
            if moving and event.type == pygame.MOUSEMOTION:
                pos = event.pos
                point = Point(*pos)
                if len(points) == 0 or points[-1].dist(point) > 30:
                    pygame.draw.circle(screen, "black", pos, radius)
                    points.append(point)
                    new_points = add_near_points(point)
                    for i in range(len(new_points)):
                        pygame.draw.circle(screen, "black", (new_points[i].x, new_points[i].y), radius)
                        points.append(new_points[i])

            if event.type == pygame.MOUSEBUTTONUP:
                moving = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # dbscan = DBSCAN(eps=30, min_samples=5)
                    # dbscan.fit(np.array(list(map(lambda p: [p.x, p.y], points))))
                    dbscan = CustomDBSCAN(eps=30, min_samples=5)
                    dbscan.fit(points)
                    labels = dbscan.labels

                    # print(labels)

                    for i in range(len(points)):
                        pygame.draw.circle(screen, colors[labels[i]], (points[i].x, points[i].y), radius)
        pygame.display.update()


if __name__ == '__main__':
    main()
