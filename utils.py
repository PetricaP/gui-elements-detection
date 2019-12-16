import contextlib
import logging
import time
from collections import namedtuple

point = namedtuple('point', 'x y')
circle = namedtuple('circle', 'center radius')
rectangle = namedtuple('rectangle', 'x y w h')


def point_to_json(self: point):
    return {'x': int(self.x), 'y': int(self.y)}


point.to_json = point_to_json


def circle_to_json(self: circle):
    return {'center': self.center.to_json(), 'radius': int(self.radius)}


circle.to_json = circle_to_json


def rectangle_to_json(self: rectangle):
    return {'x': int(self.x), 'y': int(self.y), 'w': int(self.w), 'h': int(self.h)}


rectangle.to_json = rectangle_to_json


def rectangle_from_json(json_rect):
    return rectangle(json_rect['x'], json_rect['y'], json_rect['w'], json_rect['h'])


rectangle.from_json = rectangle_from_json


@contextlib.contextmanager
def timer(operation_name):
    old = time.time()
    yield
    diff = time.time() - old
    logging.info(f'{operation_name} finished in {diff:.4f} seconds')


def overlap(r1: rectangle, r2: rectangle):
    x11, y11, w1, h1 = r1
    x21, y21, w2, h2 = r2

    x12 = x11 + w1
    y12 = y11 + h1

    x22 = x21 + w2
    y22 = y21 + h2

    d1 = min(y12, y22) - max(y11, y21)
    d2 = min(x12, x22) - max(x11, x21)

    if d1 <= 0 or d2 <= 0:
        return 0

    a1 = (x12 - x11) * (y12 - y11)
    a2 = (x22 - x21) * (y22 - y21)

    # The ratio between the overlap area and the smaller rectangle area
    return d1 * d2 / min(a1, a2)


def join(r1: rectangle, r2: rectangle):
    x1 = min(r1.x, r2.x)
    y1 = min(r1.y, r2.y)
    x2 = max(r1.x + r1.w, r2.x + r2.w)
    y2 = max(r1.y + r1.h, r2.y + r2.h)

    return rectangle(x1, y1, x2 - x1, y2 - y1)


def is_point_inside_rect(p: point, r: rectangle):
    return all([
        p.x > r.x,
        p.y > r.y,
        p.x < r.x + r.w,
        p.y < r.y + r.h
    ])


def is_inside_circle(p: point, c: circle):
    return (p.x - c.center.x) ** 2 + (p.y - c.center.y) ** 2 < c.radius ** 2


def is_rect_inside_rect(r1: rectangle, r2: rectangle):
    return all([
        r1.x > r2.x,
        r1.y > r2.y,
        r1.x + r1.w < r2.x + r2.w,
        r1.y + r1.h < r2.y + r2.h
    ])
