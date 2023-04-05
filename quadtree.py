import matplotlib.patches as patches


class Point:
    def __init__(self, p, q, m):
        self.x, self.y = p
        self.q = q
        self.m = m

    def __repr__(self):
        return f"Point(({self.x}, {self.y}), {self.q})"


class Square:
    def __init__(self, x, y, d):
        self.x, self.y = x, y
        self.d = d
        self.r = d/2
        self.centre_of_charge = None

    def __repr__(self):
        return f"Square(({self.x}, {self.y}), {self.d})"

    def contains(self, point: Point):
        return (self.x - self.r <= point.x < self.x + self.r) and \
               (self.y - self.r <= point.y < self.y + self.r)

    def subdivide(self, quadrant):
        if quadrant == 0:
            return Square(self.x - self.r/2, self.y + self.r/2, self.r)
        if quadrant == 1:
            return Square(self.x + self.r/2, self.y + self.r/2, self.r)
        if quadrant == 2:
            return Square(self.x - self.r/2, self.y - self.r/2, self.r)
        if quadrant == 3:
            return Square(self.x + self.r/2, self.y - self.r/2, self.r)


class QuadTree:
    def __init__(self, boundary, max_depth=6):
        self.boundary = boundary
        self.points = []
        self.divided = False
        self.max_d = max_depth
        self.quadrants = None

    def clear(self):
        self.points = []
        if self.divided:
            self.divided = False
            del self.quadrants

    def subdivide(self):
        self.quadrants = [QuadTree(self.boundary.subdivide(i), self.max_d - 1)
                          for i in range(4)]
        self.divided = True

        for p in self.points:
            for quad in self.quadrants:
                if quad.insert(p):
                    break
        self.points = None

    def insert(self, p):
        if not self.boundary.contains(p):
            return False

        if not self.divided:
            if len(self.points) == 0 or self.max_d <= 0:
                self.points.append(p)
                return True
            self.subdivide()

        inserted = False
        for quad in self.quadrants:
            if quad.insert(p):
                inserted = True
                break
        return inserted

    def draw_quads(self, ax):
        if self.divided:
            for quad in self.quadrants:
                quad.draw_quads(ax)
        else:
            patch = patches.Rectangle((self.boundary.x - self.boundary.r,
                                       self.boundary.y - self.boundary.r),
                                      self.boundary.d, self.boundary.d,
                                      edgecolor="black", facecolor="none", linewidth=0.5)
            ax.add_patch(patch)



