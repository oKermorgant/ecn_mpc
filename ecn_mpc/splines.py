from nav_msgs.msg import Path


class WP:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = 0

    def dist_sq(self,x,y):
        return (x-self.x)**2+(y-self.y)**2


class Splines:

    def __init__(self):
        self.path = None

    def set_path(self, path: Path):

        self.path = []
        for pose in path.poses:
            stamp = pose.header.stamp
            self.path.append(WP(pose.pose.position.x,
                                pose.pose.position.y,
                                stamp.sec + 1e-9*stamp.nanosec))

    def spline_from(self, x, y, T):

        if not self.path:
            return None

        # get nearest point
        start = min(self.path, key = lambda wp: wp.dist_sq(x,y))
        idx = self.path.index(start)




        return

