from nav_msgs.msg import Path
from scipy.interpolate import make_interp_spline
import numpy as np


class WP:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    def dist_sq(self,x,y):
        return (x-self.x)**2+(y-self.y)**2

    def relvel_to(self, other):
        return np.sqrt(self.dist_sq(other.x,other.y))/(other.t-self.t)


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

        # check we are not already there
        dx = np.sqrt(start.dist_sq(x,y))
        if dx < 1e-3:
            idx += 1
            start = self.path[idx]
            dx = np.sqrt(start.dist_sq(x,y))
        # get initial velocity
        if start != self.path[-1]:
            v = start.relvel_to(self.path[idx+1])
        else:
            v = self.path[-2].relvel_to(start)

        # first join starting point
        steps = [0.,dx/v]
        xy = np.array([[x,y],[start.x, start.y]])

        dt = start.t-dx/v
        # time ref wrt current position
        t0 = start.t-dt

        while idx+1 < len(self.path):
            wp = self.path[idx+1]
            if wp.t - t0 > T:
                break
            if abs(wp.t-t0-steps[-1]) > 1e-3:
                steps.append(wp.t-t0)
                xy = np.vstack((xy,[wp.x,wp.y]))
            idx += 1
        print([wp.t for wp in self.path[max(0,idx-len(steps)):idx]])
        print(steps)
        sp = make_interp_spline(steps,xy)

        def ref(t):
            if t > steps[-1]:
                return np.hstack((xy[-1], [0,0]))
            return np.hstack((sp(t), [0, 0]))

        return ref

