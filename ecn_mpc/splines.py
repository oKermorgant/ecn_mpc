from nav_msgs.msg import Path
from scipy.interpolate import make_interp_spline
import numpy as np


def curvature(pp, p, pn) -> float:

    z1 = pp[0]+pp[1]*1j
    z2 = p[0]+p[1]*1j
    z3 = pn[0]+pn[1]*1j

    if (z1 == z2) or (z2 == z3) or (z3 == z1):
        raise ValueError(f"Duplicate points: {z1}, {z2}, {z3}")

    w = (z3 - z1)/(z2 - z1)

    # You should change 0 to a small tolerance for floating point comparisons
    if abs(w.imag) <= 0:
        return 0.

    c = (z2 - z1)*(w - abs(w)**2)/(2j*w.imag) + z1  # Simplified denominator
    r = abs(z1 - c)

    return 1./r


max_vel = 0.277778 * 30      # 30 km/h
max_curv = 1./4.             # curvature radius in 1/m
min_vel = max_vel/2.         # at highest curvature
max_acc = .5*9.81


class WP:
    def __init__(self, pose, t):
        self.x = pose.pose.position.x
        self.y = pose.pose.position.y
        self.t = t

    def dist_sq(self,x,y):
        return (x-self.x)**2+(y-self.y)**2

    def relvel_to(self, other):
        return np.sqrt(self.dist_sq(other.x,other.y))/(other.t-self.t)


class Splines:

    def __init__(self):
        self.path = None

    def set_path(self, path: Path):

        v = np.zeros(len(path.poses))
        v[0] = v[-1] = 0.
        d = np.zeros(len(path.poses))
        wp = [0]

        def toXY(p):
            return np.array([p.pose.position.x, p.pose.position.y])

        # compute correct relative times to go through each pose
        # first pose is @ t = 0
        for k,pose in enumerate(path.poses[1:-1]):

            pp = toXY(path.poses[k])
            p = toXY(pose)
            pn = toXY(path.poses[k+2])
            v1 = p-pp
            v2 = pn-p
            d[k+1] = np.linalg.norm(v1)

            if np.dot(v1,v2) > 0:
                c = curvature(pp, p, pn) / max_curv
                c = min(1, max(c, 0))
                if c < 0 or c > 1:
                    print(f'strange curvature {c}')
                # c = 0 -> max_vel, c = 1 -> min_vel
                v[k+1] = max_vel*(1-c) + min_vel*c
            else:
                wp.append(k+1)

            path.poses[k].header.stamp.sec += k
        d[-1] = d[-2]
        wp.append(len(path.poses)-1)

        # ensure max acceleration
        vi = np.array(v[:])
        ok = False
        while not ok:
            ok = True
            dv = abs(vi[1:] - vi[:-1])
            k = np.argmax(dv)
            vm = .5*(vi[k+1] + vi[k])
            dt = d[k+1] / vm
            acc = vm/dt
            if acc > 1.1*max_acc:
                ok = False
                if vi[k+1] > vi[k]:
                    vi[k+1] = vi[k] + max_acc*dt
                else:
                    vi[k] = vi[k+1] + max_acc*dt

        # write timestamps
        self.path = [WP(path.poses[0], 0.)]
        for k, pose in enumerate(path.poses[1:]):
            vm = .5*(vi[k+1] + vi[k])
            if vm != 0.:
                t = self.path[-1].t + d[k+1] / vm
            else:
                t = self.path[-1].t + 0.01
            self.path.append(WP(pose, t))

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
        sp = make_interp_spline(steps,xy)

        def ref(t):
            if t > steps[-1]:
                return np.hstack((xy[-1], [0,0]))
            return np.hstack((sp(t), [0, 0]))

        return ref

