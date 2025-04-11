class ZoeMPC:

    def __init__(self, dt):
        self.dt = dt

    def solve(self, x0, ref):

        # ref is a function that returns the desired (x,y) state at a given time
        # assuming t=0 is the current state x0

        # should return the next control input u and the predicted states x_k
        traj = [x0]
        for _ in range(10):
            traj.append(traj[-1][:])
            traj[-1][0] += 1.

        return [0.,0.], traj
