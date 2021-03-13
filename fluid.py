"""
Based on the Jos Stam paper https://www.researchgate.net/publication/2560062_Real-Time_Fluid_Dynamics_for_Games
and the mike ash vulgarization https://mikeash.com/pyblog/fluid-simulation-for-dummies.html

https://github.com/Guilouf/python_realtime_fluidsim
"""
import numpy as np
import matplotlib
import sys, argparse
import math
import json
import matplotlib.cm as cm


class Fluid:

    def __init__(self):
        self.rotx = 1
        self.roty = 1
        self.cntx = 1
        self.cnty = -1

        self.size = 60  # map size
        self.dt = 0.2  # time interval
        self.iter = 2  # linear equation solving iteration number

        self.diff = 0.0000  # Diffusion
        self.visc = 0.0000  # viscosity

        self.s = np.full((self.size, self.size), 0, dtype=float)        # Previous density
        self.density = np.full((self.size, self.size), 0, dtype=float)  # Current density

        # array of 2d vectors, [x, y]
        self.velo = np.full((self.size, self.size, 2), 0, dtype=float) # current velocity
        self.velo0 = np.full((self.size, self.size, 2), 0, dtype=float) # previous velocity

    def step(self):
        self.diffuse(self.velo0, self.velo, self.visc)

        # for velocity
        # x0, y0, x, y
        self.project(self.velo0[:, :, 0], self.velo0[:, :, 1], self.velo[:, :, 0], self.velo[:, :, 1])

        # advect for each component (x, y) of velocity
        self.advect(self.velo[:, :, 0], self.velo0[:, :, 0], self.velo0)
        self.advect(self.velo[:, :, 1], self.velo0[:, :, 1], self.velo0)

        self.project(self.velo[:, :, 0], self.velo[:, :, 1], self.velo0[:, :, 0], self.velo0[:, :, 1])

        # for density
        self.diffuse(self.s, self.density, self.diff)

        self.advect(self.density, self.s, self.velo)

    def lin_solve(self, x, x0, a, c):
        """Implementation of the Gauss-Seidel relaxation"""
        c_recip = 1 / c

        for iteration in range(0, self.iter):
            # Calculates the interactions with the 4 closest neighbors
            x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (x[2:, 1:-1] + x[:-2, 1:-1] + x[1:-1, 2:] + x[1:-1, :-2])) * c_recip

            self.set_boundaries(x)

    def set_boundaries(self, table, objects=[], positions=[]):
        """
        Boundaries handling
        :return:
        """

        if len(table.shape) > 2:  # 3d velocity vector array
            # Simulating the bouncing effect of the velocity array
            # vertical, invert if y vector
            # horizontal component of vel is zero in vertical walls
            table[:, 0, 1] = - table[:, 0, 1]
            table[:, self.size - 1, 1] = - table[:, self.size - 1, 1]

            # horizontal, invert if x vector
            # vertical component of vel is zero in horizontal walls
            table[0, :, 0] = - table[0, :, 0]
            table[self.size - 1, :, 0] = - table[self.size - 1, :, 0]

        # corners
        table[0, 0] = 0.5 * (table[1, 0] + table[0, 1])
        table[0, self.size - 1] = 0.5 * (table[1, self.size - 1] + table[0, self.size - 2])
        table[self.size - 1, 0] = 0.5 * (table[self.size - 2, 0] + table[self.size - 1, 1])
        table[self.size - 1, self.size - 1] = 0.5 * table[self.size - 2, self.size - 1] + \
                                              table[self.size - 1, self.size - 2]
        # objects
        # objects = [np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])]
        objects = [np.zeros((4, 4), dtype=float)]
        obj = objects[0]
        positions = [[10, 30]]
        pos = positions[0]
        #table[pos[0]:pos[0] + len(obj), pos[1]:pos[1] + len(obj[0])] = 0 * table[pos[0]:pos[0] + len(obj), pos[1]:pos[1] + len(obj[0])]
        newTable = table.copy()
        d = 20
        if len(table.shape) > 2:
            for i in range(-1, len(obj) + 1, 1):
                for j in range(-1, len(obj[0]) + 1, 1):

                    if i == -1 and j > -1 and j < len(obj[0]):
                        table[pos[0] + i, pos[1] + j, 1] = - 2*table[pos[0] + i - 1, pos[1] + j, 1]
                        self.density[pos[0] + i, pos[1] + j] += d * abs(table[pos[0] + i - 1, pos[1] + j, 1])
                    if i == len(obj) and j > -1 and j < len(obj[0]):
                        table[pos[0] + i, pos[1] + j, 1] = - 2*table[pos[0] + i + 1, pos[1] + j, 1]
                        self.density[pos[0] + i, pos[1] + j] += d * abs(table[pos[0] + i + 1, pos[1] + j, 1])
                    if j == -1 and i > -1 and i < len(obj):
                        table[pos[0] + i, pos[1] + j, 0] = - 2*table[pos[0] + i, pos[1] + j - 1, 0]
                        self.density[pos[0] + i, pos[1] + j] += d * abs(table[pos[0] + i, pos[1] + j - 1, 0])
                    if j == len(obj[0]) and i > -1 and i < len(obj):
                        table[pos[0] + i, pos[1] + j, 0] = - 2*table[pos[0] + i, pos[1] + j + 1, 0]
                        self.density[pos[0] + i, pos[1] + j] += d * abs(table[pos[0] + i, pos[1] + j + 1, 0])
                    elif i > -1 and i < len(obj) and j > -1 and j < len(obj[0]):
                        table[pos[0] + i, pos[1] + j] = 0.0
                        self.density[pos[0] + i, pos[1] + j] = 400


            #table[:,:,:] = newTable[:,:,:]

    def directNeighbourValues(self, i, j, table):
        value = 0
        if j+1 < table.shape[1]:
            value += table[i, j+1]
        if i+1 < table.shape[0]:
            value += table[i+1, j]
        if j-1 >= 0:
            value += table[i, j-1]
        if i-1 >= 0:
            value += table[i-1, j]
        return value

    def diffuse(self, x, x0, diff):
        if diff != 0:
            a = self.dt * diff * (self.size - 2) * (self.size - 2)
            self.lin_solve(x, x0, a, 1 + 6 * a)
        else:  # equivalent to lin_solve with a = 0
            x[:, :] = x0[:, :]

    # for velocity, adds realistic swirly flows
    def project(self, velo_x, velo_y, p, div):
        # numpy equivalent to this in a for loop:
        # div[i, j] = -0.5 * (velo_x[i + 1, j] - velo_x[i - 1, j] + velo_y[i, j + 1] - velo_y[i, j - 1]) / self.size
        div[1:-1, 1:-1] = -0.5 * (
                velo_x[2:, 1:-1] - velo_x[:-2, 1:-1] +
                velo_y[1:-1, 2:] - velo_y[1:-1, :-2]) / self.size
        p[:, :] = 0

        self.set_boundaries(div)
        self.set_boundaries(p)
        self.lin_solve(p, div, 1, 6)

        velo_x[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * self.size
        velo_y[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * self.size

        self.set_boundaries(self.velo)

    # advect means for density: it must move along velocity field, for velocity: it must move along itself
    def advect(self, d, d0, velocity):
        dtx = self.dt * (self.size - 2)
        dty = self.dt * (self.size - 2)

        for j in range(1, self.size - 1):
            for i in range(1, self.size - 1):
                tmp1 = dtx * velocity[i, j, 0]
                tmp2 = dty * velocity[i, j, 1]
                x = i - tmp1
                y = j - tmp2

                if x < 0.5:
                    x = 0.5
                if x > (self.size - 1) - 0.5:
                    x = (self.size - 1) - 0.5
                i0 = math.floor(x)
                i1 = i0 + 1.0

                if y < 0.5:
                    y = 0.5
                if y > (self.size - 1) - 0.5:
                    y = (self.size - 1) - 0.5
                j0 = math.floor(y)
                j1 = j0 + 1.0

                s1 = x - i0
                s0 = 1.0 - s1
                t1 = y - j0
                t0 = 1.0 - t1

                i0i = int(i0)
                i1i = int(i1)
                j0i = int(j0)
                j1i = int(j1)

                try:
                    d[i, j] = s0 * (t0 * d0[i0i, j0i] + t1 * d0[i0i, j1i]) + \
                              s1 * (t0 * d0[i1i, j0i] + t1 * d0[i1i, j1i])
                except IndexError:
                    # tmp = str("inline: i0: %d, j0: %d, i1: %d, j1: %d" % (i0, j0, i1, j1))
                    # print("tmp: %s\ntmp1: %s" %(tmp, tmp1))
                    raise IndexError
        self.set_boundaries(d)

    def turn(self):
        self.cntx += 1
        self.cnty += 1
        if self.cntx == 3:
            self.cntx = -1
            self.rotx = 0
        elif self.cntx == 0:
            self.rotx = self.roty * -1
        if self.cnty == 3:
            self.cnty = -1
            self.roty = 0
        elif self.cnty == 0:
            self.roty = self.rotx
        return self.rotx, self.roty

FRAMES = 200
CONFIG = {}
f = ""

def readConfig():
    global CONFIG

    file = open(f, 'r')
    CONFIG = json.load(file)
    print(CONFIG)


def processArgs():
    global f
    # parse arguments
    parser = argparse.ArgumentParser(description="Runs fluid simulation improvement by Mariana Avalos (the-other-mariana).")
    parser.add_argument('-i', '--input', type=str, required=True,help="[STRING] Determines the initial config filename.")
    args = parser.parse_args()

    if len(sys.argv) < 2:
        return False
    elif bool(args.input):
        f = str(args.input)
    print("SUCCESS Loaded input params.")
    return True


def main() -> None:
    global f
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
        from matplotlib.animation import writers

        if not processArgs():
            print("Please provide command line arguments by typing: python fluid.py -h")
            return

        readConfig()

        inst = Fluid()

        def update_im(i):
            # notes:
            # source size will be density box size
            # velocity position will be averaged with density position

            # We add new density creators in here
            sources = CONFIG['sources']
            for s in sources:
                inst.density[s['position']['y']:s['position']['y'] + s['size'], s['position']['x']:s['position']['x'] + s['size']] += s['density']  # add density into a 3*3 square
                # We add velocity vector values in here
                dPos = int(s['size'] / 2.0)
                inst.velo[s['position']['y'] + dPos, s['position']['x'] + dPos] = [s['velocity']['y'], s['velocity']['x'] * np.sin(0.5 * i)] # [y, x] # y positive goes down

            inst.step()
            im.set_array(inst.density)
            q.set_UVC(inst.velo[:, :, 1], inst.velo[:, :, 0])
            # print(f"Density sum: {inst.density.sum()}")
            im.autoscale()

        fig = plt.figure()

        # plot density
        norm = matplotlib.colors.Normalize(vmin=1.0, vmax=1.0, clip=False)
        im = plt.imshow(inst.density, norm=norm, interpolation='bilinear')
        # cmap=cm.coolwarm
        # plot vector field
        q = plt.quiver(inst.velo[:, :, 1], inst.velo[:, :, 0], scale=10, angles='xy')
        anim = animation.FuncAnimation(fig, update_im, interval=1, frames=FRAMES)

        Writer = writers["ffmpeg"]
        writer = Writer(fps=30, metadata={'artist':'mariana'}, bitrate=1800)
        anim.save('test.mp4', writer)
        #anim.save("movie.mp4", fps=30, extra_args=['-vcodec', 'libx264'])
        #plt.show()

    except ImportError:
        import imageio

        frames = 30

        flu = Fluid()

        video = np.full((frames, flu.size, flu.size), 0, dtype=float)

        for step in range(0, frames):
            flu.density[4:7, 4:7] += 100  # add density into a 3*3 square
            flu.velo[5, 5] += [1, 2]

            flu.step()
            video[step] = flu.density

        imageio.mimsave('./video.gif', video.astype('uint8'))

# call main
if __name__ == '__main__':
    main()
