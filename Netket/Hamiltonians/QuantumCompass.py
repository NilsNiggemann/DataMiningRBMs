
import netket as nk
from netket.graph import Lattice
from netket.operator.spin import sigmax, sigmaz
def neighbor(site, delx, dely, Lx, Ly):
    # Find the coordinates of the current site
    x = site % Lx
    y = int(site / Lx) % Ly

    # Find the coordinates of the neighboring site
    x = (x + delx) % Lx
    y = (y + dely) % Ly

    # Find the index of the neighboring site
    neighbor = x + Lx * y

    return neighbor

def get_Hamiltonian(Jz, Jx, Jp, Lx, Ly):

    # Spin based Hilbert Space
    g = Lattice(basis_vectors = [[1, 0], [0, 1]], pbc = True, extent = [Lx, Ly])
    hi = nk.hilbert.Spin(s = 1/2, N = g.n_nodes)

    # print("N = {} sites".format(g.n_nodes))
    # print("")

    H = 0
    for i in range(g.n_nodes):
        # Neighbor right
        j = neighbor(i, +1, 0, Lx, Ly)
        # print("Site {}, neighbor right {}".format(i, j))
        H += -Jx * sigmax(hi, i) @ sigmax(hi, j)

        # Neighbor up
        j = neighbor(i, 0, +1, Lx, Ly)
        # print("Site {}, neighbor up {}".format(i, j))
        H += -Jz * sigmaz(hi, i) @ sigmaz(hi, j)

        # 2nd neighbor up
        # Must divide by 2 due to double counting!
        j = neighbor(i, 0, +2, Lx, Ly)
        # print("Site {}, 2nd neighbor up {}".format(i, j))
        H += (Jp / 2) * sigmaz(hi, i) @ sigmaz(hi, j)

        # Plaquette
        j = neighbor(i, +1, 0, Lx, Ly)
        k = neighbor(i, 0, +1, Lx, Ly)
        l = neighbor(i, +1, +1, Lx, Ly)
        # print("Plaquette {} {} {} {}".format(i, j, l, k))
        H += (Jp / 2) * sigmaz(hi, i) @ sigmaz(hi, j) @ sigmaz(hi, l) @ sigmaz(hi, k)

        # print("")

    return g, hi, H