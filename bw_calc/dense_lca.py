from .lca import LCA
from numpy.linalg import solve


class DenseLCA(LCA):
    def solve_linear_system(self):
        """Convert technosphere matrix from sparse to dense before solving linear system."""
        return solve(self.technosphere_matrix.todense(), self.demand_array)
