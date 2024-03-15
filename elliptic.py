import numpy as np
import cupy as cp

# For debug
# import matplotlib.pyplot as plt

class Elliptic:
    def __init__(self, poisson_coefficient):
        # Operators
        self.central_flux_operator = None
        self.gradient_operator = None
        self.inv_op = None

        # Fields
        self.potential = None
        self.electric_field = None
        self.magnetic_field = None

        # Charge density coefficient in poisson equation
        self.poisson_coefficient = poisson_coefficient
    
    def build_central_flux_operator(self, grid, basis):
        # Build using indicating array
        indicator = np.zeros((grid.elements, grid.order))
        # face differences for numerical flux
        face_diff0 = np.zeros((grid.elements, 2))
        face_diff1 = np.zeros((grid.elements, 2))
        num_flux = np.zeros_like(face_diff0)
        grad_num_flux = np.zeros_like(face_diff1)

        central_flux_operator = np.zeros((grid.elements, grid.order, grid.elements, grid.order))
        self.gradient_operator = np.zeros_like(central_flux_operator)

        for i in range(grid.elements):
            for j in range(grid.order):
                # Choose node
                indicator[i, j] = 1.0

                # Compute strong form boundary flux (central)
                face_diff0[:, 0] = indicator[:, 0] - np.roll(indicator[:, -1], 1)
                face_diff0[:, 1] = indicator[:, -1] - np.roll(indicator[:, 0], -1)
                # face_diff0[:, 0] = (0.5 * (indicator[:, 0] + np.roll(indicator[:, -1], 1)) -
                #                    indicator[:, 0])
                # face_diff0[:, 1] = -1.0 * (0.5 * (indicator[:, -1] + np.roll(indicator[:, 0], -1)) -
                #                           indicator[:, -1])
                num_flux[:, 0] = 0.5 * face_diff0[:, 0]
                num_flux[:, 1] = -0.5 * face_diff0[:, 1]

                # Compute gradient of this node
                grad = (np.tensordot(basis.derivative_matrix, indicator, axes=([1], [1])) +
                        np.tensordot(basis.numerical.get(), num_flux, axes=([1], [1]))).T

                # Compute gradient's numerical flux (central)
                face_diff1[:, 0] = grad[:, 0] - np.roll(grad[:, -1], 1)
                face_diff1[:, 1] = grad[:, -1] - np.roll(grad[:, 0], -1)
                grad_num_flux[:, 0] = 0.5 * face_diff1[:, 0]
                grad_num_flux[:, 1] = -0.5 * face_diff1[:, 1]

                # Compute operator from gradient matrix
                operator = (np.tensordot(basis.advection_matrix.T, grad, axes=([1], [1])) +
                            np.tensordot(basis.face_mass, grad_num_flux + face_diff0, axes=([1], [1]))).T

                # place this operator in the global matrix
                central_flux_operator[i, j, :, :] = operator
                self.gradient_operator[i, j, :, :] = grad

                # reset nodal indicator
                indicator[i, j] = 0

        # Reshape to matrix and set gauge condition by fixing quadrature integral = 0 as extra equation in system
        op0 = np.hstack([central_flux_operator.reshape(grid.elements * grid.order, grid.elements * grid.order),
                         grid.global_quads.get().reshape(grid.elements * grid.order, 1)])
        self.central_flux_operator = np.vstack([op0, np.append(grid.global_quads.get().flatten(), 0)])
        # Clear machine errors
        self.central_flux_operator[np.abs(self.central_flux_operator) < 1.0e-15] = 0

        # Send gradient operator to device
        self.gradient_operator = cp.asarray(self.gradient_operator)

    def invert(self, wavenumbers):
        self.inv_op = cp.zeros((wavenumbers.shape[0], self.central_flux_operator.shape[0], 
                                self.central_flux_operator.shape[1]))
        # self.inv_op = cp.asarray(np.linalg.inv(self.central_flux_operator))
        for idx in range(wavenumbers.shape[0]):
            modified_identity = np.identity(self.central_flux_operator.shape[0])
            modified_identity[-1, -1] = 0  # no wavenumber on gauge condition
            to_invert = self.central_flux_operator - 0 * wavenumbers[idx]**2 * modified_identity
            self.inv_op[idx, :, :] = cp.asarray(np.linalg.inv(to_invert))

    def poisson(self, charge_density, grid, basis, anti_alias=True):
        """
        Modified 1D Poisson solve (d^2 F/dx^2 = k^2 * F + S) using stabilized central flux
        """
        # Preprocess (last entry is average value)
        rhs = cp.zeros((grid.elements * grid.order + 1))
        rhs[:-1] = self.poisson_coefficient * cp.tensordot(charge_density, basis.device_mass, axes=([1], [1])).flatten()

        # Compute solution and remove last entry
        sol = cp.matmul(self.inv_op, rhs)[:-1] / (grid.J ** 2.0)
        self.potential = sol.reshape(grid.elements, grid.order)

        # Clean solution (anti-alias)
        if anti_alias:
            coefficients = grid.fourier_basis(self.potential)
            self.potential = grid.sum_fourier(coefficients)

        # Compute field as negative potential gradient
        self.electric_field = cp.zeros_like(grid.arr_cp)
        self.electric_field[1:-1, :] = -1.0*(grid.J *
                                             cp.tensordot(self.gradient_operator,
                                                          self.potential, axes=([0, 1], [0, 1])))

        # Clean solution (anti-alias)
        if anti_alias:
            coefficients = grid.fourier_basis(self.electric_field[1:-1, :])
            self.electric_field[1:-1, :] = grid.sum_fourier(coefficients)

        # Set ghost cells
        self.electric_field[0, :] = self.electric_field[-2, :]
        self.electric_field[-1, :] = self.electric_field[1, :]
