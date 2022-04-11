import jax
import jax.numpy as jnp

from collections import namedtuple

from .ParamTransforms import *

Layer = namedtuple(
    "Layer",
    [
        "thickness",
        "density",
        "fd_parameters_function",
        "n_parameters",
        "orientation_angle",
    ],
)


class MaterialModel:
    """Defines the layers of the laminated composite; we suggest that layers are laid symmetrically; 
        ---------
        layer[N]
        ---------
           ...
        ---------
        Layer[1]
        ---------
        Layer[0]
        ---------
        Layer[1]
        ---------
           ...
        --------
        Layer[N]
        ---------
        The midplane passes through Layer[0], so in the code we treat it as two layers with halved thickness
        """

    def __init__(self, layers):
        self._layers = layers.copy()
        first_layer = layers[0]
        halved_thickness = first_layer.thickness / 2.0
        self._layers[0] = Layer(halved_thickness, *(first_layer[1:]))

    def get_half_thickness(self,):
        return sum([_l.thickness for _l in self._layers])

    def get_inertia_terms(self,):
        I_0 = 0.0
        I_2 = 0.0
        z = 0.0

        for layer in self._layers:
            I_0 += layer.thickness * layer.density
            I_2 += layer.density * (z + layer.thickness) ** 3
            z += layer.thickness

        I_0 *= 2.0
        I_2 *= 2.0 / 3.0  # 2 accounts for 2 symmetric layers

        return I_0, I_2

    def get_parameter_transform(self):
        n_params = sum([_l.n_parameters for _l in self._layers])
        rotation_matrices = [
            MaterialModel.get_rotation_matrix(layer.orientation_angle) for layer in self._layers
        ]

        def _param_transform(param, omega):
            Ds = jnp.zeros((6,))
            betas = jnp.zeros((6,))
            cur_start_param = 0
            z = 0.
            for idx_layer, layer in enumerate(self._layers):
                cur_param = param[
                    cur_start_param : cur_start_param + layer.n_parameters
                ]
                Ds_cur, betas_cur = layer.fd_parameters_function(cur_param, omega)
                rotation_matrix = rotation_matrices[idx_layer]
                Ds += rotation_matrix @ Ds_cur*(z + layer.thickness)**3
                betas += rotation_matrix @ betas_cur*(z + layer.thickness)**3
                cur_start_param += layer.n_parameters
                z += layer.thickness

            Ds *= 2./3.
            betas *= 2./3.
            return Ds, betas

        return _param_transform

    def get_rotation_matrix(angle):
        # Reddy, laminates, p. 119 (pdf)
        c = jnp.cos(angle)
        s = jnp.sin(angle)
        R = jnp.array(
            [
                [
                    c ** 4,
                    2.0 * s ** 2 * c ** 2,
                    -4.0 * c ** 3 * s,
                    s ** 4,
                    -4.0 * c * s ** 3,
                    4.0 * c ** 2 * s ** 2,
                ],
                [
                    c ** 2 * s ** 2,
                    c ** 4 + s ** 4,
                    2 * (c ** 3 * s - c * s ** 3),
                    c ** 2 * s ** 2,
                    2.0 * (c * s ** 3 - c ** 3 * s),
                    -4.0 * c ** 2 * s ** 2,
                ],
                [
                    c ** 3 * s,
                    c * s ** 3 - s * c ** 3,
                    c ** 4 - 3.0 * c ** 2 * s ** 2,
                    -c * s ** 3,
                    3.0 * s ** 2 * c ** 2 - s ** 4,
                    2 * (c * s ** 3 - s * c ** 3),
                ],
                [
                    s ** 4,
                    2.0 * c ** 2 * s ** 2,
                    4.0 * c * s ** 3,
                    c ** 4,
                    4.0 * c ** 3 * s,
                    4.0 * c ** 2 * s ** 2,
                ],
                [
                    c * s ** 3,
                    c ** 3 * s - c * s ** 3,
                    3.0 * c ** 2 * s ** 2 - s ** 4,
                    -(c ** 3) * s,
                    c ** 4 - 3.0 * c ** 2 * s ** 2,
                    2.0 * (c ** 3 * s - c * s ** 2),
                ],
                [
                    c ** 2 * s ** 2,
                    c ** 3 * s - c * s ** 3,
                    3.0 * c ** 2 * s ** 2 - s ** 4,
                    -(c ** 3) * s,
                    c ** 4 - 3.0 * s ** 2 * c ** 2,
                    2.0 * (c ** 3 * s - c * s ** 3),
                ],
            ]
        )

        #return R
        return jnp.eye(6)

    def isotropic(isotropic_params, omega):
        E = isotropic_params[0]*1e9
        nu = isotropic_params[1]
        beta = isotropic_params[2]

        D = E/(1. - nu**2)
        Ds = jnp.array([D, nu * D, 0.0, D, 0.0, D * (1.0 - nu)])
        betas = jnp.full_like(Ds, beta)

        return Ds, betas
