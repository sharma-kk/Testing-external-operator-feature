#from firedrake import AbstractExternalOperator, dx
#from firedrake.assemble import assemble
from firedrake import *


class IntegrationOperator(AbstractExternalOperator):
    """
    Implement integration operator:

    N: U -> V

    V^{*} = \mathcal(L)(V, R)

    V = V^{**} = V^{*} -> R

    N: U x V^{*} -> R

    For crank-nicholson: N(u^{n+1}, u^{n}; v^{*})
        -> N(u, w; v^{*}) = (1/|\Omega) (u + w)/2 * dx
        -> dNdu(u, w; v^{*})
    """

    def __init__(self, *operands, function_space, derivatives=None, **kwargs):
        AbstractExternalOperator.__init__(self, *operands, function_space=function_space, derivatives=derivatives, **kwargs)


    @property
    def scheme(self):
        return self.operator_data['scheme']

    @assemble_method((0,0), (0,))
    def eval_N(self, *args, **kwargs):
        # print("\n Evaluate integration operator")

        # Get ufl operands
        u, w = self.ufl_operands

        if self.scheme == 'un':
            # test for un: int(un * dx)
            pass
        elif self.scheme == 'crank_nicholson':
            Omega = 1 # assemble(1 * dx)
            int1 = assemble((0.5/Omega)* (u[0] + w[0]) * dx)
            int2 = assemble((0.5/Omega)* (u[1] + w[1]) * dx)
            return Function(self.function_space()).assign(as_vector([int1, int2]))

    @assemble_method((1,0), (0, None))
    def eval_dNdu_action(self, *args, **kwargs):
        # print("\n Evaluate action of integration operator")

        # action(dNdu, phi) = (0.5/Omega) * int(phi * dx)
        # Get phi
        phi = self.argument_slots()[-1]

        if self.scheme == 'un':
            # test for un: 0
            pass
        elif self.scheme == 'crank_nicholson':
            Omega = 1 # assemble(Constant(1) * dx)
            action1 = assemble((0.5/Omega) * phi[0] * dx)
            action2 = assemble((0.5/Omega) * phi[1] * dx)
            return Function(self.function_space()).assign(as_vector([action1, action2]))



# Helper function #
def integral_operator(function_space, scheme='crank_nicholson', operator_data={}):
    r"""The point_expr function returns the `PointexprOperator` class initialised with :
        - point_expr : a function expression (e.g. lambda expression)
        - function space
     """
    # Set scheme
    operator_data['scheme'] = scheme
    return partial(IntegrationOperator, operator_data=operator_data, function_space=function_space)
