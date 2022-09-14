from firedrake import *

n = 50  # number of mesh elements
mesh = PeriodicUnitSquareMesh(n, n)  # periodic square mesh with side length 1

V = VectorFunctionSpace(mesh, "CG", 2)
V_out = VectorFunctionSpace(mesh, "CG", 1)

u_ = Function(V, name= "Velocity")
u = Function(V, name= "VelocityNext")

v = TestFunction(V)

Dt = 0.5*(1/n)   # time-step size
half =  Constant(0.5)
alpha = Constant(0.01)
beta = Constant(0.1)

x,y = SpatialCoordinate(mesh)

i_vel = project(as_vector([sin(2*pi*y), 0]), V) # initial velocity
u_.assign(i_vel)


F = ( inner(u - u_, v)
     + Dt*half*alpha*inner(nabla_grad(u) + nabla_grad(u_), nabla_grad(v))
     + Dt*half*beta*inner(u + u_, v)
     - Dt*beta*inner(project(as_vector([Constant(assemble(u_[0]*dx)),Constant(assemble(u_[1]*dx))]), V),v)
     )*dx


outfile = File("test_file_ext_operator.pvd")
outfile.write(project(u_,V_out, name= "Velocity"))

t = Dt
iter = 1
end = 1
while (round(t,4)<=end):

    solve(F==0,u)

    if iter%10==0:
       print("t=", round(t,2))
       outfile.write(project(u, V_out, name="Velocity"))

    u_.assign(u)
    t+= Dt
    iter+= 1
