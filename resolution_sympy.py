import biorbd
import bioviz
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# equations Degroote 2016

# constantes
# muscle parameters
model = biorbd.Model('one_muscle_model.bioMod')
Q = -40 * np.pi/180
muscle_length0 = model.muscle(0).length(model, biorbd.GeneralizedCoordinates(np.array(Q).reshape(1)))
tendon_slack_length = model.muscleGroup(0).muscle(0).characteristics().tendonSlackLength()
optimal_length = model.muscle(0).characteristics().optimalLength()
velocity_max = 10
force_iso_max = model.muscle(0).characteristics().forceIsoMax()
alpha0 = model.muscle(0).characteristics().pennationAngle()

# muscle velocity
d1 = -0.318
d2 = -8.149
d3 = -0.374
d4 = 0.886

# tendon force
kt = 35
c1 = 0.2
c2 = 0.995
c3 = 0.250

# passive force
kpe = 5.0
e0 = 0.6

# active force
b11 = 0.815
b21 = 1.055
b31 = 0.162
b41 = 0.063
b12 = 0.433
b22 = 0.717
b32 = -0.030
b42 = 0.200
b13 = 0.100
b23 = 1.00
b33 = 0.5 * np.sqrt(0.5)
b43 = 0.000

mt = 0.3
a = 1.0

def compute_tendon_length(muscle_length, mt, activation):
    tendon_length = mt - muscle_length * optimal_length * compute_cos_pennation_angle(muscle_length, mt, activation)
    return tendon_length

def compute_tendon_force(muscle_length, mt, activation):
    tendon_length = compute_tendon_length(muscle_length, mt, activation)/tendon_slack_length
    tendon_force = c1*sp.exp(kt*(tendon_length - c2)) - c3
    return tendon_force

def compute_passive_force(muscle_length, mt, activation):
    passive_force = (sp.exp(kpe * (muscle_length - 1)/e0) - 1)/(sp.exp(kpe) - 1)
    return passive_force

def compute_active_force(muscle_length, mt, activation):
    a = b11 * sp.exp((-0.5*(muscle_length - b21)**2)/(b31 + b41*muscle_length)**2)
    b = b12 * sp.exp((-0.5*(muscle_length - b22)**2)/(b32 + b42*muscle_length)**2)
    c = b13 * sp.exp((-0.5*(muscle_length - b23)**2)/(b33 + b43*muscle_length)**2)
    active_force = a + b + c
    return active_force

def compute_cos_pennation_angle(muscle_length, mt, activation):
    cos_alpha = sp.sqrt(1 - (sp.sin(alpha0/muscle_length))**2)
    return cos_alpha

def inverse_velocity(muscle_length, mt, activation):
    fv = (compute_tendon_force(muscle_length, mt, activation) / compute_cos_pennation_angle(muscle_length, mt, activation)
          - compute_passive_force(muscle_length, mt, activation)) / (activation * compute_active_force(muscle_length, mt, activation))
    muscle_velocity = (velocity_max/optimal_length) * 1/d2 * sp.sinh(1/d1 * (fv - d4))
    return muscle_velocity

# resolution sympy
# t, a, mt = sp.symbols('t, a, mt')
t = sp.symbols('t')
lm = sp.Function('lm')(t)

diffeq = sp.Eq(lm.diff(t), inverse_velocity(lm, mt, a))
sp.dsolve(diffeq, lm)