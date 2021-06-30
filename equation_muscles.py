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
Q = 20 * np.pi/180
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
e0 = 0.6     # muscle strain max

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
b33 = 0.5 * np.sqrt(0.5) #0.354
b43 = 0.000

def compute_tendon_length(t, muscle_length, q, activation):
    musculotendon_length = model.muscle(0).musculoTendonLength(model, biorbd.GeneralizedCoordinates(np.array(q).reshape(1)))
    # tendon_length = musculotendon_length - muscle_length * optimal_length * np.cos(compute_pennation_angle(t, muscle_length, q, activation))
    tendon_length = musculotendon_length - muscle_length * optimal_length * compute_cos_pennation_angle(t, muscle_length, q, activation)
    return tendon_length

def compute_tendon_force(t, muscle_length, q, activation):
    tendon_length = compute_tendon_length(t, muscle_length, q, activation)/tendon_slack_length
    tendon_force = c1*np.exp(kt*(tendon_length - c2)) - c3
    return tendon_force

def compute_passive_force(t, muscle_length, q, activation):
    passive_force = (np.exp(kpe * (muscle_length - 1)/e0) - 1)/(np.exp(kpe) - 1)
    return passive_force

def compute_active_force(t, muscle_length, q, activation):
    a = b11 * np.exp((-0.5*(muscle_length - b21)**2)/(b31 + b41*muscle_length)**2)
    b = b12 * np.exp((-0.5*(muscle_length - b22)**2)/(b32 + b42*muscle_length)**2)
    c = b13 * np.exp((-0.5*(muscle_length - b23)**2)/(b33 + b43*muscle_length)**2)
    active_force = a + b + c
    return active_force

def compute_active_force_velocity(t, muscle_velocity):
    active_force = d1*np.log((d2*muscle_velocity + d3) + np.sqrt((d2*muscle_velocity + d3)**2 + 1)) + d4
    return active_force

def compute_pennation_angle(t, muscle_length, q, activation):
    alpha = np.arcsin(optimal_length * np.sin(alpha0)/(muscle_length * optimal_length))
    return alpha

def compute_cos_pennation_angle(t, muscle_length, q, activation):
    cos_alpha = np.sqrt(1 - (np.sin(alpha0/muscle_length))**2)
    return cos_alpha

def compute_muscle_force(t, muscle_length, q, activation):
    muscle_velocity = inverse_velocity(t, muscle_length, q, activation)
    muscle_force = force_iso_max * (activation * compute_active_force(t, muscle_length, q, activation) * compute_active_force_velocity(t, muscle_velocity) + compute_passive_force(t, muscle_length, q, activation))
    return muscle_force

def inverse_velocity(t, muscle_length, q, activation):
    # fv = (compute_tendon_force(t, muscle_length, q, activation)/np.cos(compute_pennation_angle(t, muscle_length, q, activation)) - compute_passive_force(t, muscle_length, q, activation)) / (activation * compute_active_force(t, muscle_length, q, activation))
    fv = (compute_tendon_force(t, muscle_length, q, activation) / compute_cos_pennation_angle(t, muscle_length, q, activation)
          - compute_passive_force(t, muscle_length, q, activation)) / (activation * compute_active_force(t, muscle_length, q, activation))
    muscle_velocity = (velocity_max/optimal_length) * 1/d2 * np.sinh(1/d1 * (fv - d4))
    return muscle_velocity


# ML = np.linspace(0, 1.5, 100)
# MV = np.linspace(-1, 1, 100)
# active_force = compute_active_force(0, ML, Q, 1.0)
# active_force_velocity = compute_active_force_velocity(0, MV)
# passive_force = compute_passive_force(0, ML, Q, 1.0)
# tendon_length = compute_tendon_length(0, ML, Q, 1.0)/tendon_slack_length
# tendon_force = compute_tendon_force(0, ML, Q, 1.0)
#
# TL = np.linspace(0.95, 1.05)
# FT = c1*np.exp(kt*(TL - c2)) - c3
#
#
# plt.figure('FORCE LONGUEUR')
# plt.plot(ML, active_force, 'r')
# plt.plot(ML, passive_force, 'b')
# plt.plot([ML[0], ML[-1]], [1, 1], 'k--')
# plt.xlabel('longueur muscle normalisée')
# plt.ylabel('force normalisée')
# plt.legend(['active', 'passive'])
#
# plt.figure('FORCE VELOCITY')
# plt.plot(MV, active_force_velocity, 'r')
# plt.plot([0, 0], [0, 1.5], 'k--')
# plt.xlabel('vitesse muscle normalisée')
# plt.ylabel('force normalisée')
#
# plt.figure('FORCE TENDON')
# plt.plot(tendon_length, tendon_force, 'r')
# plt.plot([tendon_length[0], tendon_length[-1]], [1, 1], 'k--')
# plt.xlabel('tendon length normalisée')
# plt.ylabel('force normalisée')
#

sol = solve_ivp(inverse_velocity, [0, 0.1], [muscle_length0/optimal_length], method='RK45', dense_output=True, args=(Q, 1.0, ))

plt.figure()
plt.title('Scipy solution', fontsize=20)
plt.plot(sol.t, sol.y[0, :]*0.1)
plt.xlabel('time (s)', fontsize=20)
plt.ylabel('muscle length (m)', fontsize=20)
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.show()
