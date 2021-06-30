import biorbd
import bioviz
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import equation_muscles
from scipy.integrate import solve_ivp

def muscle_tendon_equilibre(model, emg, Q, Qdot):
    for i in range(5):
        force = model.muscleForces(emg, Q, Qdot).to_array()[0]
        strain = (model.muscle(0).characteristics().tendonSlackLength() - tendon_slack_length) / tendon_slack_length
        d = np.log(force * np.cos(model.muscle(0).characteristics().pennationAngle())/ model.muscle(0).characteristics().forceIsoMax() * cst_tendon1 + 1) * cst_tendon2
        # if strain > tendon_strain_threshold:
        #     d = model.muscle(0).characteristics().tendonSlackLength() * ((force / model.muscle(0).characteristics().forceIsoMax() - Ftoe)/klin + tendon_strain_threshold)
        # else:
        #     d = np.log(force / model.muscle(0).characteristics().forceIsoMax() * cst_tendon1 + 1) * cst_tendon2
        if d > 0:
            model.muscle(0).characteristics().setTendonSlackLength(model.muscle(0).characteristics().tendonSlackLength() + d)
        else:
            break

def muscle_tendon_equilibre2(model, emg, Q, Qdot):
    for i in range(35):
        force = model.muscleForces(emg, Q, Qdot).to_array()[0]
        a = force - biorbd.HillType(model.muscle(0)).FlPE()*600
        p = compute_passive_force(model, model.muscle(0).length(model, Q)) * 600
        force = a + p
        strain = model.muscle(0).characteristics().tendonSlackLength()/ tendon_slack_length - 1
        d = np.log(force * np.cos(model.muscle(0).characteristics().pennationAngle())/ model.muscle(0).characteristics().forceIsoMax() * cst_tendon1 + 1) * cst_tendon2
        # if (strain > tendon_strain_threshold):
        #     d = tendon_slack_length * ((force * np.cos(model.muscle(0).characteristics().pennationAngle())/ model.muscle(0).characteristics().forceIsoMax() - Ftoe)/klin + tendon_strain_threshold)
        # else:
        #     d = np.log(force * np.cos(model.muscle(0).characteristics().pennationAngle())/ model.muscle(0).characteristics().forceIsoMax() * cst_tendon1 + 1) * cst_tendon2

        if d > 0:
            model.muscle(0).characteristics().setTendonSlackLength(model.muscle(0).characteristics().tendonSlackLength() + d)
        else:
            break

def compute_tendon_force(model, tendon_length):
    strain = (tendon_length/tendon_slack_length) - 1
    if (strain > tendon_strain_threshold):
        tendon_force = klin * (strain - tendon_strain_threshold) + Ftoe
    elif (strain > 0) :
        tendon_force = (Ftoe/(np.exp(3) - 1)) * (np.exp(3*strain/tendon_strain_threshold) - 1)
    else:
        tendon_force = 0
    return tendon_force

def compute_passive_force(model, muscle_length):
    passive_force = (np.exp(kpe * (muscle_length/model.muscle(0).characteristics().optimalLength() - 1)/strain_muscle) - 1)/(np.exp(5) - 1)
    return passive_force


# OpenSim value
data = pd.read_excel('muscle_tendon_length.xlsx') # TheHelen
angle_value_os = data.values[6:, 1]
tendon_length_os = data.values[6:, 2]
fiber_length_os = data.values[6:, 3]
muscle_tendon_length_os = data.values[6:, 4]

# data = pd.read_excel('millard/length_millard.xlsx') # Millard
# angle_value_os = data.values[6:, 1]
# tendon_length_os = data.values[6:, 3]
# fiber_length_os = data.values[6:, 2]
# muscle_tendon_length_os = data.values[6:, 4]

data = pd.read_excel('muscle_force.xlsx') # TheHelen
total_force_os = data.values[6:, 2]
active_force_os = data.values[6:, 3]
passive_force_os = data.values[6:, 4]
# tendon_force_os = data.values[6:, 5]

# data = pd.read_excel('millard/force_millard.xlsx') # Millard
# total_force_os = data.values[6:, 4]
# active_force_os = data.values[6:, 2]
# passive_force_os = data.values[6:, 3]
# tendon_force_os = data.values[6:, 5]

# constant from TheHelen
Ftoe = 0.33
tendon_strain_iso = 0.04 # OpenSim
# tendon_strain_threshold = 0.609 * tendon_strain_iso
tendon_strain_threshold = (99*tendon_strain_iso*np.exp(3)) / (166*np.exp(3) - 67)
# klin = 1.712/tendon_strain_iso
klin = 67 /(100*(tendon_strain_iso - (99*tendon_strain_iso*np.exp(3))/(166*np.exp(3)-67)) )
# passive force
kpe = 5
strain_muscle = 0.6

# exp formula
cst_tendon1 = Ftoe/(np.exp(3) - 1)
cst_tendon2 = tendon_strain_threshold/3

# biorbd
model = biorbd.Model('one_muscle_model.bioMod') # b = bioviz.Viz(loaded_model=model)
tendon_slack_length = model.muscleGroup(0).muscle(0).characteristics().tendonSlackLength()
n_frame = 100

# values in array
qlin = np.linspace(-0.7, np.pi/2, n_frame)
qdotlin = np.gradient(qlin)
activation = np.repeat(1.0, 100)

# compute length and force
muscle_tendonlength_init = np.zeros(n_frame)
muscle_length_init = np.zeros(n_frame)
tendon_length_init = np.zeros(n_frame)
muscle_force_init = np.zeros(n_frame)
muscle_force_P_init = np.zeros(n_frame)
muscle_force_A_init = np.zeros(n_frame)
tendon_force_init = np.zeros(n_frame)

muscle_tendonlength= np.zeros(n_frame)
muscle_length = np.zeros(n_frame)
tendon_length = np.zeros(n_frame)
muscle_force = np.zeros(n_frame)
muscle_force_P = np.zeros(n_frame)
muscle_force_A = np.zeros(n_frame)
tendon_force = np.zeros(n_frame)

muscle_length_scipy = np.zeros(n_frame)
tendon_length_scipy = np.zeros(n_frame)
muscle_force_scipy = np.zeros(n_frame)
muscle_force_P_scipy = np.zeros(n_frame)
muscle_force_A_scipy = np.zeros(n_frame)
tendon_force_scipy = np.zeros(n_frame)
alpha = np.zeros(n_frame)
muscle_jacobian = np.zeros(n_frame)

for i in range(n_frame):
    model.muscle(0).characteristics().setTendonSlackLength(tendon_slack_length)  # init tendon slack length = 0.12
    Q = biorbd.GeneralizedCoordinates(np.array(qlin[i]).reshape(1))
    Qdot = biorbd.GeneralizedVelocity(np.array(qdotlin[i]).reshape(1))
    muscle_length_init[i] = model.muscle(0).length(model, Q)
    muscle_tendonlength_init[i] = model.muscle(0).musculoTendonLength(model, Q)
    tendon_length_init[i] = model.muscle(0).characteristics().tendonSlackLength()
    emg = model.stateSet()
    for e in emg:
        e.setActivation(activation[i])
    muscle_force_init[i] = model.muscleForces(emg, Q, Qdot).to_array()
    muscle_force_A_init[i] = biorbd.HillType(model.muscle(0)).FlCE(emg[0])
    muscle_force_P_init[i] = biorbd.HillType(model.muscle(0)).FlPE()
    tendon_force_init[i] = compute_tendon_force(model, tendon_length_init[i])

    muscle_tendon_equilibre2(model, emg, Q, Qdot)
    muscle_length[i] = model.muscle(0).length(model, Q)
    muscle_tendonlength[i] = model.muscle(0).musculoTendonLength(model, Q)
    tendon_length[i] = model.muscle(0).characteristics().tendonSlackLength()
    muscle_force[i] = model.muscleForces(emg, Q, Qdot).to_array()
    muscle_force_A[i] = biorbd.HillType(model.muscle(0)).FlCE(emg[0])
    muscle_force_P[i] = biorbd.HillType(model.muscle(0)).FlPE()
    tendon_force[i] = compute_tendon_force(model, tendon_length[i])

    sol = solve_ivp(equation_muscles.inverse_velocity, [0, 0.1], [muscle_length[i] / 0.1], method='RK45', dense_output=True, args=(qlin[i], activation[i], ))
    muscle_length_scipy[i] = sol.y[0, -1]
    tendon_length_scipy[i] = equation_muscles.compute_tendon_length(0, muscle_length_scipy[i], qlin[i], activation[i])
    muscle_force_P_scipy[i] = equation_muscles.compute_passive_force(0, muscle_length_scipy[i], qlin[i], activation[i])
    muscle_force_A_scipy[i] = equation_muscles.compute_active_force(0, muscle_length_scipy[i], qlin[i], activation[i])
    tendon_force_scipy[i] = equation_muscles.compute_tendon_force(0, muscle_length_scipy[i], qlin[i], activation[i])
    muscle_force_scipy[i] = equation_muscles.compute_muscle_force(0, muscle_length_scipy[i], qlin[i], activation[i])
    alpha[i] = equation_muscles.compute_pennation_angle(0, muscle_length_scipy[i], qlin[i], activation[i])
    muscle_jacobian[i] = model.musclesLengthJacobian(Q).to_array()

# np.save('ratio/lm_12', muscle_length_scipy)
# np.save('ratio/lt_12', tendon_length_scipy)
# np.save('ratio/fa_12', muscle_force_A_scipy)
# np.save('ratio/fp_12', muscle_force_P_scipy)


FPE = compute_passive_force(model, muscle_length)
fiber_length_os = fiber_length_os.astype(float)
FPE_os = compute_passive_force(model, fiber_length_os)

# # MVT
# b = bioviz.Viz(loaded_model=model)
# b.load_movement(qlin)

# FIGURES INIT
# --- Longueur --- #
fig = plt.figure()
plt.plot(qlin*180/np.pi, muscle_length_init, 'r')
plt.plot(angle_value_os, fiber_length_os, 'r--')
plt.plot(qlin*180/np.pi, tendon_length_init, 'b')
plt.plot(angle_value_os, tendon_length_os, 'b--')
plt.ylabel('Length (m)', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Joint angle (deg)', fontsize=20)
plt.xticks(fontsize=20)
plt.xlim([-0.7*180/np.pi, np.pi/2*180/np.pi])
plt.title('Muscle and tendon length - Thelen', fontsize=20)
fig.legend(['muscle length biorbd', 'muscle length OpenSim', 'tendon length biorbd', 'tendon length OpenSim'], fontsize=18)
plt.grid()

# --- Force --- #
fig = plt.figure()
plt.plot(qlin*180/np.pi, muscle_force_A_init*600, 'r')
plt.plot(angle_value_os, active_force_os, 'r--')
plt.plot(qlin*180/np.pi, muscle_force_P_init*600, 'b')
plt.plot(angle_value_os, passive_force_os, 'b--')
plt.ylabel('Force (N)', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Joint angle (deg)', fontsize=20)
plt.xticks(fontsize=20)
plt.xlim([-0.7*180/np.pi, np.pi/2*180/np.pi])
plt.title('Active and passive muscle force - Thelen', fontsize=20)
fig.legend(['active muscle force biorbd', 'active muscle force OpenSim', 'passive muscle force biorbd', 'passive muscle force OpenSim'], fontsize=18)
plt.grid()

# FIGURES ITERATION
# --- Longueur --- #
fig = plt.figure()
plt.plot(qlin*180/np.pi, muscle_length, color='r', marker='+', linestyle='-')
plt.plot(qlin*180/np.pi, muscle_length_init, 'r', alpha=0.5)
plt.plot(angle_value_os, fiber_length_os, 'r--')
plt.plot(qlin*180/np.pi, tendon_length, color='b', marker='+', linestyle='-')
plt.plot(qlin*180/np.pi, tendon_length_init, 'b', alpha=0.5)
plt.plot(angle_value_os, tendon_length_os, 'b--')
plt.ylabel('Length (m)', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Joint angle (deg)', fontsize=20)
plt.xticks(fontsize=20)
plt.xlim([-0.7*180/np.pi, np.pi/2*180/np.pi])
plt.title('Muscle and tendon length - Thelen', fontsize=20)
fig.legend(['muscle length iteration', 'muscle length biorbd', 'muscle length OpenSim', 'tendon length iteration', 'tendon length biorbd', 'tendon length OpenSim'], fontsize=18)
plt.grid()

# --- Force --- #
fig = plt.figure()
plt.plot(qlin*180/np.pi, muscle_force_A*600, color='r', marker='+', linestyle='-')
plt.plot(qlin*180/np.pi, muscle_force_A_init*600, 'r', alpha=0.5)
plt.plot(angle_value_os, active_force_os, 'r--')
plt.plot(qlin*180/np.pi, muscle_force_P*600, color='b', marker='+', linestyle='-')
plt.plot(qlin*180/np.pi, muscle_force_P_init*600, 'b', alpha=0.5)
plt.plot(angle_value_os, passive_force_os, 'b--')
plt.ylabel('Force (N)', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Joint angle (deg)', fontsize=20)
plt.xticks(fontsize=20)
plt.xlim([-0.7*180/np.pi, np.pi/2*180/np.pi])
plt.title('Active and passive muscle force - Thelen', fontsize=20)
fig.legend(['active muscle force iteration', 'active muscle force biorbd', 'active muscle force OpenSim', 'passive muscle force iteration', 'passive muscle force biorbd', 'passive muscle force OpenSim'], fontsize=18)
plt.grid()


# FIGURES SCIPY
# --- Longueur --- #
fig = plt.figure()
plt.plot(qlin*180/np.pi, muscle_length_scipy*0.1, color='r', marker='o', linestyle='-')
plt.plot(qlin*180/np.pi, muscle_length_init, 'r', alpha=0.5)
plt.plot(angle_value_os, fiber_length_os, 'r--')
plt.plot(qlin*180/np.pi, tendon_length_scipy, color='b', marker='o', linestyle='-')
plt.plot(qlin*180/np.pi, tendon_length_init, 'b', alpha=0.5)
plt.plot(angle_value_os, tendon_length_os, 'b--')
plt.ylabel('Length (m)', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Joint angle (deg)', fontsize=20)
plt.xticks(fontsize=20)
plt.xlim([-0.7*180/np.pi, np.pi/2*180/np.pi])
plt.title('Muscle and tendon length - Thelen', fontsize=20)
fig.legend(['muscle length scipy', 'muscle length biorbd', 'muscle length OpenSim', 'tendon length scipy', 'tendon length biorbd', 'tendon length OpenSim'], fontsize=18)
plt.grid()

# --- Force --- #
fig = plt.figure()
plt.plot(qlin*180/np.pi, muscle_force_A_scipy*600, color='r', marker='o', linestyle='-')
plt.plot(qlin*180/np.pi, muscle_force_A_init*600, 'r', alpha=0.5)
plt.plot(angle_value_os, active_force_os, 'r--')
plt.plot(qlin*180/np.pi, muscle_force_P_scipy*600, color='b', marker='o', linestyle='-')
plt.plot(qlin*180/np.pi, muscle_force_P_init*600, 'b', alpha=0.5)
plt.plot(angle_value_os, passive_force_os, 'b--')
plt.ylabel('Force (N)', fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Joint angle (deg)', fontsize=20)
plt.xticks(fontsize=20)
plt.xlim([-0.7*180/np.pi, np.pi/2*180/np.pi])
plt.title('Active and passive muscle force - Thelen', fontsize=20)
fig.legend(['active muscle force scipy', 'active muscle force biorbd', 'active muscle force OpenSim', 'passive muscle force scipy', 'passive muscle force biorbd', 'passive muscle force OpenSim'], fontsize=18)
plt.grid()

plt.show()



# # --- Longueur --- #
# fig = plt.figure()
# plt.plot(qlin*180/np.pi, muscle_length, 'r')
# plt.plot(angle_value_os, fiber_length_os, 'r--')
# plt.plot(qlin*180/np.pi, muscle_length_init, 'r', alpha=0.5)
# plt.plot(qlin*180/np.pi, muscle_length_scipy*0.1, 'b')
# plt.ylabel('Longueur muscle (m)')
# plt.xlabel('q (deg)')
# plt.xlim([-0.7*180/np.pi, np.pi/2*180/np.pi])
# plt.title('Longueur')
# fig.legend(['muscle length iterations', 'muscle length OpenSim', 'muscle length biorbd', 'muscle length scipy'])
# plt.grid()
#
# fig = plt.figure()
# plt.plot(qlin*180/np.pi, tendon_length, 'r')
# plt.plot(angle_value_os, tendon_length_os, 'r--')
# plt.plot(qlin*180/np.pi, tendon_length_init, 'r', alpha=0.5)
# plt.plot(qlin*180/np.pi, tendon_length_scipy, 'b')
# plt.ylabel('Longueur tendon (m)')
# plt.xlabel('q (deg)')
# plt.xlim([-0.7*180/np.pi, np.pi/2*180/np.pi])
# plt.title('Longueur')
# fig.legend(['tendon length iterations', 'tendon length OpenSim', 'tendon length biorbd', 'tendon length scipy'])
# plt.grid()


# fig = plt.figure()
# plt.plot(qlin*180/np.pi, muscle_length, 'r')
# plt.plot(qlin*180/np.pi, muscle_tendonlength, 'g')
# plt.plot(qlin*180/np.pi, tendon_length, 'b')
# plt.plot(angle_value_os, fiber_length_os, 'r--')
# plt.plot(angle_value_os, muscle_tendon_length_os, 'g--')
# plt.plot(angle_value_os, tendon_length_os, 'b--')
# plt.plot(qlin*180/np.pi, muscle_length_init, 'r', alpha=0.5)
# plt.plot(qlin*180/np.pi, muscle_tendonlength_init, 'g', alpha=0.5)
# plt.plot(qlin*180/np.pi, tendon_length_init, 'b', alpha=0.5)
# plt.ylabel('Longueur muscle (m)')
# plt.xlabel('q (deg)')
# plt.xlim([-np.pi/2*180/np.pi, np.pi/2*180/np.pi])
# plt.title('Longueur')
# fig.legend(['muscle length with MT', 'muscle tendon length with MT', 'tendon length with MT',
#             'muscle length OpenSim', 'muscle tendon length OpenSim', 'tendon length OpenSim',
#             'muscle length no MT', 'muscle tendon length no MT', 'tendon length no MT'])
# plt.grid()

# --- Force --- #
# fig2 = plt.figure()
# plt.plot(qlin*180/np.pi, muscle_force, 'r')
# plt.plot(qlin*180/np.pi, muscle_force_P * 600, 'g')
# plt.plot(qlin*180/np.pi, muscle_force_A * 600, 'b')
# plt.plot(angle_value_os, total_force_os, 'r--')
# plt.plot(angle_value_os, passive_force_os, 'g--')
# plt.plot(angle_value_os, active_force_os, 'b--')
# plt.plot(qlin*180/np.pi, muscle_force_init, 'r', alpha=0.5)
# plt.plot(qlin*180/np.pi, muscle_force_P_init * 600, 'g', alpha=0.5)
# plt.plot(qlin*180/np.pi, muscle_force_A_init * 600, 'b', alpha=0.5)
# plt.xlabel('q (deg)')
# plt.ylabel('force ')
# plt.title('Muscle force')
# plt.xlim([-np.pi/2*180/np.pi, np.pi/2*180/np.pi])
# fig2.legend(['total muscle force with MT', 'passive force with MT', 'active force with MT',
#              'total muscle force OpenSim', 'passive force OpenSim','active force OpenSim',
#              'total muscle force no MT', 'passive force no MT', 'active force no MT'])
# plt.grid()

# figa = plt.figure()
# plt.plot(qlin*180/np.pi, muscle_force_A * 600, 'r')
# plt.plot(angle_value_os, active_force_os, 'r--')
# plt.plot(qlin*180/np.pi, muscle_force_A_init * 600, 'r', alpha=0.5)
# plt.plot(qlin*180/np.pi, muscle_force_A_scipy * 600, 'b')
# plt.xlabel('q (deg)')
# plt.ylabel('force ')
# plt.title('Active muscle force')
# plt.xlim([-0.7*180/np.pi, np.pi/2*180/np.pi])
# figa.legend(['active force iteration', 'active force OpenSim', 'active force biorbd', 'active force scipy'])
# plt.grid()

# figp = plt.figure()
# plt.plot(qlin*180/np.pi, muscle_force_P * 600, 'r')
# plt.plot(angle_value_os, passive_force_os, 'r--')
# plt.plot(qlin*180/np.pi, muscle_force_P_init * 600, 'r', alpha=0.5)
# plt.plot(qlin*180/np.pi, muscle_force_P_scipy * 600, 'b')
# plt.xlabel('q (deg)')
# plt.ylabel('force ')
# plt.title('Passive muscle force')
# plt.xlim([-0.7*180/np.pi, np.pi/2*180/np.pi])
# figp.legend(['passive force iterations', 'passive force OpenSim', 'passive force biorbd', 'passive force scipy'])
# plt.grid()


# plt.figure('FORCE LONGUEUR')
# plt.plot(muscle_length/0.1, muscle_force_A, 'r')
# plt.plot(fiber_length_os/0.1, active_force_os/600, 'r--')
# plt.plot(muscle_length_scipy, muscle_force_A_scipy, 'b')
# plt.plot([muscle_length[0]/0.1, muscle_length[-1]/0.1], [1, 1], 'k--')
# plt.xlabel('longueur muscle normalisée')
# plt.ylabel('force normalisée')
# plt.legend(['iterations', 'OpenSim', 'scipy'])
#



# # --- Force Longueur --- #
# figlf = plt.figure()
# plt.plot(muscle_length/model.muscle(0).characteristics().optimalLength(), muscle_force_A, 'b')
# plt.plot(muscle_length/model.muscle(0).characteristics().optimalLength(), muscle_force_P, 'g')
# plt.plot(fiber_length_os/model.muscle(0).characteristics().optimalLength(), active_force_os/600, 'b--')
# plt.plot(fiber_length_os/model.muscle(0).characteristics().optimalLength(), passive_force_os/600, 'g--')
# plt.plot(muscle_length_init/model.muscle(0).characteristics().optimalLength(), muscle_force_A_init, 'b', alpha=0.5)
# plt.plot(muscle_length_init/model.muscle(0).characteristics().optimalLength(), muscle_force_P_init, 'g', alpha=0.5)
# plt.plot(muscle_length/model.muscle(0).characteristics().optimalLength(), FPE, 'm')
# plt.xlabel('normalized length')
# plt.ylabel('normalized force ')
# plt.title('Force longueur')
# figlf.legend(['active force with MT', 'passive force with MT'
#               'active force OpenSim', 'passive force OpenSim',
#               'active force no MT', 'passive force no MT', ])
# plt.grid()

# # --- tendon --- #
# figt = plt.figure()
# plt.plot(qlin*180/np.pi, tendon_force_scipy* 600, 'r')
# plt.plot(angle_value_os, tendon_force_os*600, 'r--')
# plt.plot(qlin*180/np.pi, tendon_force_init * 600, 'r', alpha=0.5)
# plt.plot(qlin*180/np.pi, muscle_force_scipy, 'k', alpha=0.5)
# plt.plot(angle_value_os, total_force_os, 'k--', alpha=0.5)
# plt.xlabel('q (deg)')
# plt.title('tendon force')
# figt.legend(['force tendon with MT', 'force tendon OpenSim', 'force tendon with no MT'])
# plt.grid()


# figst = plt.figure()
# plt.plot(tendon_length/tendon_slack_length - 1, 'r')
# plt.plot([0, 100], [tendon_strain_threshold, tendon_strain_threshold], 'k--')
# plt.show()
#
# figlt = plt.figure()
# plt.plot(tendon_length/tendon_slack_length - 1, tendon_force, 'r')
# # plt.plot(tendon_length_os/tendon_slack_length - 1, tendon_force_os, 'r--')
# plt.xlabel('tendon strain')
# plt.title('tendon force')
# # figlt.legend(['force tendon with MT', 'force tendon OpenSim', 'force tendon with no MT'])
# plt.grid()
# plt.show()

