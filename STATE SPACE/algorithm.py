import numpy as np
import matplotlib.pyplot as plt

# PARAMETER CTMS INVERTED PENDULUM
M = 0.5
m = 0.2
b = 0.1
I = 0.006
g = 9.8
l = 0.3

p = I*(M+m) + M*m*l**2 #denominator

# STATE SPACE MODEL
A = np.array([
[0, 1, 0, 0], #posisi cart berubah sesuai kecepatan
[0, -(I+m*l**2)*b/p, (m**2*g*l**2)/p, 0], #persamaan gerak cart (nilai positif dari sudut phi, kalau pendulum miring car kedorong)
[0, 0, 0, 1], #sudut berubah sesuai kecepatan sudutnya
[0, -(m*l*b)/p, m*g*l*(M+m)/p, 0] #persamaan gerak pendulum (nilai positif besar dari phi berarti gravitasi mendestabilkan pendulum, makanya kalau miring jadi makin miring-unstable)
])

B = np.array([
[0], #gaya tidak langsung mengubah posisi cart
[(I+m*l**2)/p], #gaya langsung mempengaruhi kecepatan cart
[0], #gaya tidak langsung mengubah sudut pendulum
[m*l/p] #gaya ke cart mempengaruhi kecepatan sudut penduum melalui kopling mekanik
])

C = np.array([
[1,0,0,0],   # output pertama: posisi cart
[0,0,1,0]    # output kedua: sudut pendulum
])

D = np.zeros((2,1)) #kedua matriks 0

# DISCRETIZATION (euler forward)
dt = 0.001
Ad = np.eye(4) + A*dt
Bd = B*dt

# KALMAN FILTER PARAMETER
Q = np.eye(4)*0.001
R = np.eye(2)*0.01

P = np.eye(4)

x_hat = np.zeros((4,1))

# SIMULASI SISTEM
steps = 5000   

# kondisi awal sistem (pendulum dimulai dengan sedikit miri 2.9 derajat)
x_real = np.array([
[0], #posisi cart = 0 meter
[0], #kecepatan cart = 0 m/s
[0], #sudut pendulum = 0 rad 
[0] #kecepatan sudut == 0 rad/s
])

# penyimpanan data
theta_real = [] #sudut dari simulasi
theta_est = [] #sudut hasi; estimasi 
theta_meas = [] #sudut hasil pengukuran sendor (noisy)

#loop
for k in range(steps):

    # input kontrol (belum ada kontrol)
    u = np.array([[0]])

    # REAL SYSTEM (persamaan state diskrit)
    x_real = Ad @ x_real + Bd @ u

    # MEASUREMENT DENGAN NOISE
    y = C @ x_real + np.random.normal(0,0.01,(2,1))

    # simpan measurement
    theta_meas.append(y[1,0])

    # KALMAN PREDICTION
    x_pred = Ad @ x_hat + Bd @ u
    P_pred = Ad @ P @ Ad.T + Q

    # KALMAN UPDATE
    S = C @ P_pred @ C.T + R
    K = P_pred @ C.T @ np.linalg.inv(S)

    x_hat = x_pred + K @ (y - C @ x_pred)
    P = (np.eye(4) - K @ C) @ P_pred

    # simpan data
    theta_real.append(x_real[2,0]) #sudut nyata (elemen dari kedua state)
    theta_est.append(x_hat[2,0]) #sudut estimasi kalman

#konversi 
theta_rela = np.array(theta_real)
theta_est = np.array(theta_est)
theta_meas = np.array(theta_meas)

#hitung error
err_meas = theta_meas - theta_real
err_est = theta_est - theta_real

#hitung RMSE (root mean square error)
rmse_meas = np.sqrt(np.mean(err_meas**2))
rmse_est = np.sqrt(np.mean(err_est**2))

#noise reduction gain (dB)
db_improvement = 20*np.log10(rmse_meas / rmse_est)

# PLOT
t = np.arange(steps)*dt

fig, ax = plt.subplots(2,1, figsize=(10,8))

# Plot 1 : Estimasi Kalman
ax[0].plot(t, theta_real, label="Real θ", linewidth=2)
ax[0].plot(t, theta_meas, label="Measured θ (Noisy)", alpha=0.6)
ax[0].plot(t, theta_est, label="Estimated θ (Kalman)", linewidth=2)
ax[0].set_xlabel("Time (s)")
ax[0].set_ylabel("Angle (rad)")
ax[0].set_title("Kalman Filter Estimation - CTMS Model")
ax[0].legend()
ax[0].grid()

# Plot 2 : Error dalam dB
ax[1].plot(t, 20*np.log10(np.abs(err_meas) + 1e-6), label="Sensor Error (dB)", alpha=0.3)
ax[1].plot(t, 20*np.log10(np.abs(err_est) + 1e-6), label="Kalman Error (dB)")
ax[1].axhline(y=20*np.log10(rmse_meas), linestyle='--', label="Avg Sensor Noise")
ax[1].axhline(y=20*np.log10(rmse_est), linestyle='--', label="Avg Kalman Error")
ax[1].set_title("Error Level (dB)")
ax[1].set_ylabel("Error (dB)")
ax[1].set_xlabel("Time (s)")
ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.savefig("Kalman_Filter_Result.png")
plt.show()