import numpy as np
import matplotlib.pyplot as plt

# PARAMETER CTMS INVERTED PENDULUM
M = 0.5
m = 0.2
b = 0.1
I = 0.006
g = 9.8
l = 0.3

p = I*(M+m) + M*m*l**2

# STATE SPACE MODEL
A = np.array([
[0, 1, 0, 0],
[0, -(I+m*l**2)*b/p, (m**2*g*l**2)/p, 0],
[0, 0, 0, 1],
[0, -(m*l*b)/p, m*g*l*(M+m)/p, 0]
])

B = np.array([
[0],
[(I+m*l**2)/p],
[0],
[m*l/p]
])

C = np.array([
[1,0,0,0],   # posisi cart
[0,0,1,0]    # sudut pendulum
])

D = np.zeros((2,1))

# DISCRETIZATION
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

# kondisi awal sistem
x_real = np.array([
[0],
[0],
[0.05],
[0]
])

# penyimpanan data
theta_real = []
theta_est = []
theta_meas = []

for k in range(steps):

    # input kontrol (belum ada kontrol)
    u = np.array([[0]])

    # REAL SYSTEM
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
    theta_real.append(x_real[2,0])
    theta_est.append(x_hat[2,0])

# PLOT
t = np.arange(steps)*dt

plt.figure(figsize=(10,5))
plt.plot(t, theta_real, label="Real θ", linewidth=2)
plt.plot(t, theta_meas, label="Measured θ (Noisy)", alpha=0.6)
plt.plot(t, theta_est, label="Estimated θ (Kalman)", linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.title("Kalman Filter Estimation - CTMS Model")
plt.legend()
plt.grid()
plt.show()