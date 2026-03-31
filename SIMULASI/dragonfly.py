import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# PARAMETER & TARGET
g = 9.81                         # Gravitasi (m/s^2)
T = 30 / g                       # Waktu target intersepsi (sekitar 3.058 s)
RT0 = np.array([-10, -5, 0.5])   # Posisi awal bola (m)
VT0 = np.array([10, 0, 20])      # Kecepatan awal bola (m/s)

# Menghitung kecepatan awal capung, syarat PNG
V0 = VT0 + RT0 / T

# PERSAMAAN GERAK (ODE)
def dragonfly_ode(t, x):
    # Posisi bola saat ini (Target)
    rT = RT0 + VT0 * t + 0.5 * np.array([0, 0, -g]) * t**2
    
    # Vektor jarak antara bola dan capung
    r = rT - x[0:3]
    xdot = np.zeros(6)
    xdot[0:3] = x[3:6]  # Turunan posisi adalah kecepatan
    eps = 1e-8

    # Menghindari pembagian dengan nol saat t mendekati T
    if abs(T - t) > eps:
        xdot[3:6] = np.array([0, 0, -g]) + r / ((T - t) * (T - t + 1))
    else:
        xdot[3:6] = np.array([0, 0, -g])
        
    return xdot

t_frames = np.linspace(0, 3, 94) 
x0 = np.concatenate((np.array([0, 0, 0]), V0))

# solve
sol = solve_ivp(dragonfly_ode, [0, 3], x0, t_eval=t_frames)
x_out = sol.y
# Hitung ulang lintasan bola di semua frame untuk plotting
RT_out = RT0[:, None] + VT0[:, None] * t_frames + 0.5 * np.array([[0], [0], [-g]]) * t_frames**2

#Analisis Miss Distance (Jarak Meleset)
jarak_error = np.sqrt(np.sum((x_out[0:3, :] - RT_out)**2, axis=0))
min_error = np.min(jarak_error)
t_min = t_frames[np.argmin(jarak_error)]
print(f"HASIL SIMULASI")
print(f"Jarak meleset terkecil : {min_error:.4f} meter")
print(f"Waktu tabrakan terbaik : {t_min:.3f} detik\n")
print("...")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Batasan ruang 3D (agar kamera statis)
ax.set_xlim([-12, 12])
ax.set_ylim([-8, 8])
ax.set_zlim([0, 25])
ax.set_xlabel('Sumbu X (m)')
ax.set_ylabel('Sumbu Y (m)')
ax.set_zlabel('Ketinggian Z (m)')
ax.set_title('Simulasi Intersepsi Capung vs Bola Kriket (PNG)')

# Inisialisasi garis dan titik
garis_bola, = ax.plot([], [], [], 'r--', alpha=0.5) 
titik_bola, = ax.plot([], [], [], 'ro', markersize=8, label='Bola (Target)') 

garis_capung, = ax.plot([], [], [], 'b-', alpha=0.5)
titik_capung, = ax.plot([], [], [], 'b^', markersize=8, label='Capung (Interceptor)') 

ax.legend()

# Fungsi pembaruan frame animasi
def update_frame(num):
    # Update Bola
    garis_bola.set_data(RT_out[0, :num], RT_out[1, :num])
    garis_bola.set_3d_properties(RT_out[2, :num])
    titik_bola.set_data([RT_out[0, num]], [RT_out[1, num]])
    titik_bola.set_3d_properties([RT_out[2, num]])
    
    # Update Capung
    garis_capung.set_data(x_out[0, :num], x_out[1, :num])
    garis_capung.set_3d_properties(x_out[2, :num])
    titik_capung.set_data([x_out[0, num]], [x_out[1, num]])
    titik_capung.set_3d_properties([x_out[2, num]])
    
    # Return harus mengembalikan semua objek yang diubah
    return garis_bola, titik_bola, garis_capung, titik_capung

# Generate Animasi
ani = animation.FuncAnimation(fig, update_frame, frames=len(t_frames), interval=30, blit=False)

# Memunculkan popup window di VS Code
plt.show()