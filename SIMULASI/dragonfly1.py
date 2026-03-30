import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

# ==========================================
# 1. PARAMETER FISIKA & TARGET
# ==========================================
g = 9.81
RT0 = np.array([-10, -5, 0.5])
VT0 = np.array([10, 0, 20])

# KONSTANTA CPS (Tuning Parameter)
# Harus negatif agar error kecepatan mengecil (decay) seiring waktu
c = -1.2 

# Kecepatan awal CPS (Persamaan 1.47 dari buku)
V0 = VT0 - c * RT0

# ==========================================
# 2. PERSAMAAN GERAK CPS
# ==========================================
def dragonfly_cps(t, x):
    # Kinematika Bola (Target)
    rT = RT0 + VT0 * t + 0.5 * np.array([0, 0, -g]) * t**2
    vT = VT0 + np.array([0, 0, -g]) * t
    
    # Hitung Error Posisi dan Kecepatan
    r = rT - x[0:3]
    v = vT - x[3:6]
    
    xdot = np.zeros(6)
    xdot[0:3] = x[3:6]
    
    # Hukum CPS: percepatan_error = c * kecepatan_error
    # d(vT)/dt - d(V_capung)/dt = c * v
    # [-0, 0, -g] - percepatan_capung = c * v
    percepatan_capung = np.array([0, 0, -g]) - c * v
    
    xdot[3:6] = percepatan_capung
    return xdot

# ==========================================
# 3. JALANKAN SIMULASI
# ==========================================
print("Menghitung lintasan Cross-Product Steering... Mohon tunggu.")
t_frames = np.linspace(0, 4.1, 200) # Kita simulasikan sampai bola nyaris jatuh (4.1s)
x0 = np.concatenate((np.array([0, 0, 0]), V0))

sol = solve_ivp(dragonfly_cps, [0, 4.1], x0, t_eval=t_frames)
x_out = sol.y
RT_out = RT0[:, None] + VT0[:, None] * t_frames + 0.5 * np.array([[0], [0], [-g]]) * t_frames**2

# ==========================================
# 4. ANIMASI 3D (VS CODE)
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Sesuaikan batas ruang agar manuver lebar capung terlihat
ax.set_xlim([-15, 30])
ax.set_ylim([-8, 8])
ax.set_zlim([0, 25])
ax.set_xlabel('Sumbu X (m)')
ax.set_ylabel('Sumbu Y (m)')
ax.set_zlabel('Ketinggian Z (m)')
ax.set_title('Misi Rendezvous: Cross-Product Steering')

garis_bola, = ax.plot([], [], [], 'r--', alpha=0.5) 
titik_bola, = ax.plot([], [], [], 'ro', markersize=8, label='Bola (Target)') 

garis_capung, = ax.plot([], [], [], 'b-', linewidth=2, label='Capung (CPS)')
titik_capung, = ax.plot([], [], [], 'b^', markersize=8) 

ax.legend()

def update_frame(num):
    garis_bola.set_data(RT_out[0, :num], RT_out[1, :num])
    garis_bola.set_3d_properties(RT_out[2, :num])
    titik_bola.set_data([RT_out[0, num]], [RT_out[1, num]])
    titik_bola.set_3d_properties([RT_out[2, num]])
    
    garis_capung.set_data(x_out[0, :num], x_out[1, :num])
    garis_capung.set_3d_properties(x_out[2, :num])
    titik_capung.set_data([x_out[0, num]], [x_out[1, num]])
    titik_capung.set_3d_properties([x_out[2, num]])
    
    return garis_bola, titik_bola, garis_capung, titik_capung

ani = animation.FuncAnimation(fig, update_frame, frames=len(t_frames), interval=30, blit=False)
plt.show()