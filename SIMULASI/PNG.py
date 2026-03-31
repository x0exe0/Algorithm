import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

g = 9.81
T = 30/g                        #waktu target intersepsi
RT0 = np.array([-10,-5,0.5])    #posisi bola (m)
VT0 = np.array([10,0,20])       #kecepatan awal bola (m/s)

V0 = VT0+RT0/T                  #kecepatan awal capung

def dragonfly(t,x):
    rT = RT0+VT0*t+0.5*np.array([0,0,-g])*t**2  #posisi bola saat ini
    r = rT-x[0:3]                               #jarak bola dan capung

    xdot = np.zeros(6)
    xdot[0:3] = x[3:6]
    eps = 1e-8

    if abs (T-t) > eps:
        xdot[3:6] = np.array([0,0,-g])+r/((T-t)*(T-t+1))
    else:
        xdot[3:6] = np.array([0,0,-g])
    return xdot

t_frames = np.linspace(0,3,100)     #detik 0-3
x0 = np.concatenate((np.array([0,0,0]), V0))

sol = solve_ivp(dragonfly, [0,3], x0, t_eval=t_frames)
x_out = sol.y
RT_out = RT0[:, None] + VT0[:,None]*t_frames+0.5*np.array([[0],[0],[-g]])*t_frames**2

miss_dis = np.sqrt(np.sum((x_out[0:3, :]-RT_out)**2, axis=0))       #miss distance
speed_capung = np.sqrt(np.sum(x_out[3:6, :]**2, axis=0))            #kecepatan capung

min_error = np.min(miss_dis)
print(f"HASIL SIMULASI")
print(f"Miss Distance: {min_error:.4f} m")

#visulasisasi
fig = plt.figure(figsize=(14,9))
gs = fig.add_gridspec(2, 2, width_ratios=[1.5,1], height_ratios=[1,1])

ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.set_xlim([-12,12]); ax3d.set_ylim([-8,8]); ax3d.set_zlim([0,25])
ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z(m)'); ax3d.set_title('Lintasan 3D')
garis_bola, = ax3d.plot([], [], [],'r--', alpha=0.3)
titik_bola, = ax3d.plot([], [], [], 'ro', markersize=7, label='Bola')
garis_capung, = ax3d.plot ([], [], [], 'b-', alpha=0.5)
titik_capung, = ax3d.plot ([], [], [], 'b^', markersize=7, label='capung')

ax_dist = fig.add_subplot(gs[0,1])
ax_dist.set_xlim([0,3]); ax_dist.set_ylim([0,12])
ax_dist.set_label('Waktu(s)'); ax_dist.set_ylabel('||r||')
ax_dist.set_title('Error jarak vs waktu')
ax_dist.grid(True)
trace_dist, = ax_dist.plot([], [], 'k-', linewidth=2)
titik_dist, = ax_dist.plot([], [], 'ko')

ax_speed = fig.add_subplot(gs[1, 1])
ax_speed.set_xlim([0, 3]); ax_speed.set_ylim([0, 25])
ax_speed.set_xlabel('Waktu (s)'); ax_speed.set_ylabel('V (Kecepatan m/s)')
ax_speed.set_title('Kecepatan Capung vs Waktu')
ax_speed.grid(True)
trace_speed, = ax_speed.plot([], [], 'k-', linewidth=2)
titik_speed, = ax_speed.plot([], [], 'ko')
plt.tight_layout()

def update_all(num):
    garis_bola.set_data(RT_out[0, :num], RT_out[1, :num])
    garis_bola.set_3d_properties(RT_out[2, :num])
    titik_bola.set_data([RT_out[0, num]], [RT_out[1, num]])
    titik_bola.set_3d_properties([RT_out[2, num]])
    garis_capung.set_data(x_out[0, :num], x_out[1, :num])
    garis_capung.set_3d_properties(x_out[2, :num])
    titik_capung.set_data([x_out[0, num]], [x_out[1, num]])
    titik_capung.set_3d_properties([x_out[2, num]])

    trace_dist.set_data(t_frames[:num], miss_dis[:num])
    titik_dist.set_data([t_frames[num]], [miss_dis[num]])

    trace_speed.set_data(t_frames[:num], speed_capung[:num])
    titik_speed.set_data([t_frames[num]], [speed_capung[num]])

    return garis_bola, titik_bola, garis_capung, titik_capung,trace_dist, titik_dist,trace_speed,titik_speed

ani=animation.FuncAnimation(fig, update_all, frames=len(t_frames), interval=25, blit=False)
plt.show()