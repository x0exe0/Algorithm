import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import solve_ivp

g = 9.81
RT0 = np.array([-10,-5,0.5])
VT0 = np.array([10,0,20])
c = -3.0
V0 = VT0 - c * RT0 

def dragonfly_cps(t, x):
    rT = RT0+VT0*t+0.5*np.array([0, 0, -g])*t**2
    vT = VT0+np.array([0,0,-g])*t
    r = rT-x[0:3]
    v = vT-x[3:6]
    xdot = np.zeros(6)
    xdot[0:3] = x[3:6]
    
    if np.linalg.norm(r) > 0.0009:
        xdot[3:6] = np.array([0,0,-g])-c*v
    else:
        xdot[3:6] = np.array([0,0,-g])
    return xdot

t_frames = np.linspace(0,3.5,250) 
x0 = np.concatenate((np.array([0,0,0]), V0))

sol = solve_ivp(dragonfly_cps, [0,3.5], x0, t_eval=t_frames, max_step=0.01)
x_out = sol.y

RT_out = RT0[:,None]+VT0[:,None]*t_frames +0.5*np.array([[0],[0],[-g]])*t_frames**2
VT_out = VT0[:,None]+np.array([[0],[0],[-g]])*t_frames

miss_distance = np.linalg.norm(x_out[0:3, :]-RT_out, axis=0)
relative_speed = np.linalg.norm(VT_out-x_out[3:6, :], axis=0)
speed_capung = np.linalg.norm(x_out[3:6, :], axis=0)

min_error = np.min(miss_distance)
print(f"HASIL SIMULASI CPS")
print(f"Jarak meleset terkecil: {min_error:.6f} m")
print(f"Kecepatan awal capung: {speed_capung[0]:.2f} m/s")

fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1])

ax3d = fig.add_subplot(gs[:, 0], projection='3d')
ax3d.set_xlim([-15,40]); ax3d.set_ylim([-8,8]); ax3d.set_zlim([0,25])
ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
ax3d.set_title('3D Trajectory (CPS)')

garis_bola, = ax3d.plot([], [], [], 'r--', alpha=0.5)
titik_bola, = ax3d.plot([], [], [], 'ro', markersize=6, label='Bola')
garis_capung, = ax3d.plot([], [], [], 'b-', linewidth=2)
titik_capung, = ax3d.plot([], [], [], 'b^', markersize=6, label='Capung')
ax3d.legend()

ax_dist = fig.add_subplot(gs[0, 1])
ax_dist.set_xlim([0, 3.5]); ax_dist.set_ylim([0, 15])
ax_dist.set_ylabel('||r|| (m)')
ax_dist.set_title('Separation & Relative Speed Error')
ax_dist.grid(True)
trace_dist, = ax_dist.plot([], [], 'k-', linewidth=2)
titik_dist, = ax_dist.plot([], [], 'ko')

ax_rel = fig.add_subplot(gs[1,1])
ax_rel.set_xlim([0,3.5]); ax_rel.set_ylim([0,40])
ax_rel.set_ylabel('||v|| (m/s)')
ax_rel.grid(True)
trace_rel, = ax_rel.plot([],[], 'k-', linewidth=2)
titik_rel, = ax_rel.plot([],[], 'ko')

ax_speed = fig.add_subplot(gs[2, 1])
ax_speed.set_xlim([0, 3.5]); ax_speed.set_ylim([0, 40])
ax_speed.set_xlabel('Waktu (s)'); ax_speed.set_ylabel('V (m/s)')
ax_speed.set_title(' Dragonfly Required Speed')
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

    trace_dist.set_data(t_frames[:num], miss_distance[:num])
    titik_dist.set_data([t_frames[num]], [miss_distance[num]])
    trace_rel.set_data(t_frames[:num], relative_speed[:num])
    titik_rel.set_data([t_frames[num]], [relative_speed[num]])
    trace_speed.set_data(t_frames[:num], speed_capung[:num])
    titik_speed.set_data([t_frames[num]], [speed_capung[num]])

    return garis_bola, titik_bola, garis_capung, titik_capung, trace_dist, titik_dist, trace_rel, titik_rel, trace_speed, titik_speed

ani = animation.FuncAnimation(fig, update_all, frames=len(t_frames), interval=20, blit=False)
plt.show()