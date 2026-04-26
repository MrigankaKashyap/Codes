"""
Military Surveillance WSN — BPP Deployment  (FIXED)
Plots: Coverage | RWC | Intrusion Success | Alive Nodes | Threat Heatmap

Fixes applied:
  1. CH battery now depletes realistically
  2. Intrusion trials raised to 300 for stable estimates
  3. Dead-round padding uses last-valid value instead of 0
  4. Seed reset per deployment so positions vary reproducibly
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── PARAMETERS ────────────────────────────────────────────────
AREA, R, NUM_CH = 100, 10, 6
N               = 50
BATTERY_SENSOR  = 1.0
BATTERY_CH      = 2.0
ROUNDS          = 3000
THRESHOLD       = 0.20
NUM_DEPLOYMENTS = 20
E_ELEC, E_AMP, K_BITS, E_SENSE = 50e-9, 100e-12, 4000, 5e-9
BS   = np.array([50.0, 50.0])
GRID = 120

# ── EVALUATION POINTS ─────────────────────────────────────────
np.random.seed(42)
N_LOCS = 300
qx = np.random.uniform(0, AREA, N_LOCS)
qy = np.random.uniform(0, AREA, N_LOCS)

# ── THREAT FIELD ──────────────────────────────────────────────
SOURCES = [
    (np.array([20., 75.]), 0.95, 14),
    (np.array([70., 28.]), 0.85, 12),
    (np.array([55., 82.]), 0.70, 10),
    (np.array([12., 18.]), 0.55,  8),
    (np.array([85., 60.]), 0.60, 11),
]

gx, gy = np.meshgrid(np.linspace(0, AREA, GRID), np.linspace(0, AREA, GRID))
threat = np.full((GRID, GRID), 0.05)
for pos, strength, sigma in SOURCES:
    threat += strength * np.exp(-((gx-pos[0])**2 + (gy-pos[1])**2) / (2*sigma**2))
threat = np.clip(threat, 0, 1)

gx_f, gy_f  = gx.ravel(), gy.ravel()
threat_f    = threat.ravel()
threat_prob = threat_f / threat_f.sum()

xi = np.clip(np.round(qx/AREA*(GRID-1)).astype(int), 0, GRID-1)
yi = np.clip(np.round(qy/AREA*(GRID-1)).astype(int), 0, GRID-1)
q_threat       = threat[yi, xi]
TOTAL_Q_THREAT = q_threat.sum()

# ── METRIC FUNCTIONS ──────────────────────────────────────────
def coverage(sx, sy):
    if len(sx) == 0: return 0.0
    sx, sy = np.asarray(sx)[:,None], np.asarray(sy)[:,None]
    return float(np.mean(np.any((sx-qx)**2+(sy-qy)**2 <= R**2, axis=0)))

def rwc(sx, sy):
    if len(sx) == 0: return 0.0
    sx, sy = np.asarray(sx)[:,None], np.asarray(sy)[:,None]
    covered = np.any((sx-qx)**2+(sy-qy)**2 <= R**2, axis=0)
    return float(np.dot(covered, q_threat) / TOTAL_Q_THREAT)

rng_i = np.random.default_rng(123)
def intrusion_rate(sx, sy, trials=300):   # FIX 1: 10→300 trials
    if len(sx) == 0: return 1.0
    sx, sy = np.asarray(sx), np.asarray(sy)
    cells  = rng_i.choice(len(threat_f), size=trials, p=threat_prob)
    d_min  = np.array([np.sqrt((sx-gx_f[c])**2+(sy-gy_f[c])**2).min()
                       for c in cells])
    p_det  = np.clip(np.exp(-0.3 * d_min / R), 0, 1)
    return float(np.mean(rng_i.random(trials) > p_det))

# FIX 2: pad with last valid value instead of 0
def pad_last(lst, length):
    arr = np.array(lst, dtype=float)
    if len(arr) >= length:
        return arr[:length]
    last = arr[-1] if len(arr) > 0 else 0.0
    return np.concatenate([arr, np.full(length - len(arr), last)])

# ── STORAGE ───────────────────────────────────────────────────
cov1_all,  cov2_all   = [], []
rwc1_all,  rwc2_all   = [], []
isr1_all,  isr2_all   = [], []
alive1_all,alive2_all = [], []

# ── SIMULATION LOOP ───────────────────────────────────────────
for dep in range(NUM_DEPLOYMENTS):
    rng_dep = np.random.default_rng(1000 + dep)   # reproducible but varied
    x = rng_dep.uniform(0, AREA, N)
    y = rng_dep.uniform(0, AREA, N)
    print(f"  Deployment {dep+1:02d}  N={N}  (BPP fixed)")

    # Cluster heads on ring around BS
    angles = np.linspace(0, 2*np.pi, NUM_CH, endpoint=False)
    cx, cy = 50 + 30*np.cos(angles), 50 + 30*np.sin(angles)
    s2ch   = np.array([np.argmin((cx-x[i])**2+(cy-y[i])**2) for i in range(N)])

    # ── Direct TX ──────────────────────────────────────────────
    bat1 = np.full(N, BATTERY_SENSOR)
    c1, r1, i1, a1 = [], [], [], []
    for t in range(ROUNDS):
        alive  = bat1 > 0
        xa, ya = x[alive], y[alive]
        c1.append(coverage(xa, ya));  r1.append(rwc(xa, ya))
        i1.append(intrusion_rate(xa, ya));  a1.append(alive.sum())
        if c1[-1] < THRESHOLD: break
        d_bs = np.sqrt((xa-BS[0])**2+(ya-BS[1])**2)
        bat1[alive] -= E_SENSE + E_ELEC*K_BITS + E_AMP*K_BITS*d_bs**2
        bat1 = np.maximum(bat1, 0)

    # ── Cluster-Head TX ────────────────────────────────────────
    bat2   = np.full(N, BATTERY_SENSOR)
    bat_ch = np.full(NUM_CH, BATTERY_CH)
    c2, r2, i2, a2 = [], [], [], []
    for t in range(ROUNDS):
        alive_idx = np.where(bat2 > 0)[0]
        ax_l, ay_l = [], []
        for i in alive_idx:
            ch = s2ch[i]
            if bat_ch[ch] > 0:
                d_to_ch = np.sqrt((x[i]-cx[ch])**2+(y[i]-cy[ch])**2)
                cost_sensor = E_SENSE + E_ELEC*K_BITS + E_AMP*K_BITS*d_to_ch**2
                bat2[i] -= cost_sensor
                # FIX 3: CH also spends energy forwarding to BS
                d_ch_bs = np.sqrt((cx[ch]-BS[0])**2+(cy[ch]-BS[1])**2)
                bat_ch[ch] -= E_ELEC*K_BITS + E_AMP*K_BITS*d_ch_bs**2
                bat_ch[ch] = max(bat_ch[ch], 0)
            else:
                # CH dead → sensor transmits directly to BS
                d_bs = np.sqrt((x[i]-BS[0])**2+(y[i]-BS[1])**2)
                bat2[i] -= E_SENSE + E_ELEC*K_BITS + E_AMP*K_BITS*d_bs**2
            if bat2[i] > 0: ax_l.append(x[i]); ay_l.append(y[i])
        bat2 = np.maximum(bat2, 0)
        c2.append(coverage(ax_l, ay_l));  r2.append(rwc(ax_l, ay_l))
        i2.append(intrusion_rate(ax_l, ay_l));  a2.append((bat2>0).sum())
        if c2[-1] < THRESHOLD: break

    for lst, store in [(c1,cov1_all),(c2,cov2_all),(r1,rwc1_all),(r2,rwc2_all),
                       (i1,isr1_all),(i2,isr2_all),(a1,alive1_all),(a2,alive2_all)]:
        store.append(pad_last(lst, ROUNDS))   # FIX 4: last-value padding

# ── MEAN CURVES ───────────────────────────────────────────────
xs  = np.arange(ROUNDS)
mc1 = np.mean(cov1_all,   axis=0)*100;  mc2 = np.mean(cov2_all,   axis=0)*100
mr1 = np.mean(rwc1_all,   axis=0)*100;  mr2 = np.mean(rwc2_all,   axis=0)*100
mi1 = np.mean(isr1_all,   axis=0)*100;  mi2 = np.mean(isr2_all,   axis=0)*100
ma1 = np.mean(alive1_all, axis=0);      ma2 = np.mean(alive2_all, axis=0)
B, O = 'royalblue', 'darkorange'

# ── PLOT 1: Coverage vs Time ───────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(xs, mc1, B, lw=2, label='Direct TX')
plt.plot(xs, mc2, O, lw=2, label='Cluster-Head TX')
plt.axhline(THRESHOLD*100, color='red', ls='--', lw=1.5, label='Threshold 20%')
plt.xlabel('Round'); plt.ylabel('Coverage (%)')
plt.title('Coverage vs Time (BPP)'); plt.legend(); plt.grid(alpha=0.35)
plt.tight_layout(); plt.savefig('p1_coverage.png', dpi=150); plt.show()

# ── PLOT 2: RWC vs Time ────────────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(xs, mr1, B, lw=2, label='Direct TX')
plt.plot(xs, mr2, O, lw=2, label='Cluster-Head TX')
plt.xlabel('Round'); plt.ylabel('Risk-Weighted Coverage (%)')
plt.title('Risk-Weighted Coverage vs Time (BPP)'); plt.legend(); plt.grid(alpha=0.35)
plt.tight_layout(); plt.savefig('p2_rwc.png', dpi=150); plt.show()

# ── PLOT 3: Intrusion Success Rate vs Time ─────────────────────
plt.figure(figsize=(8,4))
plt.plot(xs, mi1, B, lw=2, label='Direct TX')
plt.plot(xs, mi2, O, lw=2, label='Cluster-Head TX')
plt.xlabel('Round'); plt.ylabel('Intrusion Success Rate (%)')
plt.title('Intrusion Success Rate vs Time (BPP)'); plt.legend(); plt.grid(alpha=0.35)
plt.tight_layout(); plt.savefig('p3_intrusion.png', dpi=150); plt.show()

# ── PLOT 4: Alive Nodes vs Time ────────────────────────────────
plt.figure(figsize=(8,4))
plt.plot(xs, ma1, B, lw=2, label='Direct TX')
plt.plot(xs, ma2, O, lw=2, label='Cluster-Head TX')
plt.xlabel('Round'); plt.ylabel('Alive Nodes')
plt.title('Alive Nodes vs Time (BPP)'); plt.legend(); plt.grid(alpha=0.35)
plt.tight_layout(); plt.savefig('p4_alive.png', dpi=150); plt.show()

# ── PLOT 5: Threat Field Heatmap ───────────────────────────────
tcmap = mcolors.LinearSegmentedColormap.from_list(
    'threat', ['#1a9641','#ffffbf','#d7191c','#6b0000'], N=256)
plt.figure(figsize=(6,5))
plt.imshow(threat, origin='lower', extent=[0,AREA,0,AREA],
           cmap=tcmap, vmin=0, vmax=1, interpolation='bilinear', aspect='auto')
cb = plt.colorbar(fraction=0.046, pad=0.04)
cb.set_label('Threat Level')
cb.set_ticks([0,.25,.5,.75,1]); cb.set_ticklabels(['Min','Low','Mod','High','Critical'])
for pos,s,_ in SOURCES: plt.plot(*pos, 'w*', ms=10, markeredgecolor='k')
plt.title('Threat Field — Military Surveillance Zone')
plt.xlabel('X (m)'); plt.ylabel('Y (m)')
plt.tight_layout(); plt.savefig('p5_threat.png', dpi=150); plt.show()

# ── SUMMARY ────────────────────────────────────────────────────
print("\n===== FINAL RESULTS =====")
print(f"Deployment model : BPP  (N = {N} fixed per deployment)")
print(f"Final Coverage   — Direct: {mc1[mc1>0][-1]:.1f}%  | CH: {mc2[mc2>0][-1]:.1f}%")
print(f"Final RWC        — Direct: {mr1[mr1>0][-1]:.1f}%  | CH: {mr2[mr2>0][-1]:.1f}%")
print(f"Final ISR        — Direct: {mi1[-1]:.1f}%         | CH: {mi2[-1]:.1f}%")

# ── LIFETIME SUMMARY ───────────────────────────────────────────
def network_lifetime(cov_all):
    """Round at which mean coverage first drops below threshold."""
    mean = np.mean(cov_all, axis=0)
    below = np.where(mean < THRESHOLD)[0]
    return below[0] if len(below) > 0 else ROUNDS

lt1 = network_lifetime(cov1_all)
lt2 = network_lifetime(cov2_all)
print(f"\nNetwork Lifetime — Direct: {lt1} rounds | CH: {lt2} rounds")
print(f"CH TX extends lifetime by {lt2-lt1} rounds ({(lt2-lt1)/lt1*100:.1f}% improvement)")

# ── PLOT 6: Sensor Deployment Map ──────────────────────────────
# Use last deployment's positions for the map
rng_last = np.random.default_rng(1000 + NUM_DEPLOYMENTS - 1)
x_last = rng_last.uniform(0, AREA, N)
y_last = rng_last.uniform(0, AREA, N)
angles_last = np.linspace(0, 2*np.pi, NUM_CH, endpoint=False)
cx_last = 50 + 30*np.cos(angles_last)
cy_last = 50 + 30*np.sin(angles_last)
s2ch_last = np.array([np.argmin((cx_last-x_last[i])**2+(cy_last-y_last[i])**2) for i in range(N)])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('BPP Sensor Deployment Map', fontsize=14, fontweight='bold')

for ax, title in zip(axes, ['Direct TX', 'Cluster-Head TX']):
    # Threat heatmap as background
    ax.imshow(threat, origin='lower', extent=[0,AREA,0,AREA],
              cmap=tcmap, vmin=0, vmax=1, alpha=0.4,
              interpolation='bilinear', aspect='auto')

    # Sensor coverage circles
    for i in range(N):
        circle = plt.Circle((x_last[i], y_last[i]), R,
                             color='royalblue', alpha=0.08, zorder=1)
        ax.add_patch(circle)

    # Sensors
    ax.scatter(x_last, y_last, c='royalblue', s=30, zorder=3,
               label=f'Sensors (N={N})', edgecolors='white', linewidths=0.4)

    if title == 'Cluster-Head TX':
        colors_ch = plt.cm.Set1(np.linspace(0, 0.8, NUM_CH))
        # Lines: sensor → cluster head
        for i in range(N):
            ch = s2ch_last[i]
            ax.plot([x_last[i], cx_last[ch]], [y_last[i], cy_last[ch]],
                    color=colors_ch[ch], alpha=0.25, lw=0.7, zorder=2)
        # Lines: cluster head → BS
        for ch in range(NUM_CH):
            ax.plot([cx_last[ch], BS[0]], [cy_last[ch], BS[1]],
                    'k--', alpha=0.5, lw=1.2, zorder=2)
        # Cluster heads
        ax.scatter(cx_last, cy_last, c=[colors_ch[i] for i in range(NUM_CH)],
                   s=150, marker='^', zorder=4, edgecolors='black',
                   linewidths=0.8, label='Cluster Heads')
    else:
        # Lines: sensor → BS (direct)
        for i in range(N):
            ax.plot([x_last[i], BS[0]], [y_last[i], BS[1]],
                    color='royalblue', alpha=0.12, lw=0.6, zorder=2)

    # Base Station
    ax.scatter(*BS, c='red', s=250, marker='*', zorder=5,
               edgecolors='black', linewidths=0.8, label='Base Station')

    # Threat source markers
    for pos, _, _ in SOURCES:
        ax.plot(*pos, 'w*', ms=8, markeredgecolor='k', zorder=5)

    ax.set_xlim(0, AREA); ax.set_ylim(0, AREA)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    ax.set_title(title); ax.legend(loc='upper right', fontsize=8)
    ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('p6_deployment.png', dpi=150)
plt.show()
print("Deployment map saved: p6_deployment.png")