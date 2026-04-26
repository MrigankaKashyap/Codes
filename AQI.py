import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from scipy.spatial import cKDTree

# ─── PARAMETERS ──────────────────────────────────────────────────────────────
AREA, R          = 100, 10
NUM_CH           = 6
LAMBDA           = 0.005
CH_RING_RADIUS   = 25
BAT_S, BAT_CH    = 1.0, 2.0
ROUNDS           = 3000
THRESHOLD        = 0.2
NUM_DEP          = 20
MIN_SENSORS_RMSE = 5
E_ELEC, E_AMP    = 50e-9, 100e-12
K, E_S           = 4000, 5e-9
E_DA             = 5e-9          # data-aggregation energy coefficient (J/bit)
AGG              = 0.3           # compression ratio: aggregated pkt = AGG*K bits
GRID             = 100
BS               = np.array([50., 50.])
SNAP_R           = [600, 1200, 1800, 2400]
COLORS           = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4"]
AQI_BRK          = [0,50,100,150,200,300,500]
AQI_LAB          = ["Good","Moderate","Unhealthy (SG)","Unhealthy","Very Unhealthy","Hazardous"]
AQI_COL          = ["#00e400","#ffff00","#ff7e00","#ff0000","#8f3f97","#7e0023"]

# ─── AQI FIELD ───────────────────────────────────────────────────────────────
SOURCES = [(20,75,280,16),(70,28,220,13),(55,82,160,11),(12,18,120,9)]
gx, gy  = np.meshgrid(np.linspace(0,AREA,GRID), np.linspace(0,AREA,GRID))
aqi     = np.full((GRID,GRID), 30.)
for x0,y0,st,sg in SOURCES:
    aqi += st * np.exp(-((gx-x0)**2+(gy-y0)**2)/(2*sg**2))
rng2 = np.random.default_rng(7)
for _ in range(10):
    fx,fy = rng2.uniform(.4,2.2), rng2.uniform(.4,2.2)
    ph,amp = rng2.uniform(0,2*np.pi), rng2.uniform(10,30)
    aqi += amp*np.sin(2*np.pi*fx*gx/AREA+ph)*np.cos(2*np.pi*fy*gy/AREA)
aqi = np.clip(aqi, 0, None)

aqi_cmap = mcolors.LinearSegmentedColormap.from_list(
    "aqi", list(zip(np.linspace(0,1,len(AQI_COL)), AQI_COL)), N=512)
aqi_norm = mcolors.Normalize(0, 500)
aqi_lbl  = lambda v: next((AQI_LAB[i] for i in range(len(AQI_BRK)-1)
                            if v<=AQI_BRK[i+1]), AQI_LAB[-1])

# ─── HELPERS ─────────────────────────────────────────────────────────────────
grid_pts = np.column_stack((gx.ravel(), gy.ravel()))

def coverage(sx, sy, qx, qy):
    if len(sx)==0: return 0.
    d2 = (sx[:,None]-qx)**2+(sy[:,None]-qy)**2
    return float(np.mean(np.any(d2<=R**2, axis=0)))

def sensor_vals(sx, sy):
    xi = np.clip((sx/AREA*(GRID-1)).astype(int), 0, GRID-1)
    yi = np.clip((sy/AREA*(GRID-1)).astype(int), 0, GRID-1)
    return aqi[yi, xi]

def estimate_aqi(sx, sy):
    if len(sx)==0: return np.full((GRID,GRID), np.nan)
    _, idx = cKDTree(np.column_stack((sx,sy))).query(grid_pts)
    return sensor_vals(sx,sy)[idx].reshape(GRID,GRID)

def calc_rmse(sx, sy):
    if len(sx) < MIN_SENSORS_RMSE: return np.nan
    est = estimate_aqi(sx, sy)
    m = ~np.isnan(est)
    return float(np.sqrt(np.mean((aqi[m]-est[m])**2))) if m.any() else np.nan

def cov_map(sx, sy):
    if len(sx)==0: return np.full((GRID,GRID), np.nan)
    d2 = (sx[:,None]-gx.ravel())**2+(sy[:,None]-gy.ravel())**2
    return np.where(np.any(d2<=R**2,axis=0).reshape(GRID,GRID), aqi, np.nan)

# ─── CH PLACEMENT ────────────────────────────────────────────────────────────
ang     = np.linspace(0, 2*np.pi, NUM_CH, endpoint=False)
cx, cy  = BS[0]+CH_RING_RADIUS*np.cos(ang), BS[1]+CH_RING_RADIUS*np.sin(ang)
d_ch_bs = np.sqrt((cx-BS[0])**2+(cy-BS[1])**2)

# ─── STORAGE ─────────────────────────────────────────────────────────────────
C1,C2,A1,A2,B1,B2,R1,R2,T1s,T2s,C0 = [[] for _ in range(11)]
sd, sc = {}, {}
last   = {}

# ─── SIMULATION ──────────────────────────────────────────────────────────────
for dep in range(NUM_DEP):
    is_last = dep == NUM_DEP-1
    N  = np.random.poisson(LAMBDA*AREA**2)
    x  = np.random.uniform(0,AREA,N)
    y  = np.random.uniform(0,AREA,N)
    dists = np.sqrt((x[:,None]-cx)**2+(y[:,None]-cy)**2)
    s2ch  = np.argmin(dists, axis=1)
    print(f"Dep {dep+1:2d}: N={N}", flush=True)
    if is_last: last = dict(x=x, y=y, s2ch=s2ch)

    qx = np.random.uniform(0, AREA, 300)
    qy = np.random.uniform(0, AREA, 300)

    d_s_bs = np.sqrt((x-BS[0])**2+(y-BS[1])**2)
    d_s_ch = dists[np.arange(N), s2ch]

    # ── Direct-to-BS ─────────────────────────────────────────────────────────
    bat      = np.full(N, BAT_S)
    E_direct = E_S + E_ELEC*K + E_AMP*K*(d_s_bs**2)
    c1,a1,b1,r1 = [],[],[],[]
    T1 = ROUNDS
    for t in range(ROUNDS):
        alive = bat > 0
        sx,sy = x[alive], y[alive]
        cov = coverage(sx, sy, qx, qy)
        c1.append(cov); a1.append(int(alive.sum()))
        b1.append(float(np.mean(bat)))
        r1.append(calc_rmse(sx, sy))
        if is_last and t in SNAP_R: sd[t] = cov_map(sx,sy)
        if cov < THRESHOLD: T1=t; break
        bat[alive] -= E_direct[alive]
        bat = np.maximum(bat, 0)
    if is_last:
        for s in SNAP_R: sd.setdefault(s, cov_map(np.array([]),np.array([])))

    # ── Cluster-Head ─────────────────────────────────────────────────────────
    bat_s  = np.full(N, BAT_S)
    bat_ch = np.full(NUM_CH, BAT_CH)
    E_toch = E_S + E_ELEC*K + E_AMP*K*(d_s_ch**2)
    E_tobs = E_S + E_ELEC*K + E_AMP*K*(d_s_bs**2)
    c2,a2,b2,r2 = [],[],[],[]
    T2 = ROUNDS
    for t in range(ROUNDS):
        alive  = bat_s > 0
        ch_ok  = bat_ch > 0
        use_ch = alive & ch_ok[s2ch]
        cost   = np.where(use_ch, E_toch, E_tobs)
        bat_s  = np.maximum(bat_s - cost*alive, 0)

        for j in range(NUM_CH):
            if not ch_ok[j]: continue
            n = int(np.sum(alive & (s2ch==j)))
            if n:
                # FIXED energy model (first-order radio):
                #   receive:    n * E_ELEC * K
                #   aggregate:  n * E_DA   * K   (scales with cluster size)
                #   transmit:   E_ELEC * (AGG*K) + E_AMP * (AGG*K) * d^2
                bat_ch[j] = max(0., bat_ch[j]
                                - E_ELEC * K * n
                                - E_DA   * K * n
                                - E_ELEC * AGG * K
                                - E_AMP  * AGG * K * (d_ch_bs[j]**2))

        active = bat_s > 0
        sx, sy = x[active], y[active]
        cov = coverage(sx, sy, qx, qy)
        c2.append(cov); a2.append(int(active.sum()))
        b2.append(float(np.mean(bat_s)))
        r2.append(calc_rmse(sx, sy))
        if is_last and t in SNAP_R: sc[t] = cov_map(sx,sy)
        if cov < THRESHOLD: T2=t; break
    if is_last:
        for s in SNAP_R: sc.setdefault(s, cov_map(np.array([]),np.array([])))

    pf = lambda a: np.pad(np.array(a,float),(0,ROUNDS-len(a)),constant_values=np.nan)
    pi = lambda a: np.pad(np.array(a,int),  (0,ROUNDS-len(a)))
    C1.append(pf(c1)); C2.append(pf(c2))
    A1.append(pi(a1)); A2.append(pi(a2))
    B1.append(pf(b1)); B2.append(pf(b2))
    R1.append(pf(r1)); R2.append(pf(r2))
    T1s.append(T1); T2s.append(T2); C0.append(c1[0])

# ─── AVERAGES ────────────────────────────────────────────────────────────────
avg = lambda L: np.nanmean(L, axis=0)
c1m,c2m = avg(C1),avg(C2)
a1m,a2m = avg(A1),avg(A2)
b1m,b2m = avg(B1),avg(B2)
r1m,r2m = avg(R1),avg(R2)

t1_mean, t2_mean = np.mean(T1s), np.mean(T2s)
print(f"\n===== RESULTS ({NUM_DEP} deployments) =====")
print(f"Avg Initial Coverage : {np.mean(C0)*100:.1f}%")
print(f"Avg Lifetime Direct  : {t1_mean:.0f} rounds")
print(f"Avg Lifetime CH      : {t2_mean:.0f} rounds")
print(f"Improvement          : {(t2_mean-t1_mean)/t1_mean*100:.1f}%")

# ─── PLOT HELPERS ────────────────────────────────────────────────────────────
def heatmap_panel(ax, snap, t, show_ch=False):
    ax.imshow(aqi, origin='lower', extent=[0,AREA,0,AREA],
              cmap=aqi_cmap, norm=aqi_norm, aspect='equal', alpha=0.18)
    ax.imshow(np.where(~np.isnan(snap),snap,np.nan), origin='lower',
              extent=[0,AREA,0,AREA], cmap=aqi_cmap, norm=aqi_norm, aspect='equal', alpha=0.9)
    ax.imshow(np.where(np.isnan(snap),1.,np.nan), origin='lower',
              extent=[0,AREA,0,AREA], cmap='Greys', vmin=0, vmax=1, aspect='equal', alpha=0.25)
    for x0,y0,*_ in SOURCES:
        ax.scatter(x0,y0, s=22, c='white', edgecolors='k', zorder=5, lw=0.6)
    if show_ch:
        ax.scatter(cx,cy, s=70, marker='^', c='cyan', edgecolors='k', zorder=6, lw=0.8)
        ax.scatter(*BS, s=120, marker='*', c='white', edgecolors='k', zorder=7)
        ax.add_patch(Circle(BS,CH_RING_RADIUS,fill=False,edgecolor='white',
                            lw=0.8,ls='--',alpha=0.5))
    cv   = np.mean(~np.isnan(snap))*100
    vals = snap[~np.isnan(snap)]
    ma   = float(np.mean(vals)) if vals.size else 0.
    ax.set_title(f"Round {t}  |  Cov {cv:.1f}%  |  AQI {ma:.0f} ({aqi_lbl(ma)})",
                 fontsize=8.5, fontweight='bold')
    ax.set_xlabel("X (m)",fontsize=8); ax.set_ylabel("Y (m)",fontsize=8)
    ax.tick_params(labelsize=7)

def add_cbar(fig, axs):
    sm = plt.cm.ScalarMappable(cmap=aqi_cmap, norm=aqi_norm); sm.set_array([])
    cb = fig.colorbar(sm, ax=axs, shrink=0.85, pad=0.02, aspect=30)
    cb.set_label("AQI",fontsize=10); cb.set_ticks(AQI_BRK[:-1]); cb.ax.tick_params(labelsize=8)

# ─── PLOT 1: True AQI Field ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,6))
im = ax.imshow(aqi, origin='lower', extent=[0,AREA,0,AREA],
               cmap=aqi_cmap, norm=aqi_norm, aspect='equal')
for x0,y0,st,_ in SOURCES:
    ax.scatter(x0,y0, s=70, c='white', edgecolors='k', zorder=5)
    ax.annotate(f"  {st} AQI",(x0,y0), color='white', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.2',fc='k',alpha=0.45))
cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("AQI",fontsize=10); cb.set_ticks(AQI_BRK[:-1])
ax.set_title("True AQI Field — Spatial Pollution Profile", fontweight='bold')
ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
plt.savefig("plot1_aqi_field.png", dpi=150, bbox_inches='tight'); plt.show()

# ─── PLOT 2: Sensor & CH Distribution ────────────────────────────────────────
lx, ly, ls = last['x'], last['y'], last['s2ch']
fig, axes = plt.subplots(1,2, figsize=(14,6))

ax = axes[0]
for j in range(NUM_CH):
    m = ls==j
    for i in np.where(m)[0]:
        ax.plot([lx[i],cx[j]],[ly[i],cy[j]], color=COLORS[j], lw=0.35, alpha=0.25)
    ax.scatter(lx[m],ly[m], s=20, color=COLORS[j], alpha=0.85, zorder=4)
    ax.scatter(cx[j],cy[j], s=150, marker='^', color=COLORS[j],
               edgecolors='k', lw=1, zorder=6)
    ax.annotate(f"CH{j+1}",(cx[j],cy[j]), xytext=(4,4), textcoords='offset points',
                fontsize=7, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.15',fc='k',alpha=0.55), color='white')
ax.scatter(*BS, s=250, marker='*', c='gold', edgecolors='k', lw=1, zorder=7)
ax.annotate("BS", BS, xytext=(5,5), textcoords='offset points',
            fontsize=9, fontweight='bold', color='goldenrod')
ax.add_patch(Circle(BS,CH_RING_RADIUS,fill=False,edgecolor='k',lw=1.3,ls='--'))
handles = [mpatches.Patch(color=COLORS[j], label=f"Cluster {j+1}  (n={(ls==j).sum()})")
           for j in range(NUM_CH)]
handles += [plt.scatter([],[],s=150,marker='^',c='grey',edgecolors='k',label='Cluster Head'),
            plt.scatter([],[],s=200,marker='*',c='gold',edgecolors='k',label='Base Station')]
ax.legend(handles=handles, fontsize=7.5, loc='upper right', ncol=2, framealpha=0.85)
ax.set(xlim=(0,AREA), ylim=(0,AREA), aspect='equal',
       title=f"PPP Sensor Deployment  —  N = {len(lx)}\nColoured by Nearest CH",
       xlabel="X (m)", ylabel="Y (m)")
ax.grid(alpha=0.25)

ax = axes[1]
ax.imshow(aqi, origin='lower', extent=[0,AREA,0,AREA],
          cmap=aqi_cmap, norm=aqi_norm, aspect='equal', alpha=0.4)
for j in range(NUM_CH):
    for i in np.where(ls==j)[0]:
        ax.add_patch(Circle((lx[i],ly[i]),R, fill=True,
                             facecolor=COLORS[j], alpha=0.10, lw=0))
    ax.scatter(lx[ls==j], ly[ls==j], s=14, color=COLORS[j], alpha=0.9, zorder=5)
    ax.scatter(cx[j],cy[j], s=160, marker='^', color=COLORS[j],
               edgecolors='white', lw=1.1, zorder=7)
    ax.annotate(f"CH{j+1}",(cx[j],cy[j]), xytext=(4,4), textcoords='offset points',
                fontsize=7, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.15',fc='k',alpha=0.55))
ax.scatter(*BS, s=250, marker='*', c='gold', edgecolors='white', lw=1, zorder=8)
ax.add_patch(Circle(BS,CH_RING_RADIUS,fill=False,edgecolor='white',lw=1.4,ls='--',alpha=0.7))
for x0,y0,st,_ in SOURCES:
    ax.scatter(x0,y0, s=50, c='white', edgecolors='k', zorder=6)
    ax.annotate(f" {st}",(x0,y0), color='white', fontsize=7,
                bbox=dict(boxstyle='round,pad=0.15',fc='k',alpha=0.5))
sm = plt.cm.ScalarMappable(cmap=aqi_cmap, norm=aqi_norm); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("AQI",fontsize=10); cb.set_ticks(AQI_BRK[:-1]); cb.ax.tick_params(labelsize=8)
ax.set(xlim=(0,AREA), ylim=(0,AREA), aspect='equal',
       title="CH Ring on AQI Field\nShaded = Sensor Coverage Discs (R=10 m)",
       xlabel="X (m)", ylabel="Y (m)")
ax.grid(alpha=0.18)
plt.suptitle("Sensor Node & Cluster Head Distribution — Final Deployment",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("plot2_distribution.png", dpi=150, bbox_inches='tight'); plt.show()

# ─── PLOT 3 & 4: Snapshot Heatmaps ───────────────────────────────────────────
for title, snaps, show_ch, fname in [
    ("Direct-to-BS: Sensed AQI vs Coverage",  sd, False, "plot3_direct.png"),
    ("Cluster-Head: Sensed AQI vs Coverage",  sc, True,  "plot4_ch.png"),
]:
    fig = plt.figure(figsize=(13,10))
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.98)
    axs = [fig.add_subplot(2,2,i+1) for i in range(4)]
    for ax,t in zip(axs,SNAP_R): heatmap_panel(ax, snaps[t], t, show_ch)
    fig.subplots_adjust(left=0.07,right=0.88,top=0.92,bottom=0.07,hspace=0.35,wspace=0.30)
    add_cbar(fig, axs)
    plt.savefig(fname, dpi=150, bbox_inches='tight'); plt.show()

# ─── PLOT 5: Coverage / Alive / Battery / RMSE ───────────────────────────────
# FIX (crash): axes come from zip(axes.ravel(), ...) so do NOT include them
# in the panels tuples — each tuple has exactly 6 elements.
fig, axes = plt.subplots(2,2, figsize=(14,9))
fig.suptitle(f"Network Metrics — Avg over {NUM_DEP} Deployments",
             fontsize=12, fontweight='bold')

panels = [
    (c1m*100, c2m*100, "Coverage (%)",     "Coverage vs Rounds",    THRESHOLD*100, True),
    (a1m,     a2m,     "Alive Nodes",      "Alive Nodes vs Rounds", None,          True),
    (b1m,     b2m,     "Avg Battery (J)",  "Battery Depletion",     None,          True),
    (r1m,     r2m,     "RMSE (AQI units)", "AQI Estimation RMSE",   None,          False),
]
for ax, (d1, d2, ylabel, title, thresh, add_vlines) in zip(axes.ravel(), panels):
    ax.plot(d1, label="Direct", lw=1.5, color='steelblue')
    ax.plot(d2, label="CH",     lw=1.5, color='darkorange')
    if thresh is not None:
        ax.axhline(thresh, ls='--', color='k', lw=1, label=f"{thresh:.0f}% threshold")
    if add_vlines:
        ax.axvline(t1_mean, ls='--', color='steelblue',  alpha=0.55, lw=1.2,
                   label=f"Direct end ({t1_mean:.0f})")
        ax.axvline(t2_mean, ls='--', color='darkorange',  alpha=0.55, lw=1.2,
                   label=f"CH end ({t2_mean:.0f})")
    ax.set(title=title, ylabel=ylabel, xlabel="Rounds")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("plot5_metrics.png", dpi=150, bbox_inches='tight'); plt.show()
print(f"\nDone — all plots saved.")