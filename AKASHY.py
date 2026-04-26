"""
Real-life scenario: Temperature monitoring inside a greenhouse
- Sensors placed in a grid across the greenhouse floor
- Each sensor reads temperature in its surrounding area
- If a point falls within a sensor's radius → it is "sensed"
- Temperature varies across the greenhouse (warmer near center/heater)
- Regions: Low / Medium / High temperature zones
- Find sensor closest to the target temperature T0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

# ──────────────────────────────────────────────
#  CONFIGURATION  (change these freely)
# ──────────────────────────────────────────────
AREA_W      = 100        # greenhouse width  (metres)
AREA_H      = 100         # greenhouse height (metres)
GRID_COLS   = 3          # sensors along X
GRID_ROWS   = 3          # sensors along Y
SENS_RADIUS = 15         # sensing radius in metres
T0          = 30.0       # target temperature (°C) — what we are looking for

# Temperature field parameters (simulated greenhouse heat map)
T_MIN       = 16.0       # coolest corner temperature
T_MAX       = 38.0       # hottest spot temperature
HOT_X       = 50         # hot-spot X (centre of greenhouse heater)
HOT_Y       = 50         # hot-spot Y

RESOLUTION  = 300        # grid resolution for background heatmap

# ──────────────────────────────────────────────
#  STEP 1 — Place sensors in a uniform grid
# ──────────────────────────────────────────────
xs = np.linspace(AREA_W / (GRID_COLS + 1), AREA_W * GRID_COLS / (GRID_COLS + 1), GRID_COLS)
ys = np.linspace(AREA_H / (GRID_ROWS + 1), AREA_H * GRID_ROWS / (GRID_ROWS + 1), GRID_ROWS)
gx, gy      = np.meshgrid(xs, ys)
sensor_x    = gx.ravel()
sensor_y    = gy.ravel()
N_SENSORS   = len(sensor_x)

# ──────────────────────────────────────────────
#  STEP 2 — Temperature field across the area
#           Gaussian heat source (like a heater)
# ──────────────────────────────────────────────
px = np.linspace(0, AREA_W, RESOLUTION)
py = np.linspace(0, AREA_H, RESOLUTION)
PX, PY = np.meshgrid(px, py)

dist_to_hot = np.sqrt((PX - HOT_X)**2 + (PY - HOT_Y)**2)
T_field     = T_MIN + (T_MAX - T_MIN) * np.exp(-dist_to_hot**2 / (2 * 35**2))

# ──────────────────────────────────────────────
#  STEP 3 — Each sensor reads the temperature
#           at its own location
# ──────────────────────────────────────────────
def read_temp(x, y):
    d = np.sqrt((x - HOT_X)**2 + (y - HOT_Y)**2)
    return T_MIN + (T_MAX - T_MIN) * np.exp(-d**2 / (2 * 35**2))

sensor_temps = np.array([read_temp(x, y) for x, y in zip(sensor_x, sensor_y)])

# ──────────────────────────────────────────────
#  STEP 4 — Classify each sensor into a region
#           based on its temperature reading
# ──────────────────────────────────────────────
LOW_MAX  = T_MIN + (T_MAX - T_MIN) / 3        # boundary Low / Medium
HIGH_MIN = T_MIN + 2 * (T_MAX - T_MIN) / 3   # boundary Medium / High

def classify(t):
    if t < LOW_MAX:   return "Low"
    if t < HIGH_MIN:  return "Medium"
    return "High"

sensor_regions = [classify(t) for t in sensor_temps]

# ──────────────────────────────────────────────
#  STEP 5 — Coverage map
#           A point is SENSED if at least one
#           sensor is within SENS_RADIUS of it
# ──────────────────────────────────────────────
covered = np.zeros((RESOLUTION, RESOLUTION), dtype=bool)
for sx, sy in zip(sensor_x, sensor_y):
    d2 = (PX - sx)**2 + (PY - sy)**2
    covered |= (d2 <= SENS_RADIUS**2)

coverage_pct = 100 * covered.sum() / covered.size

# ──────────────────────────────────────────────
#  STEP 6 — Find sensor closest to T0
# ──────────────────────────────────────────────
deltas          = np.abs(sensor_temps - T0)
closest_idx     = int(np.argmin(deltas))
closest_temp    = sensor_temps[closest_idx]
closest_delta   = deltas[closest_idx]
closest_region  = sensor_regions[closest_idx]

# ──────────────────────────────────────────────
#  PRINT TABLE
# ──────────────────────────────────────────────
print("=" * 65)
print("  GREENHOUSE SENSOR NETWORK — SIMULATION RESULTS")
print("=" * 65)
print(f"  Area              : {AREA_W} m × {AREA_H} m")
print(f"  Total sensors     : {N_SENSORS} ({GRID_ROWS} rows × {GRID_COLS} cols)")
print(f"  Sensing radius    : {SENS_RADIUS} m  (binary — sensed / not sensed)")
print(f"  Target temp T₀    : {T0} °C")
print(f"  Region boundaries : Low < {LOW_MAX:.1f}°C  |  Medium < {HIGH_MIN:.1f}°C  |  High ≥ {HIGH_MIN:.1f}°C")
print(f"  Area coverage     : {coverage_pct:.1f}%")
print()
print(f"  {'ID':<5} {'X (m)':<8} {'Y (m)':<8} {'Temp (°C)':<12} {'Region':<10} {'|ΔT| from T₀':<15} {'Best?'}")
print("  " + "-" * 63)
sorted_ids = np.argsort(deltas)
for rank, i in enumerate(sorted_ids):
    star = " ◄ CLOSEST" if i == closest_idx else ""
    print(f"  {i+1:<5} {sensor_x[i]:<8.1f} {sensor_y[i]:<8.1f} "
          f"{sensor_temps[i]:<12.2f} {sensor_regions[i]:<10} "
          f"{deltas[i]:<15.2f}{star}")
print("=" * 65)
print(f"  → Sensor {closest_idx+1} at ({sensor_x[closest_idx]:.1f}, {sensor_y[closest_idx]:.1f}) m")
print(f"    reads {closest_temp:.2f}°C  — closest to T₀={T0}°C  (Δ = {closest_delta:.2f}°C)")
print(f"    Region: {closest_region}")
print("=" * 65)

# ──────────────────────────────────────────────
#  VISUALISATION
# ──────────────────────────────────────────────
REGION_COLORS = {"Low": "#3B82F6", "Medium": "#F59E0B", "High": "#EF4444"}

fig = plt.figure(figsize=(18, 11), facecolor="#0F1117")
gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])   # temperature heatmap
ax2 = fig.add_subplot(gs[0, 1])   # coverage map
ax3 = fig.add_subplot(gs[0, 2])   # region map
ax4 = fig.add_subplot(gs[1, 0])   # temperature bar chart
ax5 = fig.add_subplot(gs[1, 1])   # delta from T0 bar chart
ax6 = fig.add_subplot(gs[1, 2])   # summary table

def style(ax, title):
    ax.set_facecolor("#161B22")
    ax.set_title(title, color="white", fontsize=10, pad=7, fontweight="normal")
    ax.tick_params(colors="#888", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2a2a3a")
    ax.set_xlabel("X (m)", color="#888", fontsize=8)
    ax.set_ylabel("Y (m)", color="#888", fontsize=8)

ext = [0, AREA_W, 0, AREA_H]

# ── Plot 1: Temperature heatmap ───────────────
im = ax1.imshow(T_field, extent=ext, origin="lower", cmap="RdYlBu_r",
                aspect="auto", vmin=T_MIN, vmax=T_MAX)
cb = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cb.ax.tick_params(colors="#888", labelsize=7)
cb.set_label("Temp (°C)", color="#aaa", fontsize=8)
for i, (sx, sy, sr) in enumerate(zip(sensor_x, sensor_y, sensor_regions)):
    ec = REGION_COLORS[sr]
    ax1.scatter(sx, sy, s=55, color=ec, edgecolors="white", linewidths=0.8, zorder=4)
    ax1.text(sx + 1.5, sy + 1.5, str(i+1), color="white", fontsize=6, zorder=5)
ax1.scatter(sensor_x[closest_idx], sensor_y[closest_idx],
            s=160, color="white", marker="*", zorder=6, label=f"Closest (S{closest_idx+1})")
ax1.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white", loc="upper right")
style(ax1, "Temperature Field (Greenhouse)")

# ── Plot 2: Coverage map ──────────────────────
cov_rgba = np.zeros((*covered.shape, 4))
cov_rgba[covered]  = [0.13, 0.77, 0.37, 0.35]    # sensed — green tint
cov_rgba[~covered] = [0.80, 0.20, 0.20, 0.55]    # not sensed — red tint
ax2.imshow(T_field, extent=ext, origin="lower", cmap="gray",
           aspect="auto", alpha=0.3, vmin=T_MIN, vmax=T_MAX)
ax2.imshow(cov_rgba, extent=ext, origin="lower", aspect="auto")
for sx, sy in zip(sensor_x, sensor_y):
    circ = Circle((sx, sy), SENS_RADIUS, fill=False,
                  edgecolor="#22C55E", linewidth=0.9, linestyle="--", zorder=4)
    ax2.add_patch(circ)
    ax2.scatter(sx, sy, s=40, color="#22C55E", zorder=5)
p1 = mpatches.Patch(color="#22C55E", alpha=0.5, label=f"Sensed ({coverage_pct:.1f}%)")
p2 = mpatches.Patch(color="#EF4444", alpha=0.6, label="Not sensed")
ax2.legend(handles=[p1, p2], fontsize=7, facecolor="#1a1a2e",
           edgecolor="#333", labelcolor="white", loc="upper right")
ax2.set_xlim(0, AREA_W); ax2.set_ylim(0, AREA_H)
style(ax2, f"Coverage Map  (radius = {SENS_RADIUS} m)")

# ── Plot 3: Region map ────────────────────────
region_field = np.where(T_field < LOW_MAX, 0, np.where(T_field < HIGH_MIN, 1, 2))
from matplotlib.colors import ListedColormap
rcmap = ListedColormap(["#1E3A5F", "#78350F", "#7F1D1D"])
ax3.imshow(region_field, extent=ext, origin="lower", cmap=rcmap,
           aspect="auto", alpha=0.75)
for i, (sx, sy, sr, st) in enumerate(zip(sensor_x, sensor_y, sensor_regions, sensor_temps)):
    ec = REGION_COLORS[sr]
    ax3.scatter(sx, sy, s=60, color=ec, edgecolors="white", linewidths=0.8, zorder=4)
    ax3.text(sx, sy - 4, f"{st:.1f}°", color="white", fontsize=6,
             ha="center", zorder=5)
ax3.scatter(sensor_x[closest_idx], sensor_y[closest_idx],
            s=180, color="white", marker="*", zorder=6)
rp = [mpatches.Patch(color="#1E3A5F", label=f"Low  (< {LOW_MAX:.1f}°C)"),
      mpatches.Patch(color="#78350F", label=f"Medium ({LOW_MAX:.1f}–{HIGH_MIN:.1f}°C)"),
      mpatches.Patch(color="#7F1D1D", label=f"High (≥ {HIGH_MIN:.1f}°C)")]
ax3.legend(handles=rp, fontsize=7, facecolor="#1a1a2e",
           edgecolor="#333", labelcolor="white", loc="upper right")
style(ax3, "Temperature Regions + Sensor Readings")

# ── Plot 4: Sensor temperatures bar chart ─────
colors = [REGION_COLORS[r] for r in sensor_regions]
bars = ax4.bar(range(1, N_SENSORS+1), sensor_temps, color=colors,
               edgecolor="#333", linewidth=0.5)
ax4.axhline(T0, color="white", linestyle="--", linewidth=1.2, label=f"T₀ = {T0}°C")
ax4.axhline(LOW_MAX,  color="#60A5FA", linestyle=":", linewidth=0.9)
ax4.axhline(HIGH_MIN, color="#F87171", linestyle=":", linewidth=0.9)
bars[closest_idx].set_edgecolor("white"); bars[closest_idx].set_linewidth(2)
ax4.set_xlabel("Sensor ID", color="#888", fontsize=8)
ax4.set_ylabel("Temperature (°C)", color="#888", fontsize=8)
ax4.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
ax4.set_facecolor("#161B22")
ax4.set_title("Sensor Temperatures", color="white", fontsize=10, pad=7)
ax4.tick_params(colors="#888", labelsize=8)
for sp in ax4.spines.values(): sp.set_edgecolor("#2a2a3a")
ax4.set_xticks(range(1, N_SENSORS+1))

# ── Plot 5: Delta from T0 ─────────────────────
dcols = ["#22C55E" if i == closest_idx else "#64748B" for i in range(N_SENSORS)]
ax5.bar(range(1, N_SENSORS+1), deltas, color=dcols, edgecolor="#333", linewidth=0.5)
ax5.set_xlabel("Sensor ID", color="#888", fontsize=8)
ax5.set_ylabel("|T_sensor − T₀|  (°C)", color="#888", fontsize=8)
ax5.set_facecolor("#161B22")
ax5.set_title(f"Deviation from T₀ = {T0}°C  (green = closest)", color="white", fontsize=10, pad=7)
ax5.tick_params(colors="#888", labelsize=8)
for sp in ax5.spines.values(): sp.set_edgecolor("#2a2a3a")
ax5.set_xticks(range(1, N_SENSORS+1))

# ── Plot 6: Summary table ─────────────────────
ax6.set_facecolor("#0D1117")
ax6.axis("off")
ax6.set_title("Simulation Summary", color="white", fontsize=10, pad=7)
rows = [
    ["Scenario",          "Greenhouse monitoring"],
    ["Area",              f"{AREA_W} m × {AREA_H} m"],
    ["Total sensors",     str(N_SENSORS)],
    ["Sensing radius",    f"{SENS_RADIUS} m"],
    ["Coverage",          f"{coverage_pct:.1f}%"],
    ["Target T₀",         f"{T0} °C"],
    ["Closest sensor",    f"S{closest_idx+1}"],
    ["Closest temp",      f"{closest_temp:.2f} °C"],
    ["Δ from T₀",         f"{closest_delta:.2f} °C"],
    ["Region",            closest_region],
    ["Low region",        f"< {LOW_MAX:.1f} °C"],
    ["Medium region",     f"{LOW_MAX:.1f} – {HIGH_MIN:.1f} °C"],
    ["High region",       f"≥ {HIGH_MIN:.1f} °C"],
]
tbl = ax6.table(cellText=rows, colLabels=["Parameter", "Value"],
                cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor("#1E293B" if r % 2 == 0 else "#0F172A")
    cell.set_edgecolor("#2a2a3a")
    cell.set_text_props(color="#60A5FA" if r == 0 else "white")

fig.suptitle(
    "Simple Wireless Sensor Network — Grid Placement · Binary Sensing · Temperature Regions\n"
    "Real-life scenario: Greenhouse Temperature Monitoring",
    color="white", fontsize=11, y=0.99, fontweight="normal"
)

plt.show()