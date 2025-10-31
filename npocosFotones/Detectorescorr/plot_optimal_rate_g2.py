import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Archivos de entrada (generados por tu C) ===
HERALDED_CSV = "heralded_with_noise.csv"      # columnas: rate, window_duration, p_dark, p_click_heralded
CROSSCORR_CSV = "cross_correlation.csv"       # columnas: rate, window_duration, p_dark, g2_cross   (con el fix)

# === Parámetros de uso rápido ===
# Elegí acá el p_dark y la ventana de interés para que el script imprima el rate recomendado:
TARGET_P_DARK = 1e-4       # ejemplo
TARGET_WINDOW = 1e-7       # ejemplo (segundos)

# === Carga de datos ===
df_h = pd.read_csv(HERALDED_CSV)
df_c = pd.read_csv(CROSSCORR_CSV)

# Sanitizar tipos numéricos
for col in ["rate", "window_duration", "p_dark"]:
    if col in df_h.columns: df_h[col] = pd.to_numeric(df_h[col], errors="coerce")
    if col in df_c.columns: df_c[col] = pd.to_numeric(df_c[col], errors="coerce")

# === OPTIMIZACIÓN: para cada (p_dark, window_duration) encontrar rate que minimiza g2_cross ===
# Nos quedamos solo con combinaciones válidas



# --- Backfill de p_dark en crosscorr si falta ---
DEFAULT_PDARK_SWEEP = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

def backfill_pdark_in_cross(df_cross, df_heralded):
    if "p_dark" in df_cross.columns:
        return df_cross  # nada que hacer

    df_cross = df_cross.copy()
    df_cross["p_dark"] = np.nan

    # Intento 1: inferir desde heralded (si existe esa combinatoria)
    have_h = set(["rate", "window_duration", "p_dark"]).issubset(df_heralded.columns)
    grouped = df_cross.groupby(["rate", "window_duration"], sort=False)

    filled_idx = []
    for (rate, win), g in grouped:
        idx = g.index
        pdarks = None
        if have_h:
            dh = df_heralded[(df_heralded["rate"] == rate) & (df_heralded["window_duration"] == win)]
            if len(dh):
                # orden ascendente de p_dark presentes en heralded
                pdarks = sorted(dh["p_dark"].dropna().unique().tolist())
        if pdarks is None or len(pdarks) == 0:
            pdarks = DEFAULT_PDARK_SWEEP

        # Asignar en orden de aparición (C recorre k=0..7)
        n = min(len(idx), len(pdarks))
        df_cross.loc[idx[:n], "p_dark"] = pdarks[:n]
        filled_idx.extend(idx[:n])

    # Si quedó algo sin asignar, completar con el primer valor por seguridad
    if df_cross["p_dark"].isna().any():
        df_cross.loc[df_cross["p_dark"].isna(), "p_dark"] = DEFAULT_PDARK_SWEEP[0]

    return df_cross

df_c = backfill_pdark_in_cross(df_c, df_h)



cols_key = ["p_dark", "window_duration"]
assert set(["rate", "g2_cross"] + cols_key).issubset(df_c.columns), \
    "El CSV de correlación cruzada debe tener: rate, window_duration, p_dark, g2_cross"

# agrupamos por (p_dark, ventana) y tomamos el índice de la fila con g2 mínimo
idxmin = df_c.groupby(cols_key)["g2_cross"].idxmin()
df_opt = df_c.loc[idxmin, ["p_dark", "window_duration", "rate", "g2_cross"]].sort_values(cols_key).reset_index(drop=True)
df_opt = df_opt.rename(columns={"rate": "rate_opt", "g2_cross": "g2_min"})

# === (Opcional) Unimos info de heralded para ver el p_click_heralded en el óptimo ===
if set(["p_click_heralded"] + cols_key).issubset(df_h.columns):
    # merge por (p_dark, ventana, rate≈rate_opt)
    df_h_rounded = df_h.copy()
    df_h_rounded["rate_r"] = df_h_rounded["rate"].round(12)
    df_opt_rounded = df_opt.copy()
    df_opt_rounded["rate_opt_r"] = df_opt_rounded["rate_opt"].round(12)
    df_merge = pd.merge(
        df_opt_rounded,
        df_h_rounded,
        left_on=["p_dark", "window_duration", "rate_opt_r"],
        right_on=["p_dark", "window_duration", "rate_r"],
        how="left",
    )
    df_opt["p_click_heralded_at_opt"] = df_merge["p_click_heralded"]
else:
    df_opt["p_click_heralded_at_opt"] = np.nan

# === Reporte rápido para un p_dark y ventana elegidos ===
def nearest(arr, value):
    arr = np.asarray(arr)
    return arr[np.nanargmin(np.abs(arr - value))]

p_dark_sel = nearest(df_opt["p_dark"].unique(), TARGET_P_DARK)
win_sel    = nearest(df_opt["window_duration"].unique(), TARGET_WINDOW)
row_sel = df_opt[(df_opt["p_dark"]==p_dark_sel) & (df_opt["window_duration"]==win_sel)]
if not row_sel.empty:
    r_opt = float(row_sel["rate_opt"].iloc[0])
    g2min = float(row_sel["g2_min"].iloc[0])
    pclick = float(row_sel["p_click_heralded_at_opt"].iloc[0]) if not np.isnan(row_sel["p_click_heralded_at_opt"].iloc[0]) else None
    print(f"[RECOMENDACIÓN] p_dark≈{p_dark_sel:.1e}, ventana≈{win_sel:.1e} s → rate óptimo ≈ {r_opt:.2e} Hz; g2_min≈{g2min:.4f}"
          + (f"; p_click_heralded@opt≈{pclick:.4f}" if pclick is not None else ""))
else:
    print("No se encontró combinación cercana en la grilla para los valores objetivo.")

# === Mapas y curvas de “rate óptimo” y “g2 mínimo alcanzable” ===

# 1) Heatmap de rate óptimo (eje x: ventana ; eje y: p_dark)
pt_rate = df_opt.pivot(index="p_dark", columns="window_duration", values="rate_opt")
plt.figure(figsize=(8,6))
im = plt.imshow(pt_rate.values, aspect="auto", origin="lower")
plt.xticks(range(pt_rate.shape[1]), [f"{c:.0e}" for c in pt_rate.columns], rotation=45)
plt.yticks(range(pt_rate.shape[0]), [f"{r:.0e}" for r in pt_rate.index])
plt.colorbar(im, label="rate óptimo (Hz)")
plt.xlabel("Ventana de integración (s)")
plt.ylabel("p_dark por ventana")
plt.title("Rate óptimo que minimiza g² (correlación cruzada)")
plt.tight_layout()
plt.show()

# 2) Heatmap de g2 mínimo alcanzable en el óptimo
pt_g2 = df_opt.pivot(index="p_dark", columns="window_duration", values="g2_min")
plt.figure(figsize=(8,6))
im = plt.imshow(pt_g2.values, aspect="auto", origin="lower")
plt.xticks(range(pt_g2.shape[1]), [f"{c:.0e}" for c in pt_g2.columns], rotation=45)
plt.yticks(range(pt_g2.shape[0]), [f"{r:.0e}" for r in pt_g2.index])
plt.colorbar(im, label="g² mínimo (correlación cruzada)")
plt.xlabel("Ventana de integración (s)")
plt.ylabel("p_dark por ventana")
plt.title("g² mínimo alcanzable (en el rate óptimo)")
plt.tight_layout()
plt.show()

# 3) Curvas: rate óptimo vs ventana, separadas por p_dark
plt.figure(figsize=(8,6))
for p in sorted(df_opt["p_dark"].unique()):
    sub = df_opt[df_opt["p_dark"]==p].sort_values("window_duration")
    plt.plot(sub["window_duration"], sub["rate_opt"], marker="o", label=f"p_dark={p:.0e}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Ventana de integración (s)")
plt.ylabel("Rate óptimo (Hz)")
plt.title("Rate óptimo vs ventana para distintos p_dark")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()

# 4) Curvas: g2 mínimo vs ventana, separadas por p_dark
plt.figure(figsize=(8,6))
for p in sorted(df_opt["p_dark"].unique()):
    sub = df_opt[df_opt["p_dark"]==p].sort_values("window_duration")
    plt.plot(sub["window_duration"], sub["g2_min"], marker="s", label=f"p_dark={p:.0e}")
plt.xscale("log")
plt.xlabel("Ventana de integración (s)")
plt.ylabel("g² mínimo (correlación cruzada)")
plt.title("g² mínimo vs ventana para distintos p_dark")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(ncol=2, fontsize=9)
plt.tight_layout()
plt.show()
