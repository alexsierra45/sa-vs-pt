"""
Ejecutar una vez: python create_notebook.py
Genera experimentos_spin_glass.ipynb listo para usar.
"""
import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": f"md{abs(hash(source))%99999:05d}",
            "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None,
            "id": f"cd{abs(hash(source))%99999:05d}", "metadata": {},
            "outputs": [], "source": source}


# ─────────────────────────────────────────────────────────────────
# Contenido de cada celda
# ─────────────────────────────────────────────────────────────────

CELL_IMPORTS = """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from scipy import stats

try:
    from tqdm.notebook import tqdm
except ImportError:
    def tqdm(x, **kw): return x

import spin_glass_solvers as sgs

plt.rcParams['figure.dpi'] = 100
sns.set_theme(style='whitegrid')
RESULTADOS = Path('resultados')
RESULTADOS.mkdir(exist_ok=True)
print('Librerías cargadas.')
"""

CELL_INSTANCIA = """\
# Instancia fija N=100 con semilla 42 (compartida en todos los experimentos)
N = 100
SEED_INST = 42

G   = sgs.generar_grafo_regular(N, d=3, seed=SEED_INST)
J, H = sgs.generar_instancia(G, seed=SEED_INST)
nbrs = sgs.construir_lista_vecinos(G)
prob_dict = sgs.problema_a_dict(G, J, H)

print(f'Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas')
print(f'Grado promedio: {sum(d for _,d in G.degree())/N:.2f}')
"""

CELL_SA_HIPER = """\
# ── Experimento 1 – Búsqueda de hiperparámetros SA ──────────────
ITERS_PT = N ** 2      # 10 000 flips por temperatura
N_TEMPS   = 100        # → ~1 000 000 evaluaciones totales
N_RUNS    = 5          # corridas por configuración

CONFIGS_SA = [
    (5.0,  0.1,  'geometrico'),
    (10.0, 0.01, 'geometrico'),
    (3.0,  0.05, 'geometrico'),
    (5.0,  0.1,  'lineal'),
    (5.0,  0.1,  'logaritmico'),
]

resultados_sa_exp1 = []

for T_ini, T_fin, tipo in CONFIGS_SA:
    sched = sgs.SCHEDULE_FNS[tipo](T_ini, T_fin, N_TEMPS)
    energias, corridas = [], []

    for seed in range(N_RUNS):
        res = sgs.simulated_annealing(
            N, J, H, nbrs,
            temp_schedule=sched,
            iters_por_temp=ITERS_PT,
            seed=seed,
            registro_intervalo=5000,
        )
        res['problema']      = prob_dict
        res['hiperparametros'] = dict(T_ini=T_ini, T_fin=T_fin, tipo=tipo,
                                      N_TEMPS=N_TEMPS, iters_por_temp=ITERS_PT,
                                      seed=seed)
        energias.append(res['mejor_energia'])
        corridas.append(res)

    resultados_sa_exp1.append(dict(
        T_ini=T_ini, T_fin=T_fin, tipo=tipo,
        energia_media=float(np.mean(energias)),
        energia_std=float(np.std(energias)),
        energia_mejor=float(min(energias)),
        corridas=corridas,
    ))
    sgs.guardar_resultado({'corridas': corridas},
        RESULTADOS / f'exp1_sa_{tipo}_{T_ini}_{T_fin}.pkl')
    print(f'SA {tipo:12s} T=[{T_ini},{T_fin}]: '
          f'media={np.mean(energias):.4f} ± {np.std(energias):.4f}  '
          f'mejor={min(energias):.4f}')
"""

CELL_SA_TABLE = """\
df_sa = pd.DataFrame([{k: r[k] for k in
    ['T_ini','T_fin','tipo','energia_media','energia_std','energia_mejor']}
    for r in resultados_sa_exp1]).sort_values('energia_media')

print('=== SA – Experimento 1 ===')
display(df_sa.reset_index(drop=True))

best_sa = resultados_sa_exp1[df_sa.index[0]]
MEJOR_SA = dict(T_ini=best_sa['T_ini'], T_fin=best_sa['T_fin'],
                tipo=best_sa['tipo'], N_TEMPS=N_TEMPS, iters_por_temp=ITERS_PT)
print('\\nMejor config SA:', MEJOR_SA)
"""

CELL_PT_HIPER = """\
# ── Experimento 1 – Búsqueda de hiperparámetros PT ──────────────
PASOS_SWAP = 10
N_RUNS_PT  = 5
CONFIGS_PT = [(8, 0.1, 5.0), (16, 0.1, 5.0), (32, 0.1, 5.0)]

resultados_pt_exp1 = []

for n_rep, T_min, T_max in CONFIGS_PT:
    # Fijar ~1e6 evaluaciones totales entre todas las réplicas
    iters_tot = (1_000_000 // n_rep // PASOS_SWAP) * PASOS_SWAP
    sched_pt  = np.geomspace(T_min, T_max, n_rep)
    energias, corridas = [], []

    for seed in range(N_RUNS_PT):
        res = sgs.parallel_tempering(
            N, J, H, nbrs,
            temp_schedule=sched_pt,
            pasos_por_intercambio=PASOS_SWAP,
            iteraciones_totales=iters_tot,
            seed=seed,
            registro_intervalo=5000,
        )
        res['problema']      = prob_dict
        res['hiperparametros'] = dict(num_replicas=n_rep, T_min=T_min, T_max=T_max,
                                      pasos_por_intercambio=PASOS_SWAP,
                                      iteraciones_totales=iters_tot, seed=seed)
        energias.append(res['mejor_energia'])
        corridas.append(res)

    resultados_pt_exp1.append(dict(
        num_replicas=n_rep, T_min=T_min, T_max=T_max,
        iters_totales=iters_tot,
        energia_media=float(np.mean(energias)),
        energia_std=float(np.std(energias)),
        energia_mejor=float(min(energias)),
        corridas=corridas,
    ))
    sgs.guardar_resultado({'corridas': corridas},
        RESULTADOS / f'exp1_pt_{n_rep}rep.pkl')
    print(f'PT {n_rep:2d} réplicas: '
          f'media={np.mean(energias):.4f} ± {np.std(energias):.4f}  '
          f'mejor={min(energias):.4f}')
"""

CELL_PT_TABLE = """\
df_pt = pd.DataFrame([{k: r[k] for k in
    ['num_replicas','T_min','T_max','energia_media','energia_std','energia_mejor']}
    for r in resultados_pt_exp1]).sort_values('energia_media')

print('=== PT – Experimento 1 ===')
display(df_pt.reset_index(drop=True))

best_pt = resultados_pt_exp1[df_pt.index[0]]
MEJOR_PT = dict(num_replicas=best_pt['num_replicas'],
                T_min=best_pt['T_min'], T_max=best_pt['T_max'],
                pasos_por_intercambio=PASOS_SWAP,
                iters_totales=best_pt['iters_totales'])
print('\\nMejor config PT:', MEJOR_PT)
"""

CELL_EXP2_RUN = """\
# ── Experimento 2 – 30 corridas con mejores parámetros ──────────
N_CORRIDAS = 30

sched_sa = sgs.SCHEDULE_FNS[MEJOR_SA['tipo']](
    MEJOR_SA['T_ini'], MEJOR_SA['T_fin'], MEJOR_SA['N_TEMPS'])
sched_pt = np.geomspace(MEJOR_PT['T_min'], MEJOR_PT['T_max'],
                         MEJOR_PT['num_replicas'])

res_sa30, res_pt30 = [], []

print('SA – 30 corridas...')
for seed in tqdm(range(N_CORRIDAS)):
    r = sgs.simulated_annealing(N, J, H, nbrs,
            temp_schedule=sched_sa,
            iters_por_temp=MEJOR_SA['iters_por_temp'],
            seed=seed, registro_intervalo=5000)
    r['problema']      = prob_dict
    r['hiperparametros'] = {**MEJOR_SA, 'seed': seed}
    res_sa30.append(r)

print('PT – 30 corridas...')
for seed in tqdm(range(N_CORRIDAS)):
    r = sgs.parallel_tempering(N, J, H, nbrs,
            temp_schedule=sched_pt,
            pasos_por_intercambio=MEJOR_PT['pasos_por_intercambio'],
            iteraciones_totales=MEJOR_PT['iters_totales'],
            seed=seed, registro_intervalo=5000)
    r['problema']      = prob_dict
    r['hiperparametros'] = {**MEJOR_PT, 'seed': seed}
    res_pt30.append(r)

sgs.guardar_resultado({'corridas': res_sa30}, RESULTADOS / 'exp2_sa_30.pkl')
sgs.guardar_resultado({'corridas': res_pt30}, RESULTADOS / 'exp2_pt_30.pkl')
print('Guardado.')
"""

CELL_EXP2_STATS = """\
def resumen(resultados, nombre):
    E = [r['mejor_energia'] for r in resultados]
    T = [r['tiempo_ejecucion'] for r in resultados]
    moda_val = float(stats.mode(np.round(E, 2)).mode)
    return pd.DataFrame([dict(
        algoritmo=nombre,
        min=min(E), media=np.mean(E), mediana=np.median(E), moda=moda_val,
        std=np.std(E), t_medio=np.mean(T), t_std=np.std(T),
    )])

df_stats = pd.concat([resumen(res_sa30,'SA'), resumen(res_pt30,'PT')],
                     ignore_index=True)
print('=== Estadísticas – Experimento 2 ===')
display(df_stats)
"""

CELL_EXP2_PLOTS = """\
E_sa = [r['mejor_energia'] for r in res_sa30]
E_pt = [r['mejor_energia'] for r in res_pt30]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Boxplot
axes[0].boxplot([E_sa, E_pt], labels=['SA', 'PT'], patch_artist=True,
                boxprops=dict(facecolor='#4C72B0', alpha=0.6))
axes[0].set_title('Energías finales (30 corridas)')
axes[0].set_ylabel('Energía')

# Histograma
axes[1].hist(E_sa, bins=10, alpha=0.6, label='SA', color='steelblue')
axes[1].hist(E_pt, bins=10, alpha=0.6, label='PT', color='darkorange')
axes[1].set_title('Histograma de energías')
axes[1].set_xlabel('Energía')
axes[1].legend()

# Trayectorias superpuestas
for r in res_sa30:
    tr = r['trayectoria_energia']
    xs = np.linspace(0, r['evaluaciones_energia'], len(tr))
    axes[2].plot(xs, tr, alpha=0.25, color='steelblue', lw=0.7)
for r in res_pt30:
    tr = r['trayectoria_energia']
    xs = np.linspace(0, r['evaluaciones_energia'], len(tr))
    axes[2].plot(xs, tr, alpha=0.25, color='darkorange', lw=0.7)

from matplotlib.patches import Patch
axes[2].legend(handles=[Patch(color='steelblue', label='SA'),
                         Patch(color='darkorange', label='PT')])
axes[2].set_title('Trayectorias de energía')
axes[2].set_xlabel('Evaluaciones')
axes[2].set_ylabel('Energía')

plt.tight_layout()
plt.savefig(RESULTADOS / 'exp2_plots.png', dpi=150, bbox_inches='tight')
plt.show()
"""

CELL_EXP3_RUN = """\
# ── Experimento 3 – Escalabilidad ────────────────────────────────
TAMANIOS     = [50, 100, 200, 400]
N_INSTANCIAS = 3
N_RUNS_3     = 10
PASOS_3      = 10

filas, todos_exp3 = [], []

for N3 in tqdm(TAMANIOS, desc='N'):
    for inst in range(N_INSTANCIAS):
        G3   = sgs.generar_grafo_regular(N3, d=3, seed=1000 + inst)
        J3,H3 = sgs.generar_instancia(G3, seed=1000 + inst)
        nb3  = sgs.construir_lista_vecinos(G3)
        pd3  = sgs.problema_a_dict(G3, J3, H3)

        # Parámetros SA escalados
        sched_sa3 = sgs.SCHEDULE_FNS[MEJOR_SA['tipo']](
            MEJOR_SA['T_ini'], MEJOR_SA['T_fin'], MEJOR_SA['N_TEMPS'])
        iters3 = N3 ** 2

        # Parámetros PT: num_replicas ∝ sqrt(N)
        nr3  = max(4, int(round(MEJOR_PT['num_replicas'] * (N3/100)**0.5)))
        nr3  = min(nr3, 64)
        it3  = max(PASOS_3, (10*N3**2 // nr3 // PASOS_3) * PASOS_3)
        sched_pt3 = np.geomspace(MEJOR_PT['T_min'], MEJOR_PT['T_max'], nr3)

        for algo in ('SA', 'PT'):
            Es, Ts, runs = [], [], []
            for seed in range(N_RUNS_3):
                if algo == 'SA':
                    r = sgs.simulated_annealing(N3, J3, H3, nb3,
                            temp_schedule=sched_sa3, iters_por_temp=iters3,
                            seed=seed,
                            registro_intervalo=max(100, N3*10))
                    r['hiperparametros'] = {**MEJOR_SA, 'iters_por_temp': iters3,
                                             'seed': seed}
                else:
                    r = sgs.parallel_tempering(N3, J3, H3, nb3,
                            temp_schedule=sched_pt3,
                            pasos_por_intercambio=PASOS_3,
                            iteraciones_totales=it3,
                            seed=seed,
                            registro_intervalo=max(100, N3*10))
                    r['hiperparametros'] = dict(num_replicas=nr3,
                            T_min=MEJOR_PT['T_min'], T_max=MEJOR_PT['T_max'],
                            pasos_por_intercambio=PASOS_3,
                            iteraciones_totales=it3, seed=seed)
                r['problema'] = pd3
                Es.append(r['mejor_energia'])
                Ts.append(r['tiempo_ejecucion'])
                runs.append(r)

            sgs.guardar_resultado({'corridas': runs},
                RESULTADOS / f'exp3_{algo}_N{N3}_i{inst}.pkl')
            todos_exp3.append(dict(N=N3, instancia=inst, algoritmo=algo, corridas=runs))
            filas.append(dict(N=N3, instancia=inst, algoritmo=algo,
                               energia_media=np.mean(Es), energia_std=np.std(Es),
                               tiempo_medio=np.mean(Ts), tiempo_std=np.std(Ts)))

df_escal = pd.DataFrame(filas)
df_escal.to_csv(RESULTADOS / 'exp3_escalabilidad.csv', index=False)
print('Experimento 3 completo.')
agg = df_escal.groupby(['N','algoritmo'])[['energia_media','tiempo_medio']].mean()
display(agg)
"""

CELL_EXP3_PLOTS = """\
agg = df_escal.groupby(['N','algoritmo']).agg(
    E=('energia_media','mean'), T=('tiempo_medio','mean')).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colores = {'SA':'steelblue','PT':'darkorange'}

for algo in ('SA','PT'):
    s = agg[agg['algoritmo']==algo]
    axes[0].plot(s['N'], s['E'], 'o-', label=algo, color=colores[algo])
    axes[1].loglog(s['N'], s['T'], 's-', label=algo, color=colores[algo])

    Ns = np.log(s['N'].values.astype(float))
    Ts = np.log(s['T'].values.astype(float))
    if len(Ns) > 1:
        exp = np.polyfit(Ns, Ts, 1)[0]
        axes[1].annotate(f'{algo}: ~N^{exp:.2f}',
            xy=(s['N'].values[-1], s['T'].values[-1]),
            xytext=(-60, 8), textcoords='offset points', fontsize=9)

axes[0].set(xlabel='N', ylabel='Energía media', title='Energía vs N')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].set(xlabel='N (log)', ylabel='Tiempo (s, log)',
            title='Tiempo de ejecución vs N (log-log)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTADOS / 'exp3_escalabilidad.png', dpi=150, bbox_inches='tight')
plt.show()
"""

# ─────────────────────────────────────────────────────────────────
# Construcción del notebook
# ─────────────────────────────────────────────────────────────────

cells = [
    md("# Experimentos: Spin Glass – SA vs PT\n\n"
       "Comparación de **Simulated Annealing** y **Parallel Tempering** para "
       "encontrar el estado fundamental de un modelo de Ising con desorden "
       "en un grafo 3-regular.\n\n"
       "**Experimentos:**\n"
       "1. Ajuste de hiperparámetros (N=100, semilla=42)\n"
       "2. Comparación de 30 corridas independientes\n"
       "3. Análisis de escalabilidad (N ∈ {50,100,200,400})"),
    code(CELL_IMPORTS),
    code(CELL_INSTANCIA),

    md("## Experimento 1: Ajuste de hiperparámetros\n\n"
       "Se evalúan distintos schedules de temperatura para SA y distintos "
       "números de réplicas para PT, fijando ~1 000 000 evaluaciones por "
       "configuración y promediando 5 corridas independientes."),
    code(CELL_SA_HIPER),
    code(CELL_SA_TABLE),
    code(CELL_PT_HIPER),
    code(CELL_PT_TABLE),

    md("## Experimento 2: 30 corridas – SA vs PT\n\n"
       "Con los mejores parámetros encontrados en Exp. 1, se ejecutan 30 "
       "corridas independientes sobre la misma instancia N=100."),
    code(CELL_EXP2_RUN),
    code(CELL_EXP2_STATS),
    code(CELL_EXP2_PLOTS),

    md("## Experimento 3: Escalabilidad\n\n"
       "Se mide energía final y tiempo de ejecución para N ∈ {50,100,200,400} "
       "con 3 instancias por tamaño y 10 corridas por instancia-algoritmo. "
       "El gráfico log-log permite estimar la complejidad empírica."),
    code(CELL_EXP3_RUN),
    code(CELL_EXP3_PLOTS),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

out = Path("experimentos_spin_glass.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook generado: {out.resolve()}")
