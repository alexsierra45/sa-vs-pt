"""
spin_glass_solvers.py
=====================
Funciones reutilizables para optimizar sistemas de espines de Ising con
desorden sobre grafos d-regulares, usando Simulated Annealing (SA) y
Parallel Tempering (PT).

Energía del modelo:
    E = -Σ_{<i,j>} J_ij s_i s_j - Σ_i H_i s_i

Espines binarios: s_i ∈ {-1, +1}.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np


# ──────────────────────────────────────────────────────────────────
# 1. Generación del problema
# ──────────────────────────────────────────────────────────────────

def generar_grafo_regular(
    N: int,
    d: int = 3,
    seed: Optional[int] = None,
) -> nx.Graph:
    """Genera un grafo d-regular no dirigido con N nodos.

    Parameters
    ----------
    N : int
        Número de nodos. El producto N*d debe ser par.
    d : int
        Grado de cada nodo (default: 3).
    seed : int, optional
        Semilla para reproducibilidad.

    Returns
    -------
    nx.Graph
        Grafo d-regular aleatorio con nodos etiquetados 0..N-1.
    """
    return nx.random_regular_graph(d, N, seed=seed)


def generar_instancia(
    G: nx.Graph,
    seed: Optional[int] = None,
) -> Tuple[Dict[Tuple[int, int], float], np.ndarray]:
    """Genera acoplamientos J_ij ~ N(0,1) y campos H_i ~ N(0,1).

    Parameters
    ----------
    G : nx.Graph
        Grafo que define la topología de interacciones.
    seed : int, optional
        Semilla para reproducibilidad.

    Returns
    -------
    J : dict {(min(i,j), max(i,j)): float}
        Acoplamientos por arista.
    H : np.ndarray of shape (N,)
        Campos externos.
    """
    rng = np.random.default_rng(seed)
    J: Dict[Tuple[int, int], float] = {
        (min(u, v), max(u, v)): rng.standard_normal()
        for u, v in G.edges()
    }
    H = rng.standard_normal(G.number_of_nodes())
    return J, H


def construir_lista_vecinos(G: nx.Graph) -> List[List[int]]:
    """Construye la lista de adyacencia como lista de listas para acceso O(d).

    Parameters
    ----------
    G : nx.Graph

    Returns
    -------
    list of list of int
        neighbors[i] = lista de vecinos del nodo i.
    """
    return [list(G.neighbors(i)) for i in range(G.number_of_nodes())]


def problema_a_dict(
    G: nx.Graph,
    J: Dict[Tuple[int, int], float],
    H: np.ndarray,
) -> Dict:
    """Serializa el problema a un dict JSON-compatible.

    Las claves de J se convierten a strings 'i,j' para compatibilidad JSON.
    """
    return {
        "N": int(G.number_of_nodes()),
        "edges": [[int(u), int(v)] for u, v in G.edges()],
        "J": {f"{i},{j}": float(v) for (i, j), v in J.items()},
        "H": H.tolist(),
    }


def problema_desde_dict(
    d: Dict,
) -> Tuple[nx.Graph, Dict[Tuple[int, int], float], np.ndarray, List[List[int]]]:
    """Reconstruye el problema desde un diccionario serializado.

    Returns
    -------
    G, J, H, neighbors
    """
    N = d["N"]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from([tuple(e) for e in d["edges"]])
    J = {
        (int(k.split(",")[0]), int(k.split(",")[1])): float(v)
        for k, v in d["J"].items()
    }
    H = np.array(d["H"])
    neighbors = construir_lista_vecinos(G)
    return G, J, H, neighbors


# ──────────────────────────────────────────────────────────────────
# 2. Función de energía
# ──────────────────────────────────────────────────────────────────

def energia(
    spins: np.ndarray,
    J: Dict[Tuple[int, int], float],
    H: np.ndarray,
    neighbors: List[List[int]],
) -> float:
    """Calcula la energía total E = -Σ J_ij s_i s_j - Σ H_i s_i.

    Parameters
    ----------
    spins : np.ndarray of int8, shape (N,)
    J, H, neighbors : parámetros del problema.

    Returns
    -------
    float
    """
    E = 0.0
    for (i, j), jval in J.items():
        E -= jval * float(spins[i]) * float(spins[j])
    E -= float(np.dot(H, spins))
    return E


def delta_energia(
    spins: np.ndarray,
    i: int,
    J: Dict[Tuple[int, int], float],
    H: np.ndarray,
    neighbors: List[List[int]],
) -> float:
    """Calcula ΔE al voltear el espín i sin recalcular toda la energía.

    ΔE = 2 * s_i * (H_i + Σ_{j ∈ N(i)} J_ij s_j)

    Solo recorre los vecinos de i, O(d) en un grafo d-regular.
    """
    si = int(spins[i])
    campo = float(H[i])
    for j in neighbors[i]:
        key = (min(i, j), max(i, j))
        campo += J[key] * float(spins[j])
    return 2.0 * si * campo


# ──────────────────────────────────────────────────────────────────
# 3. Schedules de temperatura
# ──────────────────────────────────────────────────────────────────

def schedule_geometrico(T_ini: float, T_fin: float, n_pasos: int) -> np.ndarray:
    """Schedule geométrico: T_{k+1} = alpha * T_k (alpha < 1 constante)."""
    return np.geomspace(T_ini, T_fin, n_pasos)


def schedule_lineal(T_ini: float, T_fin: float, n_pasos: int) -> np.ndarray:
    """Schedule lineal: temperatura decrece uniformemente."""
    return np.linspace(T_ini, T_fin, n_pasos)


def schedule_logaritmico(T_ini: float, T_fin: float, n_pasos: int) -> np.ndarray:
    """Schedule logarítmico: T_k ∝ 1/log(1+k), enfriamiento lento al inicio."""
    k = np.arange(1, n_pasos + 1, dtype=float)
    raw = 1.0 / np.log1p(k)
    raw = (raw - raw[-1]) / (raw[0] - raw[-1])
    return T_fin + raw * (T_ini - T_fin)


SCHEDULE_FNS = {
    "geometrico": schedule_geometrico,
    "lineal": schedule_lineal,
    "logaritmico": schedule_logaritmico,
}


# ──────────────────────────────────────────────────────────────────
# 4. Simulated Annealing
# ──────────────────────────────────────────────────────────────────

def simulated_annealing(
    N: int,
    J: Dict[Tuple[int, int], float],
    H: np.ndarray,
    neighbors: List[List[int]],
    temp_schedule: np.ndarray,
    iters_por_temp: int,
    seed: Optional[int] = None,
    registro_intervalo: int = 1000,
) -> Dict:
    """Simulated Annealing para minimizar la energía del sistema de Ising.

    En cada temperatura T realiza `iters_por_temp` intentos de flip.
    Un flip del espín i se acepta si ΔE < 0 o con probabilidad exp(-ΔE/T).

    Parameters
    ----------
    N : int
    J, H, neighbors : parámetros del problema.
    temp_schedule : np.ndarray
        Secuencia de temperaturas (de mayor a menor).
    iters_por_temp : int
        Intentos de flip por nivel de temperatura.
    seed : int, optional
    registro_intervalo : int
        Cada cuántas evaluaciones se agrega la energía a la trayectoria.

    Returns
    -------
    dict con claves: config_inicial, algoritmo, mejor_energia,
        mejor_configuracion, trayectoria_energia, tiempo_ejecucion,
        evaluaciones_energia.
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()

    spins = rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
    config_inicial = spins.tolist()
    E = energia(spins, J, H, neighbors)
    mejor_E = E
    mejor_spins = spins.copy()

    trayectoria: List[float] = [E]
    evals = 1
    proxima_reg = registro_intervalo

    for T in temp_schedule:
        for _ in range(iters_por_temp):
            i = int(rng.integers(N))
            dE = delta_energia(spins, i, J, H, neighbors)
            evals += 1

            if dE < 0.0 or rng.random() < np.exp(-dE / T):
                spins[i] = -spins[i]
                E += dE
                if E < mejor_E:
                    mejor_E = E
                    mejor_spins = spins.copy()

            if evals >= proxima_reg:
                trayectoria.append(E)
                proxima_reg += registro_intervalo

    return {
        "config_inicial": config_inicial,
        "algoritmo": "SA",
        "mejor_energia": float(mejor_E),
        "mejor_configuracion": mejor_spins.tolist(),
        "trayectoria_energia": trayectoria,
        "tiempo_ejecucion": time.perf_counter() - t0,
        "evaluaciones_energia": evals,
    }


# ──────────────────────────────────────────────────────────────────
# 5. Parallel Tempering
# ──────────────────────────────────────────────────────────────────

def parallel_tempering(
    N: int,
    J: Dict[Tuple[int, int], float],
    H: np.ndarray,
    neighbors: List[List[int]],
    temp_schedule: np.ndarray,
    pasos_por_intercambio: int,
    iteraciones_totales: int,
    seed: Optional[int] = None,
    registro_intervalo: int = 1000,
) -> Dict:
    """Parallel Tempering (replica-exchange Monte Carlo).

    Mantiene R = len(temp_schedule) réplicas en paralelo, cada una a
    temperatura distinta. Cada `pasos_por_intercambio` pasos de MC, intenta
    intercambiar configuraciones entre réplicas adyacentes usando el criterio
    de Metropolis para preservar el balance detallado.

    Parameters
    ----------
    N : int
    J, H, neighbors : parámetros del problema.
    temp_schedule : np.ndarray
        Temperaturas por réplica, ordenadas de menor a mayor.
    pasos_por_intercambio : int
        Pasos de MC por réplica entre intentos de swap.
    iteraciones_totales : int
        Total de pasos de MC por réplica.
    seed : int, optional
    registro_intervalo : int
        Cada cuántas evaluaciones totales se registra la mejor energía.

    Returns
    -------
    dict (mismo formato que simulated_annealing).
    """
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    R = len(temp_schedule)

    replicas = [
        rng.choice(np.array([-1, 1], dtype=np.int8), size=N)
        for _ in range(R)
    ]
    config_inicial = replicas[0].tolist()
    energias = np.array([energia(r, J, H, neighbors) for r in replicas])

    mejor_E = float(energias.min())
    mejor_spins = replicas[int(energias.argmin())].copy()

    trayectoria: List[float] = [mejor_E]
    evals = R  # energías iniciales
    proxima_reg = registro_intervalo

    n_ciclos = iteraciones_totales // pasos_por_intercambio

    for ciclo in range(n_ciclos):
        # Paso de MC para cada réplica
        for r in range(R):
            T = float(temp_schedule[r])
            sp = replicas[r]
            E_r = energias[r]

            for _ in range(pasos_por_intercambio):
                i = int(rng.integers(N))
                dE = delta_energia(sp, i, J, H, neighbors)
                evals += 1

                if dE < 0.0 or rng.random() < np.exp(-dE / T):
                    sp[i] = -sp[i]
                    E_r += dE
                    if E_r < mejor_E:
                        mejor_E = E_r
                        mejor_spins = sp.copy()

            energias[r] = E_r

        # Intentos de intercambio entre pares adyacentes (alternando offset)
        inicio = ciclo % 2
        for r in range(inicio, R - 1, 2):
            beta_r = 1.0 / float(temp_schedule[r])
            beta_r1 = 1.0 / float(temp_schedule[r + 1])
            delta = (beta_r - beta_r1) * (energias[r] - energias[r + 1])
            if delta >= 0.0 or rng.random() < np.exp(delta):
                replicas[r], replicas[r + 1] = replicas[r + 1], replicas[r]
                energias[r], energias[r + 1] = energias[r + 1], energias[r]

        if evals >= proxima_reg:
            trayectoria.append(mejor_E)
            proxima_reg += registro_intervalo

    return {
        "config_inicial": config_inicial,
        "algoritmo": "PT",
        "mejor_energia": float(mejor_E),
        "mejor_configuracion": mejor_spins.tolist(),
        "trayectoria_energia": trayectoria,
        "tiempo_ejecucion": time.perf_counter() - t0,
        "evaluaciones_energia": evals,
    }


# ──────────────────────────────────────────────────────────────────
# 6. Persistencia
# ──────────────────────────────────────────────────────────────────

def guardar_resultado(
    resultado: Dict,
    filepath: Union[str, Path],
    formato: str = "pkl",
) -> None:
    """Guarda un resultado de experimento en disco.

    Parameters
    ----------
    resultado : dict
    filepath : str or Path
    formato : {'pkl', 'json'}
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if formato == "pkl":
        with open(filepath, "wb") as fh:
            pickle.dump(resultado, fh)
    elif formato == "json":
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(resultado, fh, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Formato desconocido: {formato!r}. Use 'pkl' o 'json'.")


def cargar_resultado(filepath: Union[str, Path]) -> Dict:
    """Carga un resultado de experimento desde disco.

    Parameters
    ----------
    filepath : str or Path (.pkl o .json)

    Returns
    -------
    dict
    """
    filepath = Path(filepath)
    if filepath.suffix == ".pkl":
        with open(filepath, "rb") as fh:
            return pickle.load(fh)
    elif filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)
    else:
        raise ValueError(f"Extensión no reconocida: {filepath.suffix!r}")
