"""RQ3 — ¿Qué información aporta el perfil funcional que no capturen la exactitud ni la longitud? Prueba inversa. Sin API.

Uso (después de analizar.py y estabilidad.py):
    python3 rq3.py [--permutaciones 300] [--bootstrap 2000]

Diseño, fijado antes de mirar los resultados
--------------------------------------------
Idea: si a partir de lo que el método extrae de una cadena se puede reconocer qué modelo la escribió mejor que con su longitud
y su acierto, el extracto conserva una huella de la forma de razonar. Se mide con clasificadores usados como instrumento.

- Unidad: cada una de las 200 cadenas (50 problemas × 4 modelos). Objetivos: (1) el modelo generador (4 clases, azar 0,25);
  (2) el dominio (2 clases, azar 0,50); (3) en medicina, el acierto (2 clases, 51/49).
- Rasgos. BASE = log(palabras), su cuadrado y el acierto, más lo que se sabe de la cadena y no es objetivo (el dominio al predecir
  el modelo; el modelo al predecir el dominio o el acierto). FUNCIONAL = cuotas de episodios por tipo y por desenlace, cuotas de
  relaciones, cobertura total y por tipo, densidad de episodios, organización (abiertos, diversidad, anidamiento, solapamiento,
  componentes, rutas, líneas) y posición de la deliberación. ESTRUCTURAL = cuotas de tipos de nodo y de arista, aristas por nodo,
  nodos por palabra, nodos ambiguos. FUNCIONAL PRIMARIO = los rasgos funcionales con ICC(1) ≥ 0,7 entre las tres corridas de
  estabilidad (regla prerregistrada del artículo); se calcula aquí mismo.
- Conjuntos comparados: base; base + funcional primario; base + funcional; base + estructural; base + todo; funcional solo; todo solo.
- Clasificador: regresión logística multinomial con penalización L2 sobre rasgos estandarizados (el escalado se ajusta solo con
  el pliegue de entrenamiento). Validación externa: dejar fuera un problema (sus cuatro cadenas) cada vez. Validación interna:
  cinco pliegues por problema dentro del entrenamiento para elegir C en {0,01; 0,1; 1; 10} por log-pérdida.
- Medidas sobre las predicciones fuera de muestra: exactitud, F1 macro, log-pérdida y sensibilidad por clase (AUC y exactitud
  balanceada en los binarios). Intervalos del 95 % por bootstrap de problemas (remuestreo por conglomerados), también para las
  ganancias frente a la base (pareadas). Prueba de McNemar exacta entre cada conjunto y la base sobre acierto/fallo por cadena;
  Wilcoxon sobre la log-pérdida media por problema.
- Permutaciones (objetivo modelo): (a) la ganancia de «base + todo» sobre «base» se compara con su distribución nula al barajar
  el bloque de perfil entre las cuatro cadenas de cada problema (rompe el vínculo perfil–modelo, conserva base y problema);
  (b) la exactitud de «todo solo» se compara con la nula al barajar las etiquetas de modelo dentro de cada problema.
- Robustez: bosque aleatorio (500 árboles) con la misma validación, para base y base + todo.
- Interpretación: ablación por grupos de rasgos (verificación, alternativas, correcciones, apoyo, progresión y organización,
  estructura), quitando cada grupo y usándolo solo con la base; coeficientes estandarizados del modelo ajustado a todas las cadenas.
- Expectativa declarada: ganancia grande para el modelo; nula o pequeña para el dominio; nula para el acierto.

Salidas en resultados/: rq3.json, tablas_rq3.md, rq3_predicciones.csv, rq3_rasgos.csv, figuras/rq3_*.png.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
import warnings
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RES = AQUI / "resultados"
FIG = RES / "figuras"
MODELOS = ["qwen3.8-27b", "gemma-4-31b-it", "olmo-3.1-32b-think", "deepseek-v4-pro"]
NOMBRE = {"qwen3.8-27b": "Qwen", "gemma-4-31b-it": "Gemma", "olmo-3.1-32b-think": "Olmo", "deepseek-v4-pro": "DeepSeek"}
TIPOS = ("PROGRESSION", "SUPPORT", "VERIFICATION", "CORRECTION", "ALTERNATIVE")
DELIB = ("VERIFICATION", "CORRECTION", "ALTERNATIVE")
NODOS = ("PROP", "CHECK", "EVID", "ALT", "METHOD", "OUT", "REVISE")
ARISTAS = ("SUPPORTS", "CHECKS", "LEADS_TO", "ALTERNATIVE_TO", "REVISES")
DESENLACES = (("VERIFICATION", "CONFIRMED"), ("VERIFICATION", "REFUTED"), ("VERIFICATION", "INCONCLUSIVE"),
              ("CORRECTION", "RECOVERED"), ("CORRECTION", "PARTIAL"),
              ("ALTERNATIVE", "REJECTED"), ("ALTERNATIVE", "UNRESOLVED"), ("ALTERNATIVE", "ADOPTED"),
              ("PROGRESSION", "REACHES_OUTPUT"))
RELS = ("EVALUATES", "REPLACES", "COMPETES_WITH", "SUPPORTS_EPISODE", "EMBEDS", "CONTINUES")
SEMILLA = 20261017
GRID_C = [0.01, 0.1, 1.0, 10.0]
UMBRAL_ICC = 0.7


# ----------------------------------------------------------------- rasgos
def envolvente(ep: dict) -> tuple[int, int]:
    return min(m["start_char"] for m in ep["members"]), max(m["end_char"] for m in ep["members"])


def rasgos(g: dict, palabras: int) -> dict:
    nodos = g["sgot_graph"]["nodes"]; n_n = len(nodos)
    aristas = [e for e in g["sgot_graph"]["edges"] if e["edge_type"] != "NEXT"]; n_a = len(aristas)
    eps = g["episodes"]; n_ep = len(eps)
    rels = [r for r in g["episode_relations"] if r["relation_type"] != "IMMEDIATELY_PRECEDES_ONSET"]; n_r = len(rels)
    p = g["profile"]; org = p["organization"]; cob = p["coverage"]["by_type"]
    largo = max(1, p["trace_length_chars"])
    ct = collections.Counter(e["episode_type"] for e in eps)
    cd = collections.Counter((e["episode_type"], e["outcome"]) for e in eps)
    cn = collections.Counter(n["node_type"] for n in nodos)
    ca = collections.Counter(e["edge_type"] for e in aristas)
    cr = collections.Counter(r["relation_type"] for r in rels)
    pos = [sum(envolvente(e)) / 2 / largo for e in eps if e["episode_type"] in DELIB]
    S, F = {}, {}
    for t in NODOS:
        S[f"cuota_nodo_{t}"] = cn[t] / n_n
    for t in ARISTAS:
        S[f"cuota_arista_{t}"] = ca[t] / max(1, n_a)
    S["aristas_por_nodo"] = n_a / n_n
    S["nodos_100pal"] = 100.0 * n_n / palabras
    S["nodos_ambiguos_frac"] = sum(1 for n in nodos if n.get("review_status") == "ambiguous") / n_n
    for t in TIPOS:
        F[f"cuota_ep_{t}"] = ct[t] / max(1, n_ep)
    for t, o in DESENLACES:
        F[f"cuota_des_{t}_{o}"] = cd[(t, o)] / max(1, n_ep)
    for t in RELS:
        F[f"cuota_rel_{t}"] = cr[t] / max(1, n_r)
    F["relaciones_por_ep"] = n_r / max(1, n_ep)
    F["cobertura"] = p["coverage"]["overall"]["ratio"]
    for t in TIPOS:
        F[f"cobertura_{t}"] = cob[t]["ratio"]
    F["episodios_100n"] = 100.0 * n_ep / n_n
    F["abiertos_frac"] = org["unresolved_or_inconclusive_episode_count"] / max(1, n_ep)
    F["diversidad_funcional"] = org["functional_diversity"]
    F["anidados_por_ep"] = org["nested_episode_pair_count"] / max(1, n_ep)
    F["solapados_por_ep"] = org["overlapping_episode_pair_count"] / max(1, n_ep)
    F["no_contiguos_frac"] = org["noncontiguous_episode_count"] / max(1, n_ep)
    F["componentes_por_ep"] = org["functional_component_count"] / max(1, n_ep)
    F["ruta_episodios_media"] = p["route_structure"]["mean_episode_count"]
    F["lineas_por_ep"] = org["progression_line_count"] / max(1, n_ep)
    F["pos_delib"] = float(np.mean(pos)) if pos else 0.5
    F["sin_delib"] = 0.0 if pos else 1.0
    return {"S": S, "F": F}


def cargar(ruta: Path, ids: set[str] | None = None) -> dict[str, dict]:
    out = {}
    for l in gzip.open(ruta, "rt", encoding="utf-8"):
        g = json.loads(l)
        if ids is None or g["trace_id"] in ids:
            out[g["trace_id"]] = g
    return out


def construir() -> tuple[list[dict], list[str], list[str]]:
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(AQUI / "cadenas_aceptadas.jsonl", encoding="utf-8") if l.strip()}
    filas = []
    for tid, g in sorted(cargar(RES / "anotacion_200.jsonl.gz").items()):
        m = meta[tid]
        r = rasgos(g, int(m["palabras_razonamiento"]))
        filas.append({"id": tid, "problema": m["problem_id"], "dominio": m["domain"], "modelo": tid.split("--")[1],
                      "correcto": str(m["correcto"]) == "True", "palabras": int(m["palabras_razonamiento"]), **r["F"], **r["S"]})
    F = list(rasgos(next(iter(cargar(RES / "anotacion_200.jsonl.gz").values())), 100)["F"])
    S = list(rasgos(next(iter(cargar(RES / "anotacion_200.jsonl.gz").values())), 100)["S"])
    return filas, F, S


def icc1(M: np.ndarray) -> float:
    n, k = M.shape
    medias = M.mean(axis=1); gran = M.mean()
    msb = k * ((medias - gran) ** 2).sum() / (n - 1)
    msw = ((M - medias[:, None]) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw)) if (msb + (k - 1) * msw) > 0 else float("nan")


def fiabilidad_rasgos(F: list[str], S: list[str]) -> dict:
    """ICC(1) de cada rasgo entre las tres corridas de las 20 cadenas de estabilidad."""
    ids = sorted(json.load(open(RES / "estabilidad_ids.json")))
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(AQUI / "cadenas_aceptadas.jsonl", encoding="utf-8") if l.strip()}
    corridas = []
    for f in ("anotacion_200.jsonl.gz", "estabilidad_r1.jsonl.gz", "estabilidad_r2.jsonl.gz"):
        c = cargar(RES / f, set(ids))
        corridas.append({t: rasgos(c[t], int(meta[t]["palabras_razonamiento"])) for t in ids})
    out = {}
    for capa, nombres in (("F", F), ("S", S)):
        for k in nombres:
            M = np.array([[c[t][capa][k] for c in corridas] for t in ids])
            out[k] = icc1(M) if M.std() > 0 else float("nan")
    return out


# ----------------------------------------------------------------- clasificación
def matriz(filas: list[dict], columnas: list[str]) -> np.ndarray:
    return np.array([[float(m[c]) for c in columnas] for m in filas])


def base_para(objetivo: str, filas: list[dict]) -> tuple[np.ndarray, list[str]]:
    cols, X = [], []
    for m in filas:
        lw = math.log(m["palabras"])
        fila = [lw, lw * lw]
        nombres = ["log_palabras", "log_palabras2"]
        if objetivo != "acierto":
            fila.append(1.0 if m["correcto"] else 0.0); nombres.append("acierto")
        if objetivo == "modelo":
            fila.append(1.0 if m["dominio"] == "derecho" else 0.0); nombres.append("dominio_derecho")
        else:
            for mo in MODELOS[1:]:
                fila.append(1.0 if m["modelo"] == mo else 0.0); nombres.append(f"modelo_{NOMBRE[mo]}")
        X.append(fila); cols = nombres
    return np.array(X), cols


def ajustar(Xtr: np.ndarray, ytr: np.ndarray, C: float, semilla: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=C, penalty="l2", multi_class="multinomial", solver="lbfgs", max_iter=5000, random_state=semilla)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(sc.transform(Xtr), ytr)
    return sc, clf


def elegir_C(Xtr: np.ndarray, ytr: np.ndarray, gtr: np.ndarray, semilla: int) -> float:
    """Validación interna por problema (5 pliegues): C con menor log-pérdida."""
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import log_loss
    k = len(np.unique(ytr))
    mejor, mejor_ll = GRID_C[0], float("inf")
    for C in GRID_C:
        P = np.zeros((len(ytr), k))
        for tr, te in GroupKFold(n_splits=5).split(Xtr, ytr, gtr):
            sc, clf = ajustar(Xtr[tr], ytr[tr], C, semilla)
            P[te] = clf.predict_proba(sc.transform(Xtr[te]))
        ll = log_loss(ytr, np.clip(P, 1e-9, 1), labels=list(range(k)))
        if ll < mejor_ll:
            mejor, mejor_ll = C, ll
    return mejor


def validar(X: np.ndarray, y: np.ndarray, grupos: np.ndarray, semilla: int, C_fijo: float | None = None) -> tuple[np.ndarray, list[float]]:
    """Dejar un problema fuera. Devuelve probabilidades fuera de muestra (n × k) y las C elegidas por pliegue."""
    from sklearn.model_selection import LeaveOneGroupOut
    k = len(np.unique(y))
    P = np.zeros((len(y), k)); Cs = []
    for tr, te in LeaveOneGroupOut().split(X, y, grupos):
        C = C_fijo if C_fijo is not None else elegir_C(X[tr], y[tr], grupos[tr], semilla)
        sc, clf = ajustar(X[tr], y[tr], C, semilla)
        P[te] = clf.predict_proba(sc.transform(X[te])); Cs.append(C)
    return P, Cs


def validar_bosque(X: np.ndarray, y: np.ndarray, grupos: np.ndarray, semilla: int) -> np.ndarray:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import LeaveOneGroupOut
    k = len(np.unique(y)); P = np.zeros((len(y), k))
    for tr, te in LeaveOneGroupOut().split(X, y, grupos):
        rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, random_state=semilla, n_jobs=-1).fit(X[tr], y[tr])
        P[te] = rf.predict_proba(X[te])
    return P


def metricas(y: np.ndarray, P: np.ndarray, clases: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, log_loss, balanced_accuracy_score, roc_auc_score
    k = len(clases); pred = P.argmax(axis=1)
    out = {"exactitud": float(accuracy_score(y, pred)), "f1_macro": float(f1_score(y, pred, average="macro")),
           "log_perdida": float(log_loss(y, np.clip(P, 1e-9, 1), labels=list(range(k)))),
           "sensibilidad_por_clase": {clases[c]: float((pred[y == c] == c).mean()) for c in range(k)}}
    if k == 2:
        out["exactitud_balanceada"] = float(balanced_accuracy_score(y, pred))
        out["auc"] = float(roc_auc_score(y, P[:, 1]))
    return out


def confusion(y: np.ndarray, P: np.ndarray, clases: list[str]) -> dict:
    pred = P.argmax(axis=1)
    return {clases[a]: {clases[b]: int(((y == a) & (pred == b)).sum()) for b in range(len(clases))} for a in range(len(clases))}


def bootstrap_problemas(y: np.ndarray, Ps: dict[str, np.ndarray], grupos: np.ndarray, clases: list[str], B: int, rng: np.random.Generator, base: str) -> dict:
    problemas = np.unique(grupos)
    indices = {p: np.where(grupos == p)[0] for p in problemas}
    claves = ["exactitud", "f1_macro", "log_perdida"] + (["auc", "exactitud_balanceada"] if len(clases) == 2 else [])
    acum = {s: {k: [] for k in claves} for s in Ps}
    gan = {s: {k: [] for k in claves} for s in Ps if s != base}
    for _ in range(B):
        sel = rng.choice(problemas, size=len(problemas), replace=True)
        idx = np.concatenate([indices[p] for p in sel])
        if len(np.unique(y[idx])) < len(clases):
            continue
        ms = {s: metricas(y[idx], P[idx], clases) for s, P in Ps.items()}
        for s in Ps:
            for k in claves:
                acum[s][k].append(ms[s][k])
                if s != base:
                    gan[s][k].append(ms[s][k] - ms[base][k])
    ic = lambda xs: [float(np.percentile(xs, 2.5)), float(np.percentile(xs, 97.5))]
    return {"ic_metricas": {s: {k: ic(v) for k, v in d.items()} for s, d in acum.items()},
            "ic_ganancias_vs_base": {s: {k: ic(v) for k, v in d.items()} for s, d in gan.items()}}


def mcnemar(y: np.ndarray, Pa: np.ndarray, Pb: np.ndarray) -> dict:
    from scipy import stats
    a = Pa.argmax(axis=1) == y; b = Pb.argmax(axis=1) == y
    solo_a, solo_b = int((a & ~b).sum()), int((~a & b).sum())
    n = solo_a + solo_b
    p = float(stats.binomtest(min(solo_a, solo_b), n, 0.5).pvalue) if n > 0 else 1.0
    return {"solo_base_acierta": solo_a, "solo_conjunto_acierta": solo_b, "p": p}


def wilcoxon_logperdida(y: np.ndarray, Pa: np.ndarray, Pb: np.ndarray, grupos: np.ndarray) -> dict:
    from scipy import stats
    la = -np.log(np.clip(Pa[np.arange(len(y)), y], 1e-9, 1)); lb = -np.log(np.clip(Pb[np.arange(len(y)), y], 1e-9, 1))
    difs = [float(lb[grupos == p].mean() - la[grupos == p].mean()) for p in np.unique(grupos)]
    d = [x for x in difs if x != 0]
    return {"mediana_dif_logperdida_por_problema": float(np.median(difs)), "problemas_mejora": int(sum(x < 0 for x in difs)),
            "problemas_empeora": int(sum(x > 0 for x in difs)), "p": float(stats.wilcoxon(d).pvalue) if len(d) >= 6 else float("nan")}


def moda(xs: list[float]) -> float:
    return collections.Counter(xs).most_common(1)[0][0]


# ----------------------------------------------------------------- experimento por objetivo
def experimento(objetivo: str, filas: list[dict], F: list[str], S: list[str], primarios: list[str], args, rng: np.random.Generator) -> dict:
    if objetivo == "acierto":
        filas = [m for m in filas if m["dominio"] == "medicina"]
    if objetivo == "modelo":
        clases = [NOMBRE[mo] for mo in MODELOS]; y = np.array([MODELOS.index(m["modelo"]) for m in filas])
    elif objetivo == "dominio":
        clases = ["medicina", "derecho"]; y = np.array([1 if m["dominio"] == "derecho" else 0 for m in filas])
    else:
        clases = ["falla", "acierta"]; y = np.array([1 if m["correcto"] else 0 for m in filas])
    grupos = np.array([m["problema"] for m in filas])
    Xb, cols_b = base_para(objetivo, filas)
    XF, XS, XP = matriz(filas, F), matriz(filas, S), matriz(filas, primarios)
    conjuntos = {
        "base": Xb,
        "base+funcional_primario": np.hstack([Xb, XP]),
        "base+funcional": np.hstack([Xb, XF]),
        "base+estructural": np.hstack([Xb, XS]),
        "base+todo": np.hstack([Xb, XF, XS]),
        "funcional_solo": XF,
        "todo_solo": np.hstack([XF, XS]),
    }
    res = {"objetivo": objetivo, "n": len(filas), "clases": clases, "azar": 1.0 / len(clases), "columnas_base": cols_b, "conjuntos": {}}
    Ps = {}
    for nombre, X in conjuntos.items():
        P, Cs = validar(X, y, grupos, SEMILLA)
        Ps[nombre] = P
        res["conjuntos"][nombre] = {"n_rasgos": X.shape[1], "C_elegidas": dict(collections.Counter(Cs)), **metricas(y, P, clases),
                                    "confusion": confusion(y, P, clases)}
        if nombre != "base":
            res["conjuntos"][nombre]["mcnemar_vs_base"] = mcnemar(y, Ps["base"], P)
            res["conjuntos"][nombre]["wilcoxon_logperdida_vs_base"] = wilcoxon_logperdida(y, Ps["base"], P, grupos)
        print(f"  {objetivo:8s} {nombre:24s} rasgos {X.shape[1]:3d}  exactitud {res['conjuntos'][nombre]['exactitud']:.3f}  log-pérdida {res['conjuntos'][nombre]['log_perdida']:.3f}", flush=True)
    res["bootstrap"] = bootstrap_problemas(y, Ps, grupos, clases, args.bootstrap, rng, "base")

    # bosque aleatorio
    res["bosque"] = {}
    for nombre in ("base", "base+todo"):
        P = validar_bosque(conjuntos[nombre], y, grupos, SEMILLA)
        res["bosque"][nombre] = {**metricas(y, P, clases)}
        Ps[f"bosque:{nombre}"] = P
    bb = bootstrap_problemas(y, {"bosque:base": Ps["bosque:base"], "bosque:base+todo": Ps["bosque:base+todo"]}, grupos, clases, args.bootstrap, rng, "bosque:base")
    res["bosque"]["ic"] = bb
    res["bosque"]["mcnemar_todo_vs_base"] = mcnemar(y, Ps["bosque:base"], Ps["bosque:base+todo"])

    # predicciones fuera de muestra (para el CSV)
    res["_predicciones"] = {nombre: P for nombre, P in Ps.items()}
    res["_y"] = y; res["_ids"] = [m["id"] for m in filas]

    if objetivo != "modelo":
        return res

    # permutaciones (solo para el modelo)
    C_todo = moda(list(np.repeat(list(res["conjuntos"]["base+todo"]["C_elegidas"].keys()), list(res["conjuntos"]["base+todo"]["C_elegidas"].values()))))
    C_solo = moda(list(np.repeat(list(res["conjuntos"]["todo_solo"]["C_elegidas"].keys()), list(res["conjuntos"]["todo_solo"]["C_elegidas"].values()))))
    exact_base = res["conjuntos"]["base"]["exactitud"]
    gan_obs = res["conjuntos"]["base+todo"]["exactitud"] - exact_base
    exact_solo_obs = res["conjuntos"]["todo_solo"]["exactitud"]
    perfil = np.hstack([XF, XS])
    nulas_gan, nulas_solo = [], []
    idx_por_problema = {p: np.where(grupos == p)[0] for p in np.unique(grupos)}
    for i in range(args.permutaciones):
        perm = np.arange(len(y)); perm_y = y.copy()
        for p, idx in idx_por_problema.items():
            perm[idx] = idx[rng.permutation(len(idx))]
            perm_y[idx] = y[idx][rng.permutation(len(idx))]
        P, _ = validar(np.hstack([Xb, perfil[perm]]), y, grupos, SEMILLA + i, C_fijo=C_todo)
        nulas_gan.append(float((P.argmax(axis=1) == y).mean()) - exact_base)
        P, _ = validar(perfil, perm_y, grupos, SEMILLA + i, C_fijo=C_solo)
        nulas_solo.append(float((P.argmax(axis=1) == perm_y).mean()))
        if (i + 1) % 50 == 0:
            print(f"  permutaciones {i + 1}/{args.permutaciones}", flush=True)
    res["permutaciones"] = {
        "n": args.permutaciones, "C_fija_base_todo": C_todo, "C_fija_todo_solo": C_solo,
        "ganancia_observada_base_todo": gan_obs, "ganancia_nula_media": float(np.mean(nulas_gan)), "ganancia_nula_p95": float(np.percentile(nulas_gan, 95)),
        "p_ganancia": float((1 + sum(g >= gan_obs for g in nulas_gan)) / (1 + len(nulas_gan))),
        "exactitud_observada_todo_solo": exact_solo_obs, "exactitud_nula_media": float(np.mean(nulas_solo)), "exactitud_nula_p95": float(np.percentile(nulas_solo, 95)),
        "p_todo_solo": float((1 + sum(e >= exact_solo_obs for e in nulas_solo)) / (1 + len(nulas_solo))),
    }

    # ablación por grupos (base + todo, quitando cada grupo; y base + solo ese grupo)
    grupos_rasgos = {
        "verificación": [c for c in F if "VERIFICATION" in c or c == "cuota_rel_EVALUATES"],
        "alternativas": [c for c in F if "ALTERNATIVE" in c or c == "cuota_rel_COMPETES_WITH"],
        "correcciones": [c for c in F if "CORRECTION" in c or c == "cuota_rel_REPLACES"],
        "apoyo": [c for c in F if "SUPPORT" in c],
        "progresión y organización": [c for c in F if not any(t in c for t in ("VERIFICATION", "ALTERNATIVE", "CORRECTION", "SUPPORT")) and c not in ("cuota_rel_EVALUATES", "cuota_rel_COMPETES_WITH", "cuota_rel_REPLACES")],
        "estructura": list(S),
    }
    todas = F + S
    res["ablacion"] = {"grupos": grupos_rasgos, "resultados": {}}
    Xtodo = np.hstack([XF, XS])
    for nombre, cols in grupos_rasgos.items():
        sin = [i for i, c in enumerate(todas) if c not in cols]; con = [i for i, c in enumerate(todas) if c in cols]
        P_sin, _ = validar(np.hstack([Xb, Xtodo[:, sin]]), y, grupos, SEMILLA)
        P_con, _ = validar(np.hstack([Xb, Xtodo[:, con]]), y, grupos, SEMILLA)
        res["ablacion"]["resultados"][nombre] = {
            "n_rasgos": len(cols),
            "sin_el_grupo": {**{k: v for k, v in metricas(y, P_sin, clases).items() if k != "sensibilidad_por_clase"}, "mcnemar_vs_base_todo": mcnemar(y, Ps["base+todo"], P_sin)},
            "base_mas_solo_el_grupo": {**metricas(y, P_con, clases), "mcnemar_vs_base": mcnemar(y, Ps["base"], P_con)},
        }
        print(f"  ablación {nombre:26s} sin: {res['ablacion']['resultados'][nombre]['sin_el_grupo']['exactitud']:.3f}  base+solo: {res['ablacion']['resultados'][nombre]['base_mas_solo_el_grupo']['exactitud']:.3f}", flush=True)

    # coeficientes estandarizados del modelo ajustado a todas las cadenas (base + todo)
    cols_todo = cols_b + todas
    sc, clf = ajustar(np.hstack([Xb, Xtodo]), y, C_todo, SEMILLA)
    coef = clf.coef_  # k × p, clases en orden 0..k-1
    res["coeficientes"] = {"C": C_todo, "por_clase": {}}
    for c, nombre in enumerate(clases):
        orden = np.argsort(-np.abs(coef[c]))[:8]
        res["coeficientes"]["por_clase"][nombre] = [{"rasgo": cols_todo[j], "coef": float(coef[c, j])} for j in orden]
    return res


# ----------------------------------------------------------------- tablas y figuras
def f(x, d=3) -> str:
    if x is None or (isinstance(x, float) and x != x):
        return "—"
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def fp(p) -> str:
    if p is None or p != p:
        return "—"
    return "< 0,001" if p < 0.001 else f(p, 3)


def escribir_tablas(out: dict, F: list[str], S: list[str]) -> None:
    L = ["# RQ3 — tablas (generadas por rq3.py)", ""]
    L += [f"## 0. Fiabilidad de los rasgos (ICC(1), 20 cadenas × 3 corridas) y rasgos funcionales primarios (ICC ≥ {f(UMBRAL_ICC, 1)})", "",
          "| Rasgo | Capa | ICC(1) | primario |", "|---|---|---|---|"]
    for k, v in sorted(out["icc_rasgos"].items(), key=lambda kv: -(kv[1] if kv[1] == kv[1] else -9)):
        L.append(f"| {k} | {'funcional' if k in F else 'estructural'} | {f(v, 2)} | {'sí' if k in out['primarios'] else ''} |")
    L.append("")
    for obj in ("modelo", "dominio", "acierto"):
        r = out[obj]
        L += [f"## {obj.capitalize()} (n = {r['n']}; clases: {', '.join(r['clases'])}; azar {f(r['azar'], 2)}; base: {', '.join(r['columnas_base'])})", "",
              "| Conjunto de rasgos | rasgos | exactitud [IC 95 %] | ganancia vs base [IC 95 %] | F1 macro | log-pérdida [IC] | McNemar vs base (solo base / solo conjunto; p) | Wilcoxon log-pérdida por problema (mejora/empeora; p) |" + (" AUC [IC] |" if len(r["clases"]) == 2 else ""),
              "|---|---|---|---|---|---|---|---|" + ("---|" if len(r["clases"]) == 2 else "")]
        for nombre, c in r["conjuntos"].items():
            ic = r["bootstrap"]["ic_metricas"][nombre]; g = r["bootstrap"]["ic_ganancias_vs_base"].get(nombre)
            mc = c.get("mcnemar_vs_base"); wl = c.get("wilcoxon_logperdida_vs_base")
            fila = (f"| {nombre} | {c['n_rasgos']} | {f(c['exactitud'])} [{f(ic['exactitud'][0])}–{f(ic['exactitud'][1])}] | "
                    + (f"{f(c['exactitud'] - r['conjuntos']['base']['exactitud'])} [{f(g['exactitud'][0])}–{f(g['exactitud'][1])}]" if g else "—")
                    + f" | {f(c['f1_macro'])} | {f(c['log_perdida'])} [{f(ic['log_perdida'][0])}–{f(ic['log_perdida'][1])}] | "
                    + (f"{mc['solo_base_acierta']} / {mc['solo_conjunto_acierta']}; {fp(mc['p'])}" if mc else "—")
                    + " | " + (f"{wl['problemas_mejora']}/{wl['problemas_empeora']}; {fp(wl['p'])}" if wl else "—") + " |")
            if len(r["clases"]) == 2:
                fila += f" {f(c['auc'])} [{f(ic['auc'][0])}–{f(ic['auc'][1])}] |"
            L.append(fila)
        L += ["", "Sensibilidad por clase (fracción de cadenas de cada clase reconocidas):", "", "| Conjunto | " + " | ".join(r["clases"]) + " |", "|---|" + "---|" * len(r["clases"])]
        for nombre, c in r["conjuntos"].items():
            L.append(f"| {nombre} | " + " | ".join(f(c["sensibilidad_por_clase"][k], 2) for k in r["clases"]) + " |")
        L += ["", "Bosque aleatorio (500 árboles, misma validación):", "", "| Conjunto | exactitud [IC 95 %] | F1 macro | log-pérdida |", "|---|---|---|---|"]
        for nombre in ("base", "base+todo"):
            b = r["bosque"][nombre]; ic = r["bosque"]["ic"]["ic_metricas"][f"bosque:{nombre}"]
            L.append(f"| {nombre} | {f(b['exactitud'])} [{f(ic['exactitud'][0])}–{f(ic['exactitud'][1])}] | {f(b['f1_macro'])} | {f(b['log_perdida'])} |")
        mcb = r["bosque"]["mcnemar_todo_vs_base"]; gb = r["bosque"]["ic"]["ic_ganancias_vs_base"]["bosque:base+todo"]["exactitud"]
        L.append(f"\nGanancia del bosque base+todo sobre base: {f(r['bosque']['base+todo']['exactitud'] - r['bosque']['base']['exactitud'])} [{f(gb[0])}–{f(gb[1])}]; McNemar {mcb['solo_base_acierta']} / {mcb['solo_conjunto_acierta']}, p = {fp(mcb['p'])}.")
        L += ["", "Matriz de confusión (filas: clase real; columnas: predicha) para base y base+todo:", ""]
        for nombre in ("base", "base+todo"):
            cf = r["conjuntos"][nombre]["confusion"]
            L += [f"*{nombre}*", "", "| real \\ predicha | " + " | ".join(r["clases"]) + " |", "|---|" + "---|" * len(r["clases"])]
            for a in r["clases"]:
                L.append(f"| {a} | " + " | ".join(str(cf[a][b]) for b in r["clases"]) + " |")
            L.append("")
        if obj == "modelo":
            pm = r["permutaciones"]
            L += [f"### Permutaciones ({pm['n']})", "",
                  f"- Ganancia de base+todo sobre base: observada {f(pm['ganancia_observada_base_todo'])}; nula (perfil barajado dentro de cada problema) media {f(pm['ganancia_nula_media'])}, percentil 95 {f(pm['ganancia_nula_p95'])}; p = {fp(pm['p_ganancia'])}.",
                  f"- Exactitud de todo_solo: observada {f(pm['exactitud_observada_todo_solo'])}; nula (etiquetas barajadas dentro de cada problema) media {f(pm['exactitud_nula_media'])}, percentil 95 {f(pm['exactitud_nula_p95'])}; p = {fp(pm['p_todo_solo'])}.", ""]
            L += ["### Ablación por grupos de rasgos (regresión logística, validación por problema)", "",
                  "| Grupo | rasgos | exactitud sin el grupo (base+todo = " + f(r["conjuntos"]["base+todo"]["exactitud"]) + ") | McNemar vs base+todo (p) | exactitud base + solo el grupo (base = " + f(r["conjuntos"]["base"]["exactitud"]) + ") | McNemar vs base (p) |",
                  "|---|---|---|---|---|---|"]
            for nombre, a in r["ablacion"]["resultados"].items():
                L.append(f"| {nombre} | {a['n_rasgos']} | {f(a['sin_el_grupo']['exactitud'])} | {fp(a['sin_el_grupo']['mcnemar_vs_base_todo']['p'])} | {f(a['base_mas_solo_el_grupo']['exactitud'])} | {fp(a['base_mas_solo_el_grupo']['mcnemar_vs_base']['p'])} |")
            L += ["", f"### Coeficientes estandarizados (modelo base+todo ajustado a las 200 cadenas, C = {r['coeficientes']['C']}); los ocho mayores en valor absoluto por clase", ""]
            for clase, lst in r["coeficientes"]["por_clase"].items():
                L.append(f"- **{clase}:** " + "; ".join(f"{d['rasgo']} ({f(d['coef'], 2)})" for d in lst))
            L.append("")
    (RES / "tablas_rq3.md").write_text("\n".join(L), encoding="utf-8")


def figuras(out: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    FIG.mkdir(exist_ok=True)
    conj = ["base", "base+funcional_primario", "base+funcional", "base+estructural", "base+todo", "funcional_solo", "todo_solo"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, obj in zip(axes, ("modelo", "dominio", "acierto")):
        r = out[obj]
        ex = [r["conjuntos"][c]["exactitud"] for c in conj]
        ic = [r["bootstrap"]["ic_metricas"][c]["exactitud"] for c in conj]
        err = [[e - i[0] for e, i in zip(ex, ic)], [i[1] - e for e, i in zip(ex, ic)]]
        ax.bar(range(len(conj)), ex, yerr=err, capsize=3, color=["#7f7f7f", "#9ecae1", "#3182bd", "#fdae6b", "#e6550d", "#31a354", "#006d2c"])
        ax.axhline(r["azar"], color="k", ls="--", lw=0.8)
        ax.set_xticks(range(len(conj))); ax.set_xticklabels(conj, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1); ax.set_title(f"Predecir {obj} (n = {r['n']})"); ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("exactitud fuera de muestra [IC 95 %]")
    fig.tight_layout(); fig.savefig(FIG / "rq3_exactitud.png", dpi=150); plt.close(fig)
    r = out["modelo"]; clases = r["clases"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, nombre in zip(axes, ("base", "base+todo")):
        cf = np.array([[r["conjuntos"][nombre]["confusion"][a][b] for b in clases] for a in clases])
        ax.imshow(cf, cmap="Blues", vmin=0, vmax=50)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, str(cf[i, j]), ha="center", va="center", color="white" if cf[i, j] > 25 else "black")
        ax.set_xticks(range(4)); ax.set_xticklabels(clases); ax.set_yticks(range(4)); ax.set_yticklabels(clases)
        ax.set_xlabel("modelo predicho"); ax.set_title(f"{nombre}: exactitud {f(r['conjuntos'][nombre]['exactitud'], 2)}")
    axes[0].set_ylabel("modelo real")
    fig.tight_layout(); fig.savefig(FIG / "rq3_confusion_modelo.png", dpi=150); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--permutaciones", type=int, default=300)
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEMILLA)
    filas, F, S = construir()
    icc = fiabilidad_rasgos(F, S)
    primarios = [k for k in F if icc[k] == icc[k] and icc[k] >= UMBRAL_ICC]
    print(f"{len(filas)} cadenas; rasgos funcionales {len(F)} (primarios {len(primarios)}: {primarios}); estructurales {len(S)}", flush=True)
    with open(RES / "rq3_rasgos.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(filas[0])); w.writeheader(); w.writerows(filas)
    out = {"n_cadenas": len(filas), "semilla": SEMILLA, "grid_C": GRID_C, "rasgos_funcionales": F, "rasgos_estructurales": S,
           "icc_rasgos": icc, "primarios": primarios}
    pred_filas = []
    for obj in ("modelo", "dominio", "acierto"):
        print(f"objetivo: {obj}", flush=True)
        r = experimento(obj, filas, F, S, primarios, args, rng)
        for nombre, P in r.pop("_predicciones").items():
            for i, tid in enumerate(r["_ids"]):
                pred_filas.append({"objetivo": obj, "conjunto": nombre, "id": tid, "real": r["clases"][r["_y"][i]],
                                   "predicha": r["clases"][int(P[i].argmax())], **{f"p_{c}": round(float(P[i][j]), 4) for j, c in enumerate(r["clases"])}})
        r.pop("_y"); r.pop("_ids")
        out[obj] = r
    with open(RES / "rq3_predicciones.csv", "w", newline="", encoding="utf-8") as fh:
        campos = sorted({k for p in pred_filas for k in p}, key=lambda k: (k.startswith("p_"), k))
        w = csv.DictWriter(fh, fieldnames=campos); w.writeheader(); w.writerows(pred_filas)
    json.dump(out, open(RES / "rq3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    escribir_tablas(out, F, S)
    figuras(out)
    print("→ resultados/rq3.json, tablas_rq3.md, rq3_predicciones.csv, rq3_rasgos.csv, figuras/rq3_*.png", flush=True)


if __name__ == "__main__":
    main()
