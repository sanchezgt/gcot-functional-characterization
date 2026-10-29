"""Inferencia de RQ2 para el artículo: un modelo de recuentos unificado, intervalos por bootstrap, fiabilidad por cadena,
sensibilidad a la longitud y corrección múltiple global. Sin API. Se corre después de analizar.py y estabilidad.py.

Uso:
    python3 inferencia.py

Entradas: resultados/metricas_por_cadena.csv (analizar.py), resultados/analisis_rq2.json (las p pareadas),
resultados/anotacion_200.jsonl.gz, estabilidad_r1.jsonl.gz, estabilidad_r2.jsonl.gz (para la fiabilidad).
Salidas: resultados/inferencia_rq2.json y resultados/tablas_inferencia.md.

Qué hace:
1. Regresión binomial negativa (NB2) del número de episodios de cada tipo sobre log(nodos), modelo (referencia Qwen) y
   dominio (referencia medicina), con errores estándar robustos agrupados por problema (cada problema aporta cuatro
   cadenas). Razones de tasa con IC 95 %, contrastes entre todos los pares de modelos (Holm), y prueba de Wald de la
   interacción modelo × dominio (¿el perfil cambia con el dominio?). En medicina, el mismo modelo con el acierto.
2. Bootstrap (percentil) de las medianas por celda y de la δ de Cliff de cada par de modelos, remuestreando problemas.
3. Fiabilidad por cadena entre las tres corridas de estabilidad: ICC(1) de una corrida e ICC(1,3) de la media, con IC por
   bootstrap sobre las 20 cadenas, y el factor de atenuación que implican.
4. Sensibilidad a la longitud: pares de cadenas del mismo problema con tamaños de grafo parecidos (cociente ≤ 1,5) y la
   regresión restringida a la banda común de nodos.
5. Corrección múltiple global (Benjamini-Hochberg) sobre todas las comparaciones pareadas de analizar.py.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import statistics
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

AQUI = Path(__file__).resolve().parent
RES = AQUI / "resultados"
MODELOS = ["qwen3.8-27b", "gemma-4-31b-it", "olmo-3.1-32b-think", "deepseek-v4-pro"]
NOMBRE = {"qwen3.8-27b": "Qwen", "gemma-4-31b-it": "Gemma", "olmo-3.1-32b-think": "Olmo", "deepseek-v4-pro": "DeepSeek"}
DOMINIOS = ["medicina", "derecho"]
RECUENTOS = [("n_VERIFICATION", "VERIFICATION"), ("n_CORRECTION", "CORRECTION"), ("n_ALTERNATIVE", "ALTERNATIVE"),
             ("n_SUPPORT", "SUPPORT"), ("deliberativos", "deliberativos (VER+COR+ALT)"), ("episodios", "episodios (todos)")]
TASAS = ["delib_100n", "ver_100n", "cor_100n", "alt_100n", "sup_100n", "cobertura"]
SEMILLA = 20261017
B_MEDIANAS, B_DELTA, B_ICC = 5000, 2000, 5000
BANDA_NODOS = (50, 200)  # banda común de tamaño de grafo para la sensibilidad
COCIENTE_MAX = 1.5       # cociente máximo de nodos entre dos cadenas del mismo problema para considerarlas emparejadas


def cargar_filas() -> list[dict]:
    filas = []
    for m in csv.DictReader(open(RES / "metricas_por_cadena.csv", encoding="utf-8")):
        for k, v in m.items():
            if k in ("id", "problema", "dominio", "modelo", "estrato", "subarea"):
                continue
            if k == "correcto":
                m[k] = v == "True"
            else:
                try:
                    m[k] = float(v)
                except ValueError:
                    pass
        filas.append(m)
    return filas


# ----------------------------------------------------------------- 1. regresión de recuentos
def diseno(fs: list[dict], interaccion: bool, acierto: bool) -> tuple[np.ndarray, list[str]]:
    nombres = ["const", "log_nodos"] + [f"modelo_{NOMBRE[mo]}" for mo in MODELOS[1:]]
    dos_dominios = len({m["dominio"] for m in fs}) > 1
    if dos_dominios:
        nombres.append("derecho")
        if interaccion:
            nombres += [f"modelo_{NOMBRE[mo]}×derecho" for mo in MODELOS[1:]]
    if acierto:
        nombres.append("acierta")
    X = []
    for m in fs:
        der = 1.0 if m["dominio"] == "derecho" else 0.0
        fila = [1.0, math.log(m["nodos"])] + [1.0 if m["modelo"] == mo else 0.0 for mo in MODELOS[1:]]
        if dos_dominios:
            fila.append(der)
            if interaccion:
                fila += [der * (1.0 if m["modelo"] == mo else 0.0) for mo in MODELOS[1:]]
        if acierto:
            fila.append(1.0 if m["correcto"] else 0.0)
        X.append(fila)
    return np.array(X), nombres


def ajustar_nb(y: np.ndarray, X: np.ndarray, grupos: np.ndarray):
    """Binomial negativa NB2 con errores robustos agrupados; si no converge, Poisson robusto. Devuelve (resultado, familia)."""
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            inicio = Poisson(y, X).fit(disp=0, maxiter=200).params
            r = NegativeBinomial(y, X, loglike_method="nb2").fit(start_params=np.append(inicio, 0.5), disp=0, maxiter=500,
                                                                  cov_type="cluster", cov_kwds={"groups": grupos})
            if r.mle_retvals.get("converged", True) and np.all(np.isfinite(r.bse)):
                return r, "binomial negativa (NB2)"
        except Exception:  # pragma: no cover - caída a Poisson
            pass
        r = Poisson(y, X).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": grupos})
        return r, "Poisson"


def resumen_modelo(r, nombres: list[str]) -> dict:
    """Razones de tasa (exp β) con IC 95 % y p, por término."""
    out = {}
    ci = r.conf_int()
    for i, n in enumerate(nombres):
        b, lo, hi, p = r.params[i], ci[i][0], ci[i][1], r.pvalues[i]
        out[n] = {"beta": float(b), "rr": float(math.exp(b)) if n != "const" else None, "ic95": [float(math.exp(lo)), float(math.exp(hi))] if n != "const" else None,
                  "p": float(p)}
    if "alpha" in getattr(r.model, "exog_names", []) or len(r.params) > len(nombres):
        out["alpha_dispersion"] = float(r.params[-1])
    return out


def holm(ps: list[float]) -> list[float]:
    orden = sorted(range(len(ps)), key=lambda i: ps[i])
    ajust, corriente = [0.0] * len(ps), 0.0
    for k, i in enumerate(orden):
        corriente = max(corriente, min(1.0, ps[i] * (len(ps) - k)))
        ajust[i] = corriente
    return ajust


def bh(ps: list[float]) -> list[float]:
    n = len(ps)
    orden = sorted(range(n), key=lambda i: ps[i])
    q, minimo = [0.0] * n, 1.0
    for k in range(n - 1, -1, -1):
        i = orden[k]
        minimo = min(minimo, ps[i] * n / (k + 1))
        q[i] = minimo
    return q


def contrastes_modelos(r, nombres: list[str]) -> list[dict]:
    """Todos los pares de modelos: diferencia de log-tasas (A − B) por t de Wald con la covarianza robusta; Holm sobre los 6."""
    k = len(r.params)
    idx = {n: i for i, n in enumerate(nombres)}
    pares, ps = [], []
    for a, b in combinations(MODELOS, 2):
        v = np.zeros(k)
        if a != MODELOS[0]:
            v[idx[f"modelo_{NOMBRE[a]}"]] += 1
        if b != MODELOS[0]:
            v[idx[f"modelo_{NOMBRE[b]}"]] -= 1
        t = r.t_test(v)
        est, ee = float(np.squeeze(t.effect)), float(np.squeeze(t.sd))
        p = float(np.squeeze(t.pvalue))
        pares.append({"a": NOMBRE[a], "b": NOMBRE[b], "rr_a_sobre_b": math.exp(est), "ic95": [math.exp(est - 1.96 * ee), math.exp(est + 1.96 * ee)], "p": p})
        ps.append(p)
    for q, pa in zip(pares, holm(ps)):
        q["p_holm"] = pa
    return pares


def wald_interaccion(r, nombres: list[str]) -> dict:
    k = len(r.params)
    filas = [i for i, n in enumerate(nombres) if "×derecho" in n]
    R = np.zeros((len(filas), k))
    for j, i in enumerate(filas):
        R[j, i] = 1.0
    w = r.wald_test(R, scalar=True) if "scalar" in r.wald_test.__code__.co_varnames else r.wald_test(R)
    return {"estadistico": float(np.squeeze(w.statistic)), "gl": len(filas), "p": float(np.squeeze(w.pvalue))}


def regresiones(filas: list[dict]) -> dict:
    out = {}
    grupos_todos = np.array([m["problema"] for m in filas])
    for k, etiqueta in RECUENTOS:
        y = np.array([m[k] for m in filas])
        X, nombres = diseno(filas, interaccion=False, acierto=False)
        r, fam = ajustar_nb(y, X, grupos_todos)
        Xi, nombres_i = diseno(filas, interaccion=True, acierto=False)
        ri, fam_i = ajustar_nb(y, Xi, grupos_todos)
        out[k] = {"etiqueta": etiqueta, "familia": fam, "n": len(y), "terminos": resumen_modelo(r, nombres),
                  "contrastes_modelos": contrastes_modelos(r, nombres),
                  "interaccion_modelo_dominio": {"familia": fam_i, **wald_interaccion(ri, nombres_i)}}
    # medicina con acierto
    med = [m for m in filas if m["dominio"] == "medicina"]
    grupos_med = np.array([m["problema"] for m in med])
    out["medicina_acierto"] = {}
    for k, etiqueta in RECUENTOS[:3] + [RECUENTOS[4]]:
        y = np.array([m[k] for m in med])
        X, nombres = diseno(med, interaccion=False, acierto=True)
        r, fam = ajustar_nb(y, X, grupos_med)
        out["medicina_acierto"][k] = {"etiqueta": etiqueta, "familia": fam, "n": len(y), "terminos": resumen_modelo(r, nombres)}
    # sensibilidad: banda común de nodos
    banda = [m for m in filas if BANDA_NODOS[0] <= m["nodos"] <= BANDA_NODOS[1]]
    grupos_b = np.array([m["problema"] for m in banda])
    out["banda_nodos"] = {"banda": BANDA_NODOS, "n": len(banda),
                          "por_celda": {f"{NOMBRE[mo]}|{do}": sum(1 for m in banda if m["modelo"] == mo and m["dominio"] == do) for mo in MODELOS for do in DOMINIOS}}
    for k, etiqueta in RECUENTOS[:3] + [RECUENTOS[4]]:
        y = np.array([m[k] for m in banda])
        X, nombres = diseno(banda, interaccion=False, acierto=False)
        r, fam = ajustar_nb(y, X, grupos_b)
        out["banda_nodos"][k] = {"etiqueta": etiqueta, "familia": fam, "terminos": resumen_modelo(r, nombres), "contrastes_modelos": contrastes_modelos(r, nombres)}
    return out


# ----------------------------------------------------------------- 2. bootstrap
def cliff_np(a: np.ndarray, b: np.ndarray) -> float:
    d = a[:, None] - b[None, :]
    return float((d > 0).mean() - (d < 0).mean())


def bootstrap(filas: list[dict], rng: np.random.Generator) -> dict:
    out = {"medianas_por_celda": {}, "delta_pares": {}}
    for mo in MODELOS:
        for do in DOMINIOS:
            fs = [m for m in filas if m["modelo"] == mo and m["dominio"] == do]
            celda = {}
            for k in TASAS + ["episodios", "nodos", "palabras"]:
                x = np.array([m[k] for m in fs])
                idx = rng.integers(0, len(x), size=(B_MEDIANAS, len(x)))
                meds = np.median(x[idx], axis=1)
                celda[k] = {"mediana": float(np.median(x)), "ic95": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))]}
            out["medianas_por_celda"][f"{NOMBRE[mo]}|{do}"] = celda
    for do in DOMINIOS:
        problemas = sorted({m["problema"] for m in filas if m["dominio"] == do})
        por = {(m["problema"], m["modelo"]): m for m in filas if m["dominio"] == do}
        out["delta_pares"][do] = {}
        for k in TASAS[:4]:
            res = []
            for a, b in combinations(MODELOS, 2):
                xa = np.array([por[(p, a)][k] for p in problemas]); xb = np.array([por[(p, b)][k] for p in problemas])
                d0 = cliff_np(xa, xb)
                idx = rng.integers(0, len(problemas), size=(B_DELTA, len(problemas)))
                ds = np.array([cliff_np(xa[i], xb[i]) for i in idx])
                res.append({"a": NOMBRE[a], "b": NOMBRE[b], "delta": d0, "ic95": [float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))]})
            out["delta_pares"][do][k] = res
    return out


# ----------------------------------------------------------------- 3. fiabilidad entre corridas
def icc1(M: np.ndarray) -> tuple[float, float]:
    """ICC(1) de una medición e ICC(1,k) de la media de k, en una matriz cadenas × corridas (ANOVA de un factor)."""
    n, k = M.shape
    medias = M.mean(axis=1); gran = M.mean()
    msb = k * ((medias - gran) ** 2).sum() / (n - 1)
    msw = ((M - medias[:, None]) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw), (msb - msw) / msb


def fiabilidad(rng: np.random.Generator) -> dict:
    import estabilidad as E
    ids = sorted(json.load(open(RES / "estabilidad_ids.json")))
    corridas = [E.cargar(RES / f, set(ids)) for f in ("anotacion_200.jsonl.gz", "estabilidad_r1.jsonl.gz", "estabilidad_r2.jsonl.gz")]
    perfiles = [{t: E.perfil(c[t]) for t in ids} for c in corridas]
    out = {"cadenas": len(ids), "corridas": len(corridas), "metricas": {}}
    for k in ("nodos", "episodios", "delib_100n", "ver_100n", "cor_100n", "alt_100n", "cobertura"):
        M = np.array([[p[t][k] for p in perfiles] for t in ids])
        i1, ik = icc1(M)
        idx = rng.integers(0, len(ids), size=(B_ICC, len(ids)))
        boots = np.array([icc1(M[i])[0] for i in idx])
        out["metricas"][k] = {"icc1": float(i1), "icc1_ic95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                              "icc1k": float(ik), "atenuacion_correlacion": float(math.sqrt(max(i1, 0.0)))}
    return out


# ----------------------------------------------------------------- 4. pares emparejados por longitud
def pares_emparejados(filas: list[dict]) -> dict:
    from scipy import stats
    out = {}
    for do in DOMINIOS:
        por = defaultdict(dict)
        for m in filas:
            if m["dominio"] == do:
                por[m["problema"]][m["modelo"]] = m
        out[do] = {}
        for k in TASAS[:4]:
            res = []
            for a, b in combinations(MODELOS, 2):
                pares = [(por[p][a], por[p][b]) for p in por if max(por[p][a]["nodos"], por[p][b]["nodos"]) / min(por[p][a]["nodos"], por[p][b]["nodos"]) <= COCIENTE_MAX]
                xa = np.array([x[k] for x, _ in pares]); xb = np.array([y[k] for _, y in pares])
                d = [x - y for x, y in zip(xa, xb) if x != y]
                p = float(stats.wilcoxon(d).pvalue) if len(d) >= 6 else float("nan")
                res.append({"a": NOMBRE[a], "b": NOMBRE[b], "n_pares": len(pares), "a_mayor": int((xa > xb).sum()), "b_mayor": int((xa < xb).sum()),
                            "mediana_a": float(np.median(xa)) if len(xa) else float("nan"), "mediana_b": float(np.median(xb)) if len(xb) else float("nan"),
                            "delta": cliff_np(xa, xb) if len(xa) else float("nan"), "p_wilcoxon": p})
            out[do][k] = res
    return out


# ----------------------------------------------------------------- 5. corrección global
def correccion_global() -> dict:
    an = json.load(open(RES / "analisis_rq2.json", encoding="utf-8"))
    pruebas = []
    for do, res in an["entre_modelos"].items():
        for k, r in res.items():
            for q in r["pares"]:
                if q["p_wilcoxon"] == q["p_wilcoxon"]:
                    pruebas.append({"dominio": do, "metrica": k, "a": NOMBRE[q["a"]], "b": NOMBRE[q["b"]], "p": q["p_wilcoxon"], "delta": q["delta_cliff"]})
    ps = [t["p"] for t in pruebas]
    for t, q, h in zip(pruebas, bh(ps), holm(ps)):
        t["q_bh"] = q; t["p_holm_global"] = h
    return {"n_pruebas": len(pruebas), "significativas_sin_corregir": sum(t["p"] < 0.05 for t in pruebas),
            "significativas_bh_0.05": sum(t["q_bh"] < 0.05 for t in pruebas), "significativas_holm_global": sum(t["p_holm_global"] < 0.05 for t in pruebas),
            "pruebas": sorted(pruebas, key=lambda t: t["p"])}


# ----------------------------------------------------------------- tablas
def f(x: float, d: int = 2) -> str:
    if x is None or x != x:
        return "—"
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def fp(p: float) -> str:
    if p is None or p != p:
        return "—"
    return "< 0,001" if p < 0.001 else f(p, 3)


def rr(t: dict) -> str:
    return f"{f(t['rr'])} [{f(t['ic95'][0])}–{f(t['ic95'][1])}] ({fp(t['p'])})"


def escribir_tablas(out: dict) -> None:
    L = ["# Inferencia de RQ2 (generada por inferencia.py)", ""]
    L += ["## 1. Regresión de recuentos: episodios ~ log(nodos) + modelo + dominio, errores robustos agrupados por problema", "",
          "Razón de tasas [IC 95 %] (p). Referencias: Qwen y medicina. Elasticidad = exp(β de log(nodos)): cuánto crece el recuento al duplicar los nodos sería 2^β.", "",
          "| Recuento | Familia | β log(nodos) | Gemma / Qwen | Olmo / Qwen | DeepSeek / Qwen | derecho / medicina | Interacción modelo × dominio (Wald, p) |",
          "|---|---|---|---|---|---|---|---|"]
    for k, _ in RECUENTOS:
        r = out["regresiones"][k]; t = r["terminos"]; w = r["interaccion_modelo_dominio"]
        L.append(f"| {r['etiqueta']} | {r['familia']} | {f(t['log_nodos']['beta'])} ({fp(t['log_nodos']['p'])}) | {rr(t['modelo_Gemma'])} | {rr(t['modelo_Olmo'])} | {rr(t['modelo_DeepSeek'])} | {rr(t['derecho'])} | χ²({w['gl']}) = {f(w['estadistico'], 1)}, p = {fp(w['p'])} |")
    L += ["", "Contrastes entre modelos (razón de tasas A / B [IC 95 %], p de Holm sobre los 6 pares):", ""]
    for k, _ in RECUENTOS[:5]:
        r = out["regresiones"][k]
        L += [f"**{r['etiqueta']}**", "", "| A | B | A / B | IC 95 % | p Holm |", "|---|---|---|---|---|"]
        for q in r["contrastes_modelos"]:
            L.append(f"| {q['a']} | {q['b']} | {f(q['rr_a_sobre_b'])} | {f(q['ic95'][0])}–{f(q['ic95'][1])} | {fp(q['p_holm'])} |")
        L.append("")
    L += ["### Medicina con el acierto (episodios ~ log(nodos) + modelo + acierta)", "", "| Recuento | Familia | β log(nodos) | acierta / falla | Gemma / Qwen | Olmo / Qwen | DeepSeek / Qwen |", "|---|---|---|---|---|---|---|"]
    for k, r in out["regresiones"]["medicina_acierto"].items():
        t = r["terminos"]
        L.append(f"| {r['etiqueta']} | {r['familia']} | {f(t['log_nodos']['beta'])} ({fp(t['log_nodos']['p'])}) | {rr(t['acierta'])} | {rr(t['modelo_Gemma'])} | {rr(t['modelo_Olmo'])} | {rr(t['modelo_DeepSeek'])} |")
    L.append("")

    L += ["## 2. Intervalos por bootstrap (remuestreo de problemas; percentil 2,5–97,5)", "", "### Medianas por celda [IC 95 %]", ""]
    for do in DOMINIOS:
        L += [f"**{do.capitalize()}**", "", "| Métrica | " + " | ".join(NOMBRE[m] for m in MODELOS) + " |", "|---|" + "---|" * len(MODELOS)]
        for k in TASAS + ["episodios"]:
            cel = [out["bootstrap"]["medianas_por_celda"][f"{NOMBRE[mo]}|{do}"][k] for mo in MODELOS]
            L.append(f"| {k} | " + " | ".join(f"{f(c['mediana'])} [{f(c['ic95'][0])}–{f(c['ic95'][1])}]" for c in cel) + " |")
        L.append("")
    L += ["### δ de Cliff entre modelos [IC 95 %] (positivo: A mayor)", ""]
    for do in DOMINIOS:
        L += [f"**{do.capitalize()}**", "", "| Métrica | " + " | ".join(f"{q['a']} − {q['b']}" for q in out["bootstrap"]["delta_pares"][do][TASAS[0]]) + " |", "|---|" + "---|" * 6]
        for k in TASAS[:4]:
            L.append(f"| {k} | " + " | ".join(f"{f(q['delta'])} [{f(q['ic95'][0])}–{f(q['ic95'][1])}]" for q in out["bootstrap"]["delta_pares"][do][k]) + " |")
        L.append("")

    fi = out["fiabilidad"]
    L += [f"## 3. Fiabilidad por cadena entre corridas ({fi['cadenas']} cadenas × {fi['corridas']} corridas)", "",
          "ICC(1): fiabilidad de una sola corrida; ICC(1,3): de la media de tres. Atenuación: factor por el que una correlación observada con la métrica de una corrida subestima la verdadera (√ICC).", "",
          "| Métrica | ICC(1) [IC 95 %] | ICC(1,3) | atenuación |", "|---|---|---|---|"]
    for k, v in fi["metricas"].items():
        L.append(f"| {k} | {f(v['icc1'])} [{f(v['icc1_ic95'][0])}–{f(v['icc1_ic95'][1])}] | {f(v['icc1k'])} | {f(v['atenuacion_correlacion'])} |")
    L.append("")

    pe = out["pares_emparejados"]
    L += [f"## 4. Sensibilidad a la longitud", "", f"### Pares del mismo problema con cociente de nodos ≤ {COCIENTE_MAX}", "",
          "Wilcoxon sobre los pares conservados; δ de Cliff sobre esas cadenas.", ""]
    for do in DOMINIOS:
        L += [f"**{do.capitalize()}**", "", "| Métrica | A | B | pares | A > B | B > A | mediana A | mediana B | δ | p |", "|---|---|---|---|---|---|---|---|---|---|"]
        for k in TASAS[:4]:
            for q in pe[do][k]:
                L.append(f"| {k} | {q['a']} | {q['b']} | {q['n_pares']} | {q['a_mayor']} | {q['b_mayor']} | {f(q['mediana_a'])} | {f(q['mediana_b'])} | {f(q['delta'])} | {fp(q['p_wilcoxon'])} |")
        L.append("")
    bn = out["regresiones"]["banda_nodos"]
    L += [f"### Regresión restringida a la banda común de {bn['banda'][0]}–{bn['banda'][1]} nodos (n = {bn['n']}; por celda: " + ", ".join(f"{c} {n}" for c, n in bn["por_celda"].items()) + ")", "",
          "| Recuento | Familia | β log(nodos) | Gemma / Qwen | Olmo / Qwen | DeepSeek / Qwen | derecho / medicina |", "|---|---|---|---|---|---|---|"]
    for k, _ in RECUENTOS[:3] + [RECUENTOS[4]]:
        r = bn[k]; t = r["terminos"]
        L.append(f"| {r['etiqueta']} | {r['familia']} | {f(t['log_nodos']['beta'])} ({fp(t['log_nodos']['p'])}) | {rr(t['modelo_Gemma'])} | {rr(t['modelo_Olmo'])} | {rr(t['modelo_DeepSeek'])} | {rr(t['derecho'])} |")
    L.append("")

    cg = out["correccion_global"]
    L += [f"## 5. Corrección múltiple global sobre las {cg['n_pruebas']} comparaciones pareadas de analizar.py", "",
          f"Significativas sin corregir: {cg['significativas_sin_corregir']}; con Benjamini-Hochberg (q < 0,05): {cg['significativas_bh_0.05']}; con Holm global: {cg['significativas_holm_global']}.", "",
          "Las que sobreviven a Benjamini-Hochberg:", "", "| Dominio | Métrica | A | B | δ | p | q BH |", "|---|---|---|---|---|---|---|"]
    for t in cg["pruebas"]:
        if t["q_bh"] < 0.05:
            L.append(f"| {t['dominio']} | {t['metrica']} | {t['a']} | {t['b']} | {f(t['delta'])} | {fp(t['p'])} | {f(t['q_bh'], 3)} |")
    L.append("")
    (RES / "tablas_inferencia.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEMILLA)
    filas = cargar_filas()
    out = {"n_cadenas": len(filas), "semilla": SEMILLA, "regresiones": regresiones(filas), "bootstrap": bootstrap(filas, rng),
           "fiabilidad": fiabilidad(rng), "pares_emparejados": pares_emparejados(filas), "correccion_global": correccion_global()}
    json.dump(out, open(RES / "inferencia_rq2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    escribir_tablas(out)
    print(f"{len(filas)} cadenas → resultados/inferencia_rq2.json, tablas_inferencia.md")


if __name__ == "__main__":
    main()
