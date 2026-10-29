"""Análisis de RQ2 sobre las 200 anotaciones del corpus principal. Sin API.

Uso (Python 3 con scipy y matplotlib; no necesita la .venv del pipeline):
    python3 analizar.py

Entradas: resultados/anotacion_200.jsonl.gz (salidas 00_sgotgraph_objetivo3), cadenas_aceptadas.jsonl (metadatos por cadena)
y cadenas/*.jsonl.gz (todo lo generado, para la distribución de longitudes).
Salidas en resultados/: metricas_por_cadena.csv, analisis_rq2.json, tablas_rq2.md y figuras/*.png.

Qué calcula:
- Métricas por cadena: longitud (palabras, nodos), episodios por tipo y por 100 nodos, desenlaces, relaciones funcionales,
  cobertura, organización (perfil del pipeline), posición relativa de la deliberación, acuerdo del consenso.
- Por celda modelo × dominio: mediana e IQR; proporciones de desenlaces agregadas.
- Comparaciones entre modelos dentro de cada dominio, pareadas por problema (n = 25): Friedman, Wilcoxon por pares con Holm
  y delta de Cliff. Entre dominios dentro de cada modelo: Mann-Whitney y delta de Cliff.
- Medicina: acierta frente a falla, agregado y dentro de cada problema (solo los problemas con modelos de los dos lados).
- Longitud de las 1 038 cadenas generadas, por modelo y dominio.
"""
from __future__ import annotations

import csv
import gzip
import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RES = AQUI / "resultados"
FIG = RES / "figuras"
MODELOS = ["qwen3.8-27b", "gemma-4-31b-it", "olmo-3.1-32b-think", "deepseek-v4-pro"]
NOMBRE = {"qwen3.8-27b": "Qwen", "gemma-4-31b-it": "Gemma", "olmo-3.1-32b-think": "Olmo", "deepseek-v4-pro": "DeepSeek"}
DOMINIOS = ["medicina", "derecho"]
DELIBERATIVOS = ("VERIFICATION", "CORRECTION", "ALTERNATIVE")
TIPOS = ("PROGRESSION", "SUPPORT", "VERIFICATION", "CORRECTION", "ALTERNATIVE")
RELACIONES = ("EVALUATES", "REPLACES", "COMPETES_WITH", "SUPPORTS_EPISODE", "EMBEDS", "CONTINUES", "RESUMES", "INTERRUPTS")
VENTANA_CORRECCION = 500  # caracteres tras una verificación refutada en los que se busca una corrección

# Métricas que entran en las comparaciones estadísticas (clave, etiqueta)
METRICAS_CLAVE = [
    ("palabras", "palabras de razonamiento"),
    ("nodos", "nodos del grafo"),
    ("episodios", "episodios"),
    ("delib_100n", "deliberativos por 100 nodos"),
    ("ver_100n", "VERIFICATION por 100 nodos"),
    ("cor_100n", "CORRECTION por 100 nodos"),
    ("alt_100n", "ALTERNATIVE por 100 nodos"),
    ("sup_100n", "SUPPORT por 100 nodos"),
    ("prop_delib", "fracción deliberativa de los episodios"),
    ("cobertura", "cobertura del texto por episodios"),
    ("cobertura_delib", "cobertura deliberativa"),
    ("componentes_100n", "componentes funcionales por 100 nodos"),
    ("acuerdo_episodios", "acuerdo medio del consenso en episodios"),
]


def envolvente(ep: dict) -> tuple[int, int]:
    ms = ep["members"]
    return min(m["start_char"] for m in ms), max(m["end_char"] for m in ms)


def metricas_cadena(g: dict, meta: dict) -> dict:
    nodos = g["sgot_graph"]["nodes"]
    aristas = g["sgot_graph"]["edges"]
    eps = g["episodes"]
    rels = g["episode_relations"]
    p = g["profile"]
    n_nodos = len(nodos)
    largo = max(1, p["trace_length_chars"])
    por_tipo = Counter(e["episode_type"] for e in eps)
    desenlaces = Counter((e["episode_type"], e["outcome"]) for e in eps)
    rel_tipo = Counter(r["relation_type"] for r in rels)
    tipo_nodo = Counter(n["node_type"] for n in nodos)
    por_id = {e["episode_id"]: e for e in eps}

    # Verificación refutada seguida de corrección: misma diana (VER EVALUATES X y COR REPLACES X) o corrección que
    # empieza dentro de la verificación o en los VENTANA_CORRECCION caracteres siguientes.
    diana_eval = defaultdict(set)
    diana_repl = defaultdict(set)
    for r in rels:
        if r["relation_type"] == "EVALUATES":
            diana_eval[r["source_episode_id"]].add(r["target_episode_id"])
        elif r["relation_type"] == "REPLACES":
            diana_repl[r["source_episode_id"]].add(r["target_episode_id"])
    dianas_corregidas = set().union(*diana_repl.values()) if diana_repl else set()
    inicios_cor = [envolvente(e)[0] for e in eps if e["episode_type"] == "CORRECTION"]
    refutadas = [e for e in eps if e["episode_type"] == "VERIFICATION" and e["outcome"] == "REFUTED"]
    ref_misma_diana = sum(1 for e in refutadas if diana_eval[e["episode_id"]] & dianas_corregidas)
    ref_cor_cercana = 0
    for e in refutadas:
        a, b = envolvente(e)
        if any(a <= s <= b + VENTANA_CORRECCION for s in inicios_cor):
            ref_cor_cercana += 1

    # Posición relativa (centro de la envolvente / longitud) de los episodios deliberativos
    pos = [sum(envolvente(e)) / 2 / largo for e in eps if e["episode_type"] in DELIBERATIVOS]
    delib = sum(por_tipo[t] for t in DELIBERATIVOS)
    acuerdo = [e["consensus"]["agreement"] for e in eps if "consensus" in e]
    cob = p["coverage"]["by_type"]
    org = p["organization"]
    f100 = lambda x: 100.0 * x / n_nodos
    m = {
        "id": g["trace_id"],
        "problema": meta["problem_id"],
        "dominio": meta["domain"],
        "modelo": g["trace_id"].split("--")[1],
        "estrato": meta.get("estrato", ""),
        "subarea": meta.get("subarea", ""),
        "correcto": str(meta.get("correcto")) == "True",
        "palabras": int(meta["palabras_razonamiento"]),
        "caracteres": largo,
        "nodos": n_nodos,
        "nodos_100pal": 100.0 * n_nodos / int(meta["palabras_razonamiento"]),
        "aristas_semanticas": sum(1 for a in aristas if a["edge_type"] != "NEXT"),
        "nodos_ambiguos_frac": sum(1 for n in nodos if n.get("review_status") == "ambiguous") / n_nodos,
        "frac_nodos_CHECK": tipo_nodo["CHECK"] / n_nodos,
        "frac_nodos_ALT": tipo_nodo["ALT"] / n_nodos,
        "frac_nodos_REVISE": tipo_nodo["REVISE"] / n_nodos,
        "episodios": len(eps),
        "deliberativos": delib,
        "delib_100n": f100(delib),
        "prop_delib": delib / len(eps) if eps else 0.0,
        "ver_100n": f100(por_tipo["VERIFICATION"]),
        "cor_100n": f100(por_tipo["CORRECTION"]),
        "alt_100n": f100(por_tipo["ALTERNATIVE"]),
        "sup_100n": f100(por_tipo["SUPPORT"]),
        "pro_100n": f100(por_tipo["PROGRESSION"]),
        "delib_1000pal": 1000.0 * delib / int(meta["palabras_razonamiento"]),
        "cobertura": p["coverage"]["overall"]["ratio"],
        "cobertura_delib": min(1.0, sum(cob[t]["ratio"] for t in DELIBERATIVOS)),
        "pos_delib": statistics.fmean(pos) if pos else float("nan"),
        "delib_ultimo_tercio": sum(1 for x in pos if x >= 2 / 3) / len(pos) if pos else float("nan"),
        "lineas_progresion": org["progression_line_count"],
        "componentes": org["functional_component_count"],
        "componentes_100n": f100(org["functional_component_count"]),
        "pares_anidados": org["nested_episode_pair_count"],
        "pares_solapados": org["overlapping_episode_pair_count"],
        "no_contiguos": org["noncontiguous_episode_count"],
        "abiertos": org["unresolved_or_inconclusive_episode_count"],
        "diversidad_funcional": org["functional_diversity"],
        "ruta_episodios_media": p["route_structure"]["mean_episode_count"],
        "acuerdo_episodios": statistics.fmean(acuerdo) if acuerdo else float("nan"),
        "episodios_unanimes_frac": sum(1 for e in eps if e["consensus"].get("unanimous")) / len(eps) if eps else float("nan"),
        "ver_refutadas": len(refutadas),
        "ref_con_correccion_misma_diana": ref_misma_diana,
        "ref_con_correccion_cercana": ref_cor_cercana,
    }
    for t in TIPOS:
        m[f"n_{t}"] = por_tipo[t]
    for (t, o), c in sorted(desenlaces.items()):
        m[f"d_{t}_{o}"] = c
    for r in RELACIONES:
        m[f"r_{r}"] = rel_tipo[r]
    return m


# ----------------------------------------------------------------- estadística (scipy si está; si no, solo descriptivos)
try:
    from scipy import stats as _st
except ImportError:  # pragma: no cover
    _st = None


def mediana_iqr(xs: list[float]) -> tuple[float, float, float]:
    xs = sorted(x for x in xs if x == x)
    if not xs:
        return float("nan"), float("nan"), float("nan")
    q = statistics.quantiles(xs, n=4, method="inclusive") if len(xs) > 1 else [xs[0]] * 3
    return statistics.median(xs), q[0], q[2]


def cliff(a: list[float], b: list[float]) -> float:
    """Delta de Cliff: P(a > b) − P(a < b), entre −1 y 1."""
    mas = sum(1 for x in a for y in b if x > y)
    menos = sum(1 for x in a for y in b if x < y)
    return (mas - menos) / (len(a) * len(b)) if a and b else float("nan")


def holm(ps: list[float]) -> list[float]:
    orden = sorted(range(len(ps)), key=lambda i: ps[i])
    ajust = [0.0] * len(ps)
    corriente = 0.0
    for k, i in enumerate(orden):
        corriente = max(corriente, min(1.0, ps[i] * (len(ps) - k)))
        ajust[i] = corriente
    return ajust


def wilcoxon(a: list[float], b: list[float]) -> float:
    d = [x - y for x, y in zip(a, b) if x - y != 0]
    if _st is None or len(d) < 6:
        return float("nan")
    return float(_st.wilcoxon(d, zero_method="wilcox", alternative="two-sided").pvalue)


def mannwhitney(a: list[float], b: list[float]) -> float:
    if _st is None or len(a) < 3 or len(b) < 3:
        return float("nan")
    return float(_st.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def friedman(grupos: list[list[float]]) -> float:
    if _st is None:
        return float("nan")
    return float(_st.friedmanchisquare(*grupos).pvalue)


def ols(fs: list[dict], k: str, con_acierto: bool) -> dict:
    """Mínimos cuadrados ordinarios de la métrica k sobre log(nodos), modelo y dominio (y acierto). Coeficientes con p (t de Student)."""
    import math
    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return {}
    nombres = ["const", "log_nodos"] + [f"modelo_{NOMBRE[mo]}" for mo in MODELOS[1:]] + ["dominio_derecho"] + (["acierta"] if con_acierto else [])
    X, y = [], []
    for m in fs:
        if k == "palabras":
            fila = [1.0, 0.0]  # palabras no se ajusta por nodos (son casi lo mismo)
        else:
            fila = [1.0, math.log(m["nodos"])]
        fila += [1.0 if m["modelo"] == mo else 0.0 for mo in MODELOS[1:]] + [1.0 if m["dominio"] == "derecho" else 0.0]
        if con_acierto:
            fila.append(1.0 if m["correcto"] else 0.0)
        X.append(fila)
        y.append(math.log(m[k]) if k == "palabras" else m[k])
    X, y = np.array(X), np.array(y)
    # fuera las columnas constantes (log_nodos en palabras; dominio cuando solo hay uno)
    utiles = [j for j in range(X.shape[1]) if j == 0 or X[:, j].std() > 0]
    X, nombres = X[:, utiles], [nombres[j] for j in utiles]
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    gl = len(y) - X.shape[1]
    s2 = float(resid @ resid) / gl
    cov = s2 * np.linalg.inv(X.T @ X)
    ee = np.sqrt(np.diag(cov))
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    res = {"n": len(y), "r2": r2, "coeficientes": {}}
    for nombre, b, e in zip(nombres, beta, ee):
        t = b / e if e > 0 else float("nan")
        p = float(2 * _st.t.sf(abs(t), gl)) if _st is not None and t == t else float("nan")
        res["coeficientes"][nombre] = {"beta": float(b), "ee": float(e), "p": p}
    return res


def spearman(a: list[float], b: list[float]) -> tuple[float, float]:
    if _st is None:
        return float("nan"), float("nan")
    r = _st.spearmanr(a, b)
    return float(r.statistic), float(r.pvalue)


# ----------------------------------------------------------------- análisis
def analizar() -> dict:
    meta = {json.loads(l)["id"]: json.loads(l) for l in open(AQUI / "cadenas_aceptadas.jsonl", encoding="utf-8") if l.strip()}
    filas = []
    for l in gzip.open(RES / "anotacion_200.jsonl.gz", "rt", encoding="utf-8"):
        g = json.loads(l)
        filas.append(metricas_cadena(g, meta[g["trace_id"]]))
    filas.sort(key=lambda m: (m["dominio"], m["problema"], MODELOS.index(m["modelo"])))
    columnas = sorted({k for m in filas for k in m}, key=lambda k: (k not in filas[0], k))
    with open(RES / "metricas_por_cadena.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas, restval=0)
        w.writeheader()
        w.writerows(filas)

    celda = lambda mo, do: [m for m in filas if m["modelo"] == mo and m["dominio"] == do]
    out: dict = {"n_cadenas": len(filas), "celdas": {}, "desenlaces": {}, "relaciones": {}, "entre_modelos": {},
                 "entre_dominios": {}, "medicina_acierto": {}, "longitud_generadas": {}, "correlaciones": {}}

    # 1. Descriptivos por celda
    descriptivas = [k for k, _ in METRICAS_CLAVE] + ["pos_delib", "nodos_100pal", "delib_1000pal", "delib_ultimo_tercio", "pro_100n",
                                                     "lineas_progresion", "pares_anidados", "abiertos", "diversidad_funcional",
                                                     "nodos_ambiguos_frac", "episodios_unanimes_frac", "ver_refutadas"]
    for mo in MODELOS:
        for do in DOMINIOS:
            fs = celda(mo, do)
            c = {"n": len(fs), "aciertos": sum(m["correcto"] for m in fs)}
            for k in descriptivas:
                med, q1, q3 = mediana_iqr([m[k] for m in fs])
                c[k] = {"mediana": med, "q1": q1, "q3": q3, "media": statistics.fmean([m[k] for m in fs if m[k] == m[k]])}
            # episodios totales de la celda, por tipo
            c["episodios_totales"] = {t: sum(m[f"n_{t}"] for m in fs) for t in TIPOS}
            out["celdas"][f"{mo}|{do}"] = c

    # 2. Desenlaces y relaciones agregados por modelo (y por modelo × dominio)
    def agreg(fs: list[dict]) -> dict:
        d = defaultdict(Counter)
        for m in fs:
            for k, v in m.items():
                if k.startswith("d_"):
                    _, t, o = k.split("_", 2)
                    d[t][o] += v
        rel = Counter({r: sum(m[f"r_{r}"] for m in fs) for r in RELACIONES})
        ref = sum(m["ver_refutadas"] for m in fs)
        return {"desenlaces": {t: dict(d[t]) for t in TIPOS}, "relaciones": dict(rel),
                "ver_refutadas": ref,
                "ref_con_correccion_misma_diana": sum(m["ref_con_correccion_misma_diana"] for m in fs),
                "ref_con_correccion_cercana": sum(m["ref_con_correccion_cercana"] for m in fs),
                "episodios": {t: sum(m[f"n_{t}"] for m in fs) for t in TIPOS}}
    for mo in MODELOS:
        out["desenlaces"][mo] = agreg([m for m in filas if m["modelo"] == mo])
        for do in DOMINIOS:
            out["desenlaces"][f"{mo}|{do}"] = agreg(celda(mo, do))
    out["desenlaces"]["todas"] = agreg(filas)

    # 3. Entre modelos, dentro de cada dominio, pareado por problema
    for do in DOMINIOS:
        problemas = sorted({m["problema"] for m in filas if m["dominio"] == do})
        por = {(m["problema"], m["modelo"]): m for m in filas if m["dominio"] == do}
        res = {}
        for k, _ in METRICAS_CLAVE:
            grupos = {mo: [por[(p, mo)][k] for p in problemas] for mo in MODELOS}
            pares = []
            for a, b in combinations(MODELOS, 2):
                pares.append({"a": a, "b": b, "mediana_a": statistics.median(grupos[a]), "mediana_b": statistics.median(grupos[b]),
                              "p_wilcoxon": wilcoxon(grupos[a], grupos[b]), "delta_cliff": cliff(grupos[a], grupos[b]),
                              "a_mayor_en": sum(1 for x, y in zip(grupos[a], grupos[b]) if x > y),
                              "b_mayor_en": sum(1 for x, y in zip(grupos[a], grupos[b]) if x < y)})
            ps = [q["p_wilcoxon"] for q in pares]
            if all(p == p for p in ps):
                for q, pa in zip(pares, holm(ps)):
                    q["p_holm"] = pa
            res[k] = {"p_friedman": friedman([grupos[mo] for mo in MODELOS]), "pares": pares}
        out["entre_modelos"][do] = res

    # 4. Entre dominios, dentro de cada modelo (problemas distintos: no pareado)
    for mo in MODELOS:
        res = {}
        for k, _ in METRICAS_CLAVE:
            a = [m[k] for m in celda(mo, "medicina")]
            b = [m[k] for m in celda(mo, "derecho")]
            res[k] = {"mediana_med": statistics.median(a), "mediana_der": statistics.median(b),
                      "p_mannwhitney": mannwhitney(a, b), "delta_cliff_med_vs_der": cliff(a, b)}
        ks = list(res)
        for k, pa in zip(ks, holm([res[k]["p_mannwhitney"] for k in ks])):
            res[k]["p_holm"] = pa
        out["entre_dominios"][mo] = res

    # 5. Medicina: acierta frente a falla. Agregado (Mann-Whitney) y dentro de problema (problemas mixtos).
    med = [m for m in filas if m["dominio"] == "medicina"]
    res = {"n_aciertos": sum(m["correcto"] for m in med), "n_fallos": sum(not m["correcto"] for m in med), "agregado": {},
           "dentro_de_problema": {}, "por_modelo": {}}
    por_problema = defaultdict(list)
    for m in med:
        por_problema[m["problema"]].append(m)
    mixtos = {p: fs for p, fs in por_problema.items() if 0 < sum(f["correcto"] for f in fs) < len(fs)}
    res["problemas_mixtos"] = len(mixtos)
    for k, _ in METRICAS_CLAVE:
        a = [m[k] for m in med if m["correcto"]]
        b = [m[k] for m in med if not m["correcto"]]
        res["agregado"][k] = {"mediana_acierta": statistics.median(a), "mediana_falla": statistics.median(b),
                              "p_mannwhitney": mannwhitney(a, b), "delta_cliff": cliff(a, b)}
        # dentro de problema: media de aciertos menos media de fallos, por problema mixto; signo y Wilcoxon sobre esas diferencias
        difs = []
        for fs in mixtos.values():
            ac = [f[k] for f in fs if f["correcto"]]
            fa = [f[k] for f in fs if not f["correcto"]]
            difs.append(statistics.fmean(ac) - statistics.fmean(fa))
        res["dentro_de_problema"][k] = {"n_problemas": len(difs), "mediana_dif_acierta_menos_falla": statistics.median(difs),
                                        "positivas": sum(d > 0 for d in difs), "negativas": sum(d < 0 for d in difs),
                                        "p_wilcoxon": wilcoxon(difs, [0.0] * len(difs))}
    for mo in MODELOS:
        fs = celda(mo, "medicina")
        ac, fa = [m for m in fs if m["correcto"]], [m for m in fs if not m["correcto"]]
        res["por_modelo"][mo] = {"aciertos": len(ac)}
        for k in ("delib_100n", "ver_100n", "palabras"):
            res["por_modelo"][mo][f"{k}_acierta"] = mediana_iqr([m[k] for m in ac])[0]
            res["por_modelo"][mo][f"{k}_falla"] = mediana_iqr([m[k] for m in fa])[0]
            res["por_modelo"][mo][f"{k}_delta"] = cliff([m[k] for m in ac], [m[k] for m in fa])
    # por tercil de longitud: ¿la diferencia en verificación sobrevive a igual longitud?
    orden = sorted(med, key=lambda m: m["palabras"])
    res["por_tercil_palabras"] = []
    for i in range(3):
        t = orden[i * len(orden) // 3:(i + 1) * len(orden) // 3]
        a, b = [m["ver_100n"] for m in t if m["correcto"]], [m["ver_100n"] for m in t if not m["correcto"]]
        res["por_tercil_palabras"].append({"palabras_min": t[0]["palabras"], "palabras_max": t[-1]["palabras"], "aciertos": len(a), "fallos": len(b),
                                           "ver_100n_acierta": statistics.median(a) if a else float("nan"),
                                           "ver_100n_falla": statistics.median(b) if b else float("nan"), "delta_cliff": cliff(a, b)})
    out["medicina_acierto"] = res

    # 5b. Ajuste por longitud: mínimos cuadrados de cada tasa sobre log(nodos), modelo (referencia Qwen) y dominio (referencia medicina);
    # en medicina, además el acierto. Da el efecto de modelo a igual tamaño de grafo.
    out["ajuste_longitud"] = {}
    for k in ("delib_100n", "ver_100n", "cor_100n", "alt_100n", "sup_100n", "cobertura", "acuerdo_episodios", "episodios"):
        out["ajuste_longitud"][k] = ols(filas, k, con_acierto=False)
    out["ajuste_longitud_medicina"] = {k: ols(med, k, con_acierto=True) for k in ("ver_100n", "delib_100n", "cor_100n", "palabras")}

    # 6. Longitud de todo lo generado
    gen = []
    for f in sorted((AQUI / "cadenas").glob("*.jsonl.gz")):
        for l in gzip.open(f, "rt", encoding="utf-8"):
            d = json.loads(l)
            gen.append({"modelo": d["id"].split("--")[1], "dominio": d["domain"], "palabras": int(d["palabras_razonamiento"]),
                        "termina": d.get("generation_finish_reason") == "stop" and bool(d.get("respuesta_elegida")),
                        "correcto": str(d.get("correcto")) == "True"})
    out["longitud_generadas"]["n"] = len(gen)
    for mo in MODELOS:
        for do in ["todas"] + DOMINIOS:
            xs = [x for x in gen if x["modelo"] == mo and (do == "todas" or x["dominio"] == do)]
            if not xs:
                continue
            pal = sorted(x["palabras"] for x in xs)
            med, q1, q3 = mediana_iqr(pal)
            out["longitud_generadas"][f"{mo}|{do}"] = {
                "n": len(xs), "mediana": med, "q1": q1, "q3": q3, "p90": pal[int(0.9 * (len(pal) - 1))], "max": pal[-1],
                "frac_mas_4000": sum(1 for x in pal if x > 4000) / len(pal), "frac_mas_3000": sum(1 for x in pal if x > 3000) / len(pal),
                "frac_sin_letra": sum(1 for x in xs if not x["termina"]) / len(xs),
                "acierto": sum(x["correcto"] for x in xs) / len(xs)}

    # 7. Correlaciones: ¿la tasa por 100 nodos elimina la dependencia de la longitud?
    out["correlaciones"]["nodos_vs_palabras"] = spearman([m["palabras"] for m in filas], [m["nodos"] for m in filas])
    out["correlaciones"]["deliberativos_vs_nodos"] = spearman([m["nodos"] for m in filas], [m["deliberativos"] for m in filas])
    out["correlaciones"]["delib_100n_vs_nodos"] = spearman([m["nodos"] for m in filas], [m["delib_100n"] for m in filas])
    out["correlaciones"]["delib_100n_vs_palabras"] = spearman([m["palabras"] for m in filas], [m["delib_100n"] for m in filas])
    out["correlaciones"]["cobertura_vs_nodos"] = spearman([m["nodos"] for m in filas], [m["cobertura"] for m in filas])
    out["correlaciones"]["acuerdo_vs_nodos"] = spearman([m["nodos"] for m in filas], [m["acuerdo_episodios"] for m in filas])

    json.dump(out, open(RES / "analisis_rq2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    escribir_tablas(out, filas, gen)
    figuras(filas, gen)
    return out


# ----------------------------------------------------------------- tablas en Markdown
def f(x: float, d: int = 2) -> str:
    if x != x:
        return "—"
    s = f"{x:,.{d}f}".replace(",", " ").replace(".", ",")
    return s


def fp(p: float) -> str:
    if p != p:
        return "—"
    return "< 0,001" if p < 0.001 else f(p, 3)


def escribir_tablas(out: dict, filas: list[dict], gen: list[dict]) -> None:
    L = ["# Tablas de RQ2 (generadas por analizar.py)", ""]
    L += ["## 1. Descriptivos por modelo y dominio (mediana [Q1–Q3])", ""]
    claves = [("palabras", "Palabras", 0), ("nodos", "Nodos", 0), ("episodios", "Episodios", 0), ("delib_100n", "Deliberativos / 100 nodos", 1),
              ("ver_100n", "VER / 100 n", 1), ("cor_100n", "COR / 100 n", 1), ("alt_100n", "ALT / 100 n", 1), ("sup_100n", "SUP / 100 n", 1),
              ("prop_delib", "Fracción deliberativa", 2), ("cobertura", "Cobertura", 2), ("cobertura_delib", "Cobertura deliberativa", 2),
              ("pos_delib", "Posición media de la deliberación", 2), ("delib_ultimo_tercio", "Deliberación en el último tercio", 2),
              ("componentes_100n", "Componentes / 100 n", 1), ("pares_anidados", "Pares anidados", 0), ("abiertos", "Episodios abiertos", 0),
              ("acuerdo_episodios", "Acuerdo del consenso", 2), ("nodos_ambiguos_frac", "Nodos ambiguos", 2)]
    for do in DOMINIOS:
        L += [f"### {do.capitalize()}", "", "| Métrica | " + " | ".join(NOMBRE[m] for m in MODELOS) + " |", "|---|" + "---|" * len(MODELOS)]
        for k, et, d in claves:
            cel = []
            for mo in MODELOS:
                c = out["celdas"][f"{mo}|{do}"][k]
                cel.append(f"{f(c['mediana'], d)} [{f(c['q1'], d)}–{f(c['q3'], d)}]")
            L.append(f"| {et} | " + " | ".join(cel) + " |")
        L.append("")

    L += ["## 2. Episodios totales por tipo y celda", "", "| Celda | PROGRESSION | SUPPORT | VERIFICATION | CORRECTION | ALTERNATIVE | Total |", "|---|---|---|---|---|---|---|"]
    for mo in MODELOS:
        for do in DOMINIOS:
            e = out["celdas"][f"{mo}|{do}"]["episodios_totales"]
            L.append(f"| {NOMBRE[mo]} · {do} | " + " | ".join(str(e[t]) for t in TIPOS) + f" | {sum(e.values())} |")
    L.append("")

    L += ["## 3. Desenlaces de los episodios deliberativos por modelo (recuentos agregados de las 50 cadenas)", "",
          "| Modelo | VER confirmada / inconclusa / refutada | COR recuperada / parcial / fallida / abierta | ALT rechazada / adoptada / fusionada / abierta | Refutadas con corrección (misma diana / cercana) |",
          "|---|---|---|---|---|"]
    for mo in MODELOS + ["todas"]:
        d = out["desenlaces"][mo]
        v, c, a = d["desenlaces"]["VERIFICATION"], d["desenlaces"]["CORRECTION"], d["desenlaces"]["ALTERNATIVE"]
        L.append(f"| {NOMBRE.get(mo, 'Todas')} | {v.get('CONFIRMED', 0)} / {v.get('INCONCLUSIVE', 0)} / {v.get('REFUTED', 0)} "
                 f"| {c.get('RECOVERED', 0)} / {c.get('PARTIAL', 0)} / {c.get('FAILED', 0)} / {c.get('UNRESOLVED', 0)} "
                 f"| {a.get('REJECTED', 0)} / {a.get('ADOPTED', 0)} / {a.get('MERGED', 0)} / {a.get('UNRESOLVED', 0)} "
                 f"| {d['ref_con_correccion_misma_diana']} / {d['ref_con_correccion_cercana']} de {d['ver_refutadas']} |")
    L.append("")

    L += ["## 4. Relaciones funcionales por modelo (recuentos agregados)", "", "| Modelo | EVALUATES | REPLACES | COMPETES_WITH | SUPPORTS_EPISODE | EMBEDS | CONTINUES |", "|---|---|---|---|---|---|---|"]
    for mo in MODELOS + ["todas"]:
        r = out["desenlaces"][mo]["relaciones"]
        L.append(f"| {NOMBRE.get(mo, 'Todas')} | " + " | ".join(str(r.get(k, 0)) for k in ("EVALUATES", "REPLACES", "COMPETES_WITH", "SUPPORTS_EPISODE", "EMBEDS", "CONTINUES")) + " |")
    L.append("")

    L += ["## 5. Comparaciones entre modelos, pareadas por problema (n = 25 por dominio)", "",
          "Wilcoxon de rangos con signo por pares, p ajustada por Holm sobre los 6 pares de cada métrica; delta de Cliff (positivo: A mayor). "
          "«A > B» cuenta en cuántos de los 25 problemas A supera a B.", ""]
    for do in DOMINIOS:
        L += [f"### {do.capitalize()}", ""]
        for k, et in METRICAS_CLAVE:
            r = out["entre_modelos"][do][k]
            L += [f"**{et}** — Friedman p = {fp(r['p_friedman'])}", "", "| A | B | mediana A | mediana B | A > B | B > A | δ de Cliff | p Holm |", "|---|---|---|---|---|---|---|---|"]
            for q in r["pares"]:
                L.append(f"| {NOMBRE[q['a']]} | {NOMBRE[q['b']]} | {f(q['mediana_a'])} | {f(q['mediana_b'])} | {q['a_mayor_en']} | {q['b_mayor_en']} | {f(q['delta_cliff'])} | {fp(q.get('p_holm', float('nan')))} |")
            L.append("")

    L += ["## 6. Medicina frente a derecho, dentro de cada modelo (Mann-Whitney, Holm sobre las métricas; δ positivo: mayor en medicina)", ""]
    for mo in MODELOS:
        L += [f"### {NOMBRE[mo]}", "", "| Métrica | mediana med | mediana der | δ de Cliff | p Holm |", "|---|---|---|---|---|"]
        for k, et in METRICAS_CLAVE:
            r = out["entre_dominios"][mo][k]
            L.append(f"| {et} | {f(r['mediana_med'])} | {f(r['mediana_der'])} | {f(r['delta_cliff_med_vs_der'])} | {fp(r['p_holm'])} |")
        L.append("")

    m = out["medicina_acierto"]
    L += ["## 7. Medicina: acierta frente a falla", "",
          f"{m['n_aciertos']} cadenas acertadas y {m['n_fallos']} falladas (100 cadenas). Agregado: Mann-Whitney sin ajustar. "
          f"Dentro de problema: en los {m['problemas_mixtos']} problemas con modelos de los dos lados, media de los que aciertan menos media de los que fallan.", "",
          "| Métrica | mediana acierta | mediana falla | δ de Cliff | p MW | problemas mixtos: dif. mediana | positivas / negativas | p Wilcoxon |", "|---|---|---|---|---|---|---|---|"]
    for k, et in METRICAS_CLAVE:
        a, d = m["agregado"][k], m["dentro_de_problema"][k]
        L.append(f"| {et} | {f(a['mediana_acierta'])} | {f(a['mediana_falla'])} | {f(a['delta_cliff'])} | {fp(a['p_mannwhitney'])} | {f(d['mediana_dif_acierta_menos_falla'])} | {d['positivas']} / {d['negativas']} | {fp(d['p_wilcoxon'])} |")
    L += ["", "Por modelo (medianas acierta / falla y δ de Cliff, positivo: mayor en las acertadas):", "",
          "| Modelo | aciertos (de 25) | VER/100 n | δ | delib/100 n | δ | palabras | δ |", "|---|---|---|---|---|---|---|---|"]
    for mo in MODELOS:
        r = m["por_modelo"][mo]
        L.append(f"| {NOMBRE[mo]} | {r['aciertos']} | {f(r['ver_100n_acierta'], 1)} / {f(r['ver_100n_falla'], 1)} | {f(r['ver_100n_delta'])} "
                 f"| {f(r['delib_100n_acierta'], 1)} / {f(r['delib_100n_falla'], 1)} | {f(r['delib_100n_delta'])} "
                 f"| {f(r['palabras_acierta'], 0)} / {f(r['palabras_falla'], 0)} | {f(r['palabras_delta'])} |")
    L += ["", "VERIFICATION por 100 nodos, acierta frente a falla, por tercil de longitud (medicina):", "",
          "| Tercil de palabras | aciertos / fallos | VER/100 n acierta | VER/100 n falla | δ de Cliff |", "|---|---|---|---|---|"]
    for t in m["por_tercil_palabras"]:
        L.append(f"| {f(t['palabras_min'], 0)}–{f(t['palabras_max'], 0)} | {t['aciertos']} / {t['fallos']} | {f(t['ver_100n_acierta'])} | {f(t['ver_100n_falla'])} | {f(t['delta_cliff'])} |")
    L.append("")

    L += ["## 7b. Ajuste por longitud (mínimos cuadrados: métrica ~ log(nodos) + modelo + dominio; referencia Qwen · medicina)", "",
          "| Métrica | R² | log(nodos) | Gemma | Olmo | DeepSeek | derecho |", "|---|---|---|---|---|---|---|"]
    def coef(c: dict) -> str:
        return f"{f(c['beta'])} ({fp(c['p'])})"
    for k, r in out["ajuste_longitud"].items():
        c = r["coeficientes"]
        L.append(f"| {k} | {f(r['r2'])} | {coef(c['log_nodos'])} | {coef(c['modelo_Gemma'])} | {coef(c['modelo_Olmo'])} | {coef(c['modelo_DeepSeek'])} | {coef(c['dominio_derecho'])} |")
    L += ["", "Medicina, con el acierto como variable (métrica ~ log(nodos) + modelo + acierta; palabras en logaritmo y sin nodos):", "",
          "| Métrica | R² | log(nodos) | Gemma | Olmo | DeepSeek | acierta |", "|---|---|---|---|---|---|---|"]
    for k, r in out["ajuste_longitud_medicina"].items():
        c = r["coeficientes"]
        L.append(f"| {k} | {f(r['r2'])} | {coef(c['log_nodos']) if 'log_nodos' in c else '—'} | {coef(c['modelo_Gemma'])} | {coef(c['modelo_Olmo'])} | {coef(c['modelo_DeepSeek'])} | {coef(c['acierta'])} |")
    L.append("")

    L += [f"## 8. Longitud de todo lo generado ({out['longitud_generadas']['n']} cadenas)", "",
          "| Modelo · dominio | n | mediana [Q1–Q3] | p90 | máx | > 3 000 | > 4 000 | sin letra final | acierto |", "|---|---|---|---|---|---|---|---|---|"]
    for mo in MODELOS:
        for do in ["todas"] + DOMINIOS:
            r = out["longitud_generadas"].get(f"{mo}|{do}")
            if r:
                L.append(f"| {NOMBRE[mo]} · {do} | {r['n']} | {f(r['mediana'], 0)} [{f(r['q1'], 0)}–{f(r['q3'], 0)}] | {f(r['p90'], 0)} | {f(r['max'], 0)} | {f(100 * r['frac_mas_3000'], 0)} % | {f(100 * r['frac_mas_4000'], 0)} % | {f(100 * r['frac_sin_letra'], 0)} % | {f(100 * r['acierto'], 0)} % |")
    L.append("")

    L += ["## 9. Correlaciones de Spearman (200 cadenas)", "", "| Par | ρ | p |", "|---|---|---|"]
    for k, (rho, p) in out["correlaciones"].items():
        L.append(f"| {k} | {f(rho)} | {fp(p)} |")
    L.append("")
    (RES / "tablas_rq2.md").write_text("\n".join(L), encoding="utf-8")


# ----------------------------------------------------------------- figuras
def figuras(filas: list[dict], gen: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    FIG.mkdir(exist_ok=True)
    colores = {"qwen3.8-27b": "#1f77b4", "gemma-4-31b-it": "#ff7f0e", "olmo-3.1-32b-think": "#2ca02c", "deepseek-v4-pro": "#d62728"}

    # A. Deliberativos por 100 nodos y por tipo, por modelo y dominio
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey="row")
    for j, (k, tit) in enumerate([("delib_100n", "Deliberativos / 100 nodos"), ("ver_100n", "VERIFICATION / 100 nodos"),
                                  ("cor_100n", "CORRECTION / 100 nodos"), ("alt_100n", "ALTERNATIVE / 100 nodos")]):
        for i, do in enumerate(DOMINIOS):
            ax = axes[i][j]
            datos = [[m[k] for m in filas if m["modelo"] == mo and m["dominio"] == do] for mo in MODELOS]
            ax.boxplot(datos, tick_labels=[NOMBRE[mo] for mo in MODELOS], showfliers=False)
            for x, (mo, d) in enumerate(zip(MODELOS, datos), start=1):
                ax.scatter([x + (hash((mo, n)) % 7 - 3) * 0.03 for n in range(len(d))], d, s=9, alpha=0.5, color=colores[mo])
            ax.set_title(f"{tit} · {do}", fontsize=10)
            ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "deliberativos_por_modelo.png", dpi=150)
    plt.close(fig)

    # B. Composición de episodios por celda (fracción de episodios por tipo)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    etiquetas, base = [], None
    col_t = {"PROGRESSION": "#7f7f7f", "SUPPORT": "#c7c7c7", "VERIFICATION": "#2ca02c", "CORRECTION": "#d62728", "ALTERNATIVE": "#1f77b4"}
    celdas = [(mo, do) for do in DOMINIOS for mo in MODELOS]
    tot = {c: sum(sum(m[f"n_{t}"] for t in TIPOS) for m in filas if (m["modelo"], m["dominio"]) == c) for c in celdas}
    import numpy as np
    base = np.zeros(len(celdas))
    for t in TIPOS:
        vals = np.array([sum(m[f"n_{t}"] for m in filas if (m["modelo"], m["dominio"]) == c) / tot[c] for c in celdas])
        ax.bar(range(len(celdas)), vals, bottom=base, color=col_t[t], label=t)
        base += vals
    ax.set_xticks(range(len(celdas)))
    ax.set_xticklabels([f"{NOMBRE[mo]}\n{do[:3]}" for mo, do in celdas])
    ax.set_ylabel("fracción de los episodios de la celda")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Composición funcional por modelo y dominio")
    fig.tight_layout()
    fig.savefig(FIG / "composicion_por_celda.png", dpi=150)
    plt.close(fig)

    # C. Longitud de todo lo generado: función de distribución acumulada por modelo y dominio
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, do in zip(axes, DOMINIOS):
        for mo in MODELOS:
            xs = sorted(x["palabras"] for x in gen if x["modelo"] == mo and x["dominio"] == do)
            if xs:
                ax.step(xs, [(i + 1) / len(xs) for i in range(len(xs))], where="post", color=colores[mo], label=f"{NOMBRE[mo]} (n = {len(xs)})")
        ax.axvline(4000, color="k", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("palabras de razonamiento (log)")
        ax.set_title(f"Todo lo generado · {do}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("fracción acumulada de cadenas")
    fig.tight_layout()
    fig.savefig(FIG / "longitud_generadas.png", dpi=150)
    plt.close(fig)

    # D. Posición relativa de los episodios deliberativos en el texto, por modelo (todas las cadenas)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, t in zip(axes, DELIBERATIVOS):
        for mo in MODELOS:
            pos = posiciones_tipo(filas_crudas(), mo, t)
            if pos:
                ax.hist(pos, bins=10, range=(0, 1), histtype="step", density=True, color=colores[mo], label=NOMBRE[mo])
        ax.set_title(f"Posición relativa de {t}")
        ax.set_xlabel("posición en el texto (0 = inicio, 1 = fin)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "posicion_deliberacion.png", dpi=150)
    plt.close(fig)


_CRUDAS: list[dict] = []


def filas_crudas() -> list[dict]:
    if not _CRUDAS:
        for l in gzip.open(RES / "anotacion_200.jsonl.gz", "rt", encoding="utf-8"):
            g = json.loads(l)
            _CRUDAS.append({"modelo": g["trace_id"].split("--")[1], "largo": max(1, g["profile"]["trace_length_chars"]),
                            "episodios": [(e["episode_type"], envolvente(e)) for e in g["episodes"]]})
    return _CRUDAS


def posiciones_tipo(crudas: list[dict], modelo: str, tipo: str) -> list[float]:
    return [(a + b) / 2 / c["largo"] for c in crudas if c["modelo"] == modelo for t, (a, b) in c["episodios"] if t == tipo]


if __name__ == "__main__":
    o = analizar()
    print(f"{o['n_cadenas']} cadenas analizadas → resultados/metricas_por_cadena.csv, analisis_rq2.json, tablas_rq2.md, figuras/")
