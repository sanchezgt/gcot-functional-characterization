"""Estabilidad del instrumento en trazas reales: acuerdo entre tres corridas independientes de las mismas 20 cadenas.

Uso:
    python3 estabilidad.py [--empaquetar DIR_R1 DIR_R2]

Con --empaquetar, recoge las salidas 00_sgotgraph_objetivo3.json de las dos repeticiones (carpetas de salida del pipeline) en
resultados/estabilidad_r1.jsonl.gz y estabilidad_r2.jsonl.gz. Después (o sin la opción) mide, sobre las 20 cadenas de
resultados/estabilidad_ids.json, con la corrida principal (anotacion_200.jsonl.gz) como tercera:

- Episodios deliberativos (VERIFICATION, CORRECTION, ALTERNATIVE): para cada episodio de una corrida, si en otra hay uno del
  mismo tipo cuya envolvente de texto se solapa (IoU ≥ 0,5; también se da con cualquier solapamiento) y si además coincide
  el desenlace. Como no hay verdad, la medida es simétrica: fracción de episodios de cada corrida con pareja en la otra.
- Perfil por cadena: nodos, episodios y tasas por 100 nodos en cada corrida; Spearman entre corridas sobre las 20 cadenas y
  diferencia absoluta relativa a la media.
- Orden de los modelos: mediana de VERIFICATION y deliberativos por 100 nodos por modelo en cada corrida.

Salida: resultados/informe_estabilidad.json y un resumen por pantalla.
"""
from __future__ import annotations

import argparse
import gzip
import json
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

AQUI = Path(__file__).resolve().parent
RES = AQUI / "resultados"
DELIBERATIVOS = ("VERIFICATION", "CORRECTION", "ALTERNATIVE")
NOMBRE = {"qwen3.8-27b": "Qwen", "gemma-4-31b-it": "Gemma", "olmo-3.1-32b-think": "Olmo", "deepseek-v4-pro": "DeepSeek"}


def empaquetar(carpeta: Path, destino: Path, ids: set[str]) -> None:
    salidas: dict[str, tuple[str, dict]] = {}
    for f in carpeta.rglob("00_sgotgraph_objetivo3.json"):
        g = json.load(open(f, encoding="utf-8"))
        if g["trace_id"] in ids and (g["trace_id"] not in salidas or str(f) > salidas[g["trace_id"]][0]):
            salidas[g["trace_id"]] = (str(f), g)
    faltan = ids - set(salidas)
    with gzip.open(destino, "wt", encoding="utf-8") as fh:
        for tid in sorted(salidas):
            fh.write(json.dumps(salidas[tid][1], ensure_ascii=False) + "\n")
    print(f"{destino.name}: {len(salidas)} salidas; faltan {sorted(faltan)}; validación "
          f"{dict(Counter(g['validation'].get('status') for _, g in salidas.values()))}")


def cargar(ruta: Path, ids: set[str]) -> dict[str, dict]:
    out = {}
    for l in gzip.open(ruta, "rt", encoding="utf-8"):
        g = json.loads(l)
        if g["trace_id"] in ids:
            out[g["trace_id"]] = g
    return out


def envolvente(ep: dict) -> tuple[int, int]:
    return min(m["start_char"] for m in ep["members"]), max(m["end_char"] for m in ep["members"])


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def emparejar(eps_x: list[dict], eps_y: list[dict], umbral: float) -> list[dict]:
    """Para cada episodio deliberativo de X, la mejor pareja del mismo tipo en Y (IoU máximo ≥ umbral; umbral 0 = cualquier solapamiento)."""
    res = []
    for e in eps_x:
        if e["episode_type"] not in DELIBERATIVOS:
            continue
        a = envolvente(e)
        mejor, mejor_iou = None, 0.0
        for f in eps_y:
            if f["episode_type"] != e["episode_type"]:
                continue
            v = iou(a, envolvente(f))
            if v > mejor_iou:
                mejor, mejor_iou = f, v
        ok = mejor is not None and (mejor_iou >= umbral if umbral > 0 else mejor_iou > 0)
        res.append({"tipo": e["episode_type"], "pareja": ok, "mismo_desenlace": ok and mejor["outcome"] == e["outcome"], "iou": mejor_iou})
    return res


def perfil(g: dict) -> dict:
    n = len(g["sgot_graph"]["nodes"])
    c = Counter(e["episode_type"] for e in g["episodes"])
    return {"nodos": n, "episodios": len(g["episodes"]), "delib_100n": 100 * sum(c[t] for t in DELIBERATIVOS) / n,
            "ver_100n": 100 * c["VERIFICATION"] / n, "cor_100n": 100 * c["CORRECTION"] / n, "alt_100n": 100 * c["ALTERNATIVE"] / n,
            "cobertura": g["profile"]["coverage"]["overall"]["ratio"]}


def prop(xs: list[bool]) -> dict:
    return {"valor": round(sum(xs) / len(xs), 3) if xs else None, "n": len(xs)}


def medir(corridas: list[dict[str, dict]]) -> dict:
    ids = sorted(set.intersection(*(set(c) for c in corridas)))
    inf: dict = {"cadenas": len(ids), "corridas": len(corridas), "episodios": {}, "perfil": {}, "orden_modelos": {}}

    # 1. Episodios deliberativos, por pares de corridas y en conjunto
    for umbral, etiqueta in ((0.5, "iou_0.5"), (0.0, "solapamiento")):
        todos, por_tipo, por_modelo = [], defaultdict(list), defaultdict(list)
        for i, j in combinations(range(len(corridas)), 2):
            for tid in ids:
                for x, y in ((i, j), (j, i)):
                    for r in emparejar(corridas[x][tid]["episodes"], corridas[y][tid]["episodes"], umbral):
                        todos.append(r); por_tipo[r["tipo"]].append(r); por_modelo[tid.split("--")[1]].append(r)
        inf["episodios"][etiqueta] = {
            "con_pareja_mismo_tipo": prop([r["pareja"] for r in todos]),
            "con_pareja_mismo_tipo_y_desenlace": prop([r["mismo_desenlace"] for r in todos]),
            "por_tipo": {t: {"pareja": prop([r["pareja"] for r in rs]), "pareja_y_desenlace": prop([r["mismo_desenlace"] for r in rs])} for t, rs in sorted(por_tipo.items())},
            "por_modelo": {NOMBRE[m]: {"pareja": prop([r["pareja"] for r in rs]), "pareja_y_desenlace": prop([r["mismo_desenlace"] for r in rs])} for m, rs in sorted(por_modelo.items())},
        }
    # episodios presentes en las 3 corridas (desde la corrida principal, índice 0)
    if len(corridas) >= 3:
        en_todas = []
        for tid in ids:
            p1 = emparejar(corridas[0][tid]["episodes"], corridas[1][tid]["episodes"], 0.5)
            p2 = emparejar(corridas[0][tid]["episodes"], corridas[2][tid]["episodes"], 0.5)
            en_todas += [a["pareja"] and b["pareja"] for a, b in zip(p1, p2)]
        inf["episodios"]["principal_con_pareja_en_las_otras_dos_iou_0.5"] = prop(en_todas)

    # 2. Perfil por cadena entre corridas
    try:
        from scipy import stats
    except ImportError:  # pragma: no cover
        stats = None
    perfiles = [{tid: perfil(c[tid]) for tid in ids} for c in corridas]
    for k in ("nodos", "episodios", "delib_100n", "ver_100n", "cor_100n", "alt_100n", "cobertura"):
        rhos, difs = [], []
        for i, j in combinations(range(len(corridas)), 2):
            a = [perfiles[i][t][k] for t in ids]; b = [perfiles[j][t][k] for t in ids]
            if stats is not None:
                rhos.append(float(stats.spearmanr(a, b).statistic))
            difs += [abs(x - y) / ((x + y) / 2) if (x + y) else 0.0 for x, y in zip(a, b)]
        inf["perfil"][k] = {"spearman_entre_corridas": [round(r, 3) for r in rhos], "spearman_media": round(statistics.fmean(rhos), 3) if rhos else None,
                            "dif_relativa_mediana": round(statistics.median(difs), 3)}
    inf["perfil"]["por_cadena"] = {tid: {k: [round(p[tid][k], 2) for p in perfiles] for k in ("nodos", "episodios", "delib_100n", "ver_100n")} for tid in ids}

    # 3. Orden de los modelos en cada corrida
    for k in ("ver_100n", "delib_100n", "alt_100n"):
        inf["orden_modelos"][k] = []
        for p in perfiles:
            med = {NOMBRE[m]: round(statistics.median([p[t][k] for t in ids if t.split("--")[1] == m]), 2) for m in NOMBRE}
            inf["orden_modelos"][k].append(dict(sorted(med.items(), key=lambda kv: -kv[1])))
    return inf


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--empaquetar", nargs=2, type=Path, metavar=("DIR_R1", "DIR_R2"))
    a = ap.parse_args()
    ids = set(json.load(open(RES / "estabilidad_ids.json")))
    if a.empaquetar:
        for carpeta, nombre in zip(a.empaquetar, ("estabilidad_r1.jsonl.gz", "estabilidad_r2.jsonl.gz")):
            empaquetar(carpeta, RES / nombre, ids)
    corridas = [cargar(RES / "anotacion_200.jsonl.gz", ids), cargar(RES / "estabilidad_r1.jsonl.gz", ids), cargar(RES / "estabilidad_r2.jsonl.gz", ids)]
    inf = medir(corridas)
    json.dump(inf, open(RES / "informe_estabilidad.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    e = inf["episodios"]
    print(f"{inf['cadenas']} cadenas × {inf['corridas']} corridas")
    print("episodios deliberativos con pareja del mismo tipo (IoU ≥ 0,5):", e["iou_0.5"]["con_pareja_mismo_tipo"], "; con desenlace:", e["iou_0.5"]["con_pareja_mismo_tipo_y_desenlace"])
    print("  con cualquier solapamiento:", e["solapamiento"]["con_pareja_mismo_tipo"], "; con desenlace:", e["solapamiento"]["con_pareja_mismo_tipo_y_desenlace"])
    print("  por tipo (IoU ≥ 0,5):", {t: v["pareja"]["valor"] for t, v in e["iou_0.5"]["por_tipo"].items()})
    print("  por modelo (IoU ≥ 0,5):", {m: v["pareja"]["valor"] for m, v in e["iou_0.5"]["por_modelo"].items()})
    if "principal_con_pareja_en_las_otras_dos_iou_0.5" in e:
        print("  de la corrida principal, con pareja en las otras dos:", e["principal_con_pareja_en_las_otras_dos_iou_0.5"])
    print("perfil por cadena, Spearman medio entre corridas:", {k: v["spearman_media"] for k, v in inf["perfil"].items() if k != "por_cadena"})
    print("  diferencia relativa mediana:", {k: v["dif_relativa_mediana"] for k, v in inf["perfil"].items() if k != "por_cadena"})
    for k, ords in inf["orden_modelos"].items():
        print(f"orden de modelos por {k}:", ords)


if __name__ == "__main__":
    main()
