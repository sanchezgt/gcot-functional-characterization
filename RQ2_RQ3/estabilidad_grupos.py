"""Estabilidad de las medias de grupo entre corridas: con las tres corridas completas de las 20 cadenas de repetibilidad
(anotacion_200 + estabilidad_r1 + estabilidad_r2; 5 cadenas por modelo: 3 de medicina y 2 de derecho) calcula, por corrida y
modelo, la tasa media y mediana de episodios por 100 nodos (verificaciones, alternativas, correcciones, deliberativos, soporte)
y comprueba si el orden entre modelos y las distancias entre ellos se conservan de una corrida a otra. Responde a la objeción
de que errores que se compensan en el total podrían redistribuirse entre modelos. Escribe resultados/estabilidad_grupos.json.
Uso: python3 estabilidad_grupos.py"""
import gzip, json, statistics
from pathlib import Path

B = Path(__file__).resolve().parent / "resultados"
ids = set(json.load(open(B / "estabilidad_ids.json")))
MODELOS = ["qwen3.8-27b", "gemma-4-31b-it", "olmo-3.1-32b-think", "deepseek-v4-pro"]
NOMBRE = {"qwen3.8-27b": "Qwen", "gemma-4-31b-it": "Gemma", "olmo-3.1-32b-think": "Olmo", "deepseek-v4-pro": "DeepSeek"}
TIPOS = {"VERIFICATION": "verificaciones", "ALTERNATIVE": "alternativas", "CORRECTION": "correcciones", "SUPPORT": "soporte"}

def tasas(archivo):
    out = {}
    for r in map(json.loads, gzip.open(B / archivo, "rt", encoding="utf-8")):
        if r["trace_id"] not in ids: continue
        n = len(r["sgot_graph"]["nodes"]); cuenta = {k: 0 for k in TIPOS}
        for e in r["episodes"]:
            if e["episode_type"] in cuenta: cuenta[e["episode_type"]] += 1
        t = {TIPOS[k]: 100 * v / n for k, v in cuenta.items()}
        t["deliberativos"] = t["verificaciones"] + t["alternativas"] + t["correcciones"]
        out[r["trace_id"]] = t
    assert len(out) == 20, (archivo, len(out))
    return out

corridas = {"corrida 1 (anotacion_200)": tasas("anotacion_200.jsonl.gz"), "corrida 2 (r1)": tasas("estabilidad_r1.jsonl.gz"), "corrida 3 (r2)": tasas("estabilidad_r2.jsonl.gz")}
medidas = ["verificaciones", "alternativas", "correcciones", "deliberativos", "soporte"]
res = {}
for med in medidas:
    res[med] = {}
    print(f"\n=== {med} por 100 nodos: media (mediana) por modelo en cada corrida ===")
    print(f"{'':28s}" + "".join(f"{NOMBRE[m]:>18s}" for m in MODELOS) + "   orden")
    for nombre, tasas_c in corridas.items():
        fila = {}
        for m in MODELOS:
            vals = [t[med] for i, t in tasas_c.items() if i.split("--")[1] == m]
            fila[m] = {"media": statistics.mean(vals), "mediana": statistics.median(vals), "valores": sorted(round(v, 2) for v in vals)}
        orden = sorted(MODELOS, key=lambda m: -fila[m]["media"])
        res[med][nombre] = {NOMBRE[m]: fila[m] for m in MODELOS}; res[med][nombre]["orden_por_media"] = [NOMBRE[m] for m in orden]
        print(f"{nombre:28s}" + "".join(f"{fila[m]['media']:9.2f} ({fila[m]['mediana']:5.2f})" for m in MODELOS) + "   " + " > ".join(NOMBRE[m] for m in orden))
    # recorrido de la media de cada modelo entre corridas frente a la distancia entre modelos
    rec = {m: max(res[med][c][NOMBRE[m]]["media"] for c in corridas) - min(res[med][c][NOMBRE[m]]["media"] for c in corridas) for m in MODELOS}
    medias = {m: statistics.mean(res[med][c][NOMBRE[m]]["media"] for c in corridas) for m in MODELOS}
    print("recorrido entre corridas de la media de cada modelo: " + ", ".join(f"{NOMBRE[m]} {rec[m]:.2f}" for m in MODELOS))
    print("media de las tres corridas: " + ", ".join(f"{NOMBRE[m]} {medias[m]:.2f}" for m in MODELOS))
    res[med]["recorrido_entre_corridas"] = {NOMBRE[m]: rec[m] for m in MODELOS}; res[med]["media_tres_corridas"] = {NOMBRE[m]: medias[m] for m in MODELOS}
# estabilidad por cadena: correlación de Spearman entre corridas de la tasa de deliberativos (contexto)
import itertools
from scipy.stats import spearmanr
print("\n=== por cadena: Spearman de la tasa de deliberativos entre pares de corridas ===")
nombres = list(corridas)
for a, b in itertools.combinations(nombres, 2):
    x = [corridas[a][i]["deliberativos"] for i in sorted(ids)]; y = [corridas[b][i]["deliberativos"] for i in sorted(ids)]
    print(f"{a} vs {b}: rho = {spearmanr(x, y).correlation:.2f}")
(B / "estabilidad_grupos.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print("\nescrito resultados/estabilidad_grupos.json")
