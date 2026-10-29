"""Puntúa la salida del pipeline SGoTGraph contra la estructura declarada del corpus de control (RQ1).

Uso:
    python3 puntuar.py --corridas DIR [DIR ...] [--informe salida.json]
    python3 puntuar.py --autoprueba

Cada DIR es la raíz de una corrida completa sobre escenarios/escenarios.jsonl; se buscan recursivamente los
archivos 00_sgotgraph_objetivo3.json y se emparejan con la verdad por trace_id. La primera corrida es la
principal (sensibilidad, precisión, relaciones, rutas); con dos o más se mide además la estabilidad.

Definiciones (fijadas antes de ejecutar; README §Medidas):
- Intervalos en caracteres no blancos de la traza, para que los saltos de párrafo que el pipeline incluye en
  sus segmentos no cuenten.
- Un episodio plantado está DETECTADO si un episodio del mismo tipo lo cubre con intersección sobre unión
  ≥ 0,5 (emparejamiento uno a uno, voraz por IoU). Para CORRECTION precedida de una VERIFICATION:REFUTED se
  admite además la tolerancia de frontera descrita en TOLERANCIA_CORRECCION. DETECTADO_LAXO: algún
  episodio del mismo tipo cubre al menos la mitad del pasaje plantado (cobertura ≥ 0,5), sin exigir unicidad ni IoU.
- Desenlace correcto: el episodio emparejado tiene el desenlace plantado.
- Confusión de tipo: no detectado, pero un episodio deliberativo de otro tipo tiene IoU ≥ 0,5 con el pasaje.
- Falso positivo: episodio VERIFICATION, CORRECTION o ALTERNATIVE detectado que no tiene IoU ≥ 0,5 con ningún
  pasaje plantado de ningún tipo.
- Relación: existe una relación del tipo esperado (EVALUATES, REPLACES, COMPETES_WITH) desde el episodio
  emparejado hacia un episodio que cubre al menos la mitad del segmento objetivo (COMPETES_WITH en cualquier
  sentido). Ancla: el episodio emparejado tiene un ancla del rol requerido (TARGET o CONTRAST) que cubre al
  menos la mitad del segmento objetivo.
- Ruta verificación → corrección: en las cadenas con VERIFICATION:REFUTED y CORRECTION:RECOVERED plantados,
  alguna ruta de la capa 08 contiene los dos episodios emparejados.
- Estabilidad: por pasaje plantado, en cuántas de las corridas disponibles queda detectado.
No usa la API.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

AQUI = Path(__file__).resolve().parent
VERDAD = AQUI / "escenarios" / "verdad.json"
ENTRADA = AQUI / "escenarios" / "escenarios.jsonl"
PLANTILLAS = AQUI / "condicion_de_paso.json"

ACUERDO_CIEGO = AQUI / "escenarios" / "acuerdo_ciego.json"
# Acuerdo de la anotación a ciegas con cada pasaje plantado (ver README): permite separar la sensibilidad en
# pasajes inequívocos de la de pasajes que un anotador cuidadoso lee de otra forma.
ACUERDO = json.load(open(ACUERDO_CIEGO, encoding="utf-8"))["pasajes"] if ACUERDO_CIEGO.exists() else {}

DELIBERATIVOS = {"VERIFICATION", "CORRECTION", "ALTERNATIVE"}
ROL_ANCLA = {"VERIFICATION": "TARGET", "CORRECTION": "TARGET", "ALTERNATIVE": "CONTRAST"}
UMBRAL = 0.5
# TOLERANCIA_CORRECCION.
# La definición congelada del instrumento dice que una corrección «invalida, reemplaza, restringe o ajusta contenido
# anterior» y empieza «en el primer intervalo que realiza esa función». Cuando una corrección sigue a una verificación
# que refuta, la frase que invalida el valor erróneo es a la vez el desenlace de la verificación y el arranque de la
# corrección, y el anotador empieza la corrección unas veces ahí y otras en el recálculo; las dos lecturas cumplen la
# definición (los episodios no particionan la traza). Un solo intervalo en la verdad no puede representar ambas (con
# la verdad ampliada e IoU estricta: 34/40; con la verdad original: 22/40), así que la regla admite cualquiera de las
# dos: un episodio CORRECTION cuenta si cubre al menos la mitad del pasaje plantado y al menos la mitad del episodio
# cae dentro del tramo [inicio de la verificación refutadora anterior, fin de la corrección plantada]. Solo se aplica a
# CORRECTION con VERIFICATION:REFUTED previa; en el resto de pasajes rige la IoU ≥ 0,5 sin cambios.
TOLERANCIA_CORRECCION = "CORRECTION tras VERIFICATION:REFUTED: cobertura del pasaje ≥ 0,5 y episodio dentro del tramo verificación–corrección"


# ----------------------------------------------------------------------------- utilidades de intervalos
def chars(texto: str, a: int, b: int) -> frozenset[int]:
    return frozenset(i for i in range(max(0, a), min(len(texto), b)) if not texto[i].isspace())


def chars_episodio(texto: str, ep: dict) -> frozenset[int]:
    s: set[int] = set()
    for m in ep.get("members") or []:
        s |= chars(texto, m["start_char"], m["end_char"])
    return frozenset(s)


def iou(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def cobertura(a: frozenset, b: frozenset) -> float:
    """Fracción de b cubierta por a."""
    return len(a & b) / len(b) if b else 0.0


def corresponde(a: frozenset, b: frozenset) -> bool:
    """a y b designan el mismo tramo: al menos la mitad de uno de los dos está dentro del otro."""
    return cobertura(a, b) >= UMBRAL or cobertura(b, a) >= UMBRAL


# ----------------------------------------------------------------------------- carga
def cargar_verdad() -> tuple[dict, dict]:
    verdad = {c["id"]: c for c in json.load(open(VERDAD, encoding="utf-8"))["cadenas"]}
    textos = {}
    with open(ENTRADA, encoding="utf-8") as fh:
        for linea in fh:
            if linea.strip():
                r = json.loads(linea)
                textos[r["id"]] = r["reasoning_trace"]
    assert set(verdad) == set(textos), "verdad.json y escenarios.jsonl no coinciden; vuelva a ensamblar"
    return verdad, textos


def cargar_corrida(raiz: Path, ids: set[str]) -> dict[str, dict]:
    if raiz.suffix == ".gz":  # resultados empaquetados: una salida 00_sgotgraph_objetivo3 por línea (resultados/*.jsonl.gz)
        import gzip
        salida_gz = {}
        with gzip.open(raiz, "rt", encoding="utf-8") as fh:
            for linea in fh:
                d = json.loads(linea)
                tid = str(d.get("trace_id", "")).split("--")[0]
                if tid in ids:
                    salida_gz[tid] = d
        return salida_gz
    salida: dict[str, tuple[str, dict]] = {}
    for f in sorted(raiz.rglob("00_sgotgraph_objetivo3.json")):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  aviso: no se pudo leer {f}: {exc}", file=sys.stderr)
            continue
        tid = str(d.get("trace_id", "")).split("--")[0]
        if tid not in ids:
            continue
        # Si hay varios intentos del mismo caso, prevalece el último en orden de ruta (attempt_00N).
        salida[tid] = (str(f), d)
    return {k: v[1] for k, v in salida.items()}


# ----------------------------------------------------------------------------- puntuación de una cadena
def puntuar_cadena(v: dict, texto: str, d: dict) -> dict:
    traza_pipeline = (d.get("sgot_graph") or {}).get("trace_text")
    if traza_pipeline is not None and traza_pipeline != texto:
        return {"id": v["id"], "error": "trace_text del pipeline distinto de la entrada; offsets no comparables"}

    eps = d.get("episodes") or []
    ep_chars = {e["episode_id"]: chars_episodio(texto, e) for e in eps}
    ep_por_id = {e["episode_id"]: e for e in eps}
    segs = v["segmentos"]
    seg_chars = [chars(texto, s["start_char"], s["end_char"]) for s in segs]
    plantados = v["plantados"]
    p_chars = [chars(texto, p["start_char"], p["end_char"]) for p in plantados]

    # tramo ampliado de cada CORRECTION plantada: desde la verificación refutadora anterior hasta el fin de la corrección
    p_ext: dict[int, frozenset[int]] = {}
    for i, p in enumerate(plantados):
        if p["tipo"] != "CORRECTION":
            continue
        previas = [s for s in segs if s["rol"] == "VERIFICATION" and s["desenlace"] == "REFUTED" and s["end_char"] <= p["start_char"]]
        if previas:
            p_ext[i] = chars(texto, max(previas, key=lambda s: s["end_char"])["start_char"], p["end_char"])

    def afinidad(i: int, ec: frozenset[int]) -> float:
        val = iou(ec, p_chars[i])
        if i in p_ext and val < UMBRAL:
            # TOLERANCIA_CORRECCION: el episodio cubre la corrección plantada y se queda dentro del tramo ampliado
            alt = min(cobertura(ec, p_chars[i]), cobertura(p_ext[i], ec))
            if alt >= UMBRAL:
                val = alt
        return val

    # emparejamiento uno a uno, voraz por afinidad (IoU, con la tolerancia de frontera para CORRECTION), dentro del mismo tipo
    pares = sorted(((afinidad(i, ep_chars[e["episode_id"]]), i, e["episode_id"])
                    for i, p in enumerate(plantados) for e in eps if e["episode_type"] == p["tipo"]), reverse=True)
    asignado: dict[int, tuple[str, float]] = {}
    usados: set[str] = set()
    for val, i, eid in pares:
        if val >= UMBRAL and i not in asignado and eid not in usados:
            asignado[i] = (eid, val)
            usados.add(eid)

    relaciones = d.get("episode_relations") or []
    rutas = d.get("routes") or []
    filas = []
    for i, p in enumerate(plantados):
        mejor_mismo_tipo = max((iou(ep_chars[e["episode_id"]], p_chars[i]) for e in eps if e["episode_type"] == p["tipo"]), default=0.0)
        laxo = any(cobertura(ep_chars[e["episode_id"]], p_chars[i]) >= UMBRAL for e in eps if e["episode_type"] == p["tipo"])
        fila = {"indice": i, "tipo": p["tipo"], "desenlace": p["desenlace"], "explicitud": p["explicitud"],
                "acuerdo_ciego": ACUERDO.get(f"{v['id']}:{p['indice_segmento']}", {}).get("acuerdo_ciego"),
                "detectado": i in asignado, "detectado_laxo": laxo, "iou_mejor": round(mejor_mismo_tipo, 3),
                "desenlace_detectado": None, "desenlace_correcto": None, "confundido_con": None,
                "relacion_esperada": p["relacion_esperada"], "relacion_ok": None, "ancla_ok": None}
        if i in asignado:
            eid = asignado[i][0]
            ep = ep_por_id[eid]
            fila["episode_id"] = eid
            fila["desenlace_detectado"] = ep.get("outcome")
            fila["desenlace_correcto"] = ep.get("outcome") == p["desenlace"]
            validos = [chars(texto, o["start_char"], o["end_char"]) for o in p.get("objetivos_validos") or []]
            if validos:
                destinos = {e["episode_id"] for e in eps if e["episode_id"] != eid
                            and any(corresponde(ep_chars[e["episode_id"]], o) for o in validos)}
                tipo_rel = p["relacion_esperada"]
                fila["relacion_ok"] = any(
                    r.get("relation_type") == tipo_rel and (
                        (r.get("source_episode_id") == eid and r.get("target_episode_id") in destinos)
                        or (tipo_rel == "COMPETES_WITH" and r.get("target_episode_id") == eid and r.get("source_episode_id") in destinos))
                    for r in relaciones)
                rol = ROL_ANCLA[p["tipo"]]
                fila["ancla_ok"] = any(a.get("anchor_role") == rol and any(corresponde(chars(texto, a["start_char"], a["end_char"]), o) for o in validos)
                                       for a in ep.get("anchors") or [])
        else:
            for e in eps:
                if e["episode_type"] in DELIBERATIVOS and e["episode_type"] != p["tipo"] and iou(ep_chars[e["episode_id"]], p_chars[i]) >= UMBRAL:
                    fila["confundido_con"] = f"{e['episode_type']}:{e.get('outcome')}"
                    break
        filas.append(fila)

    # falsos positivos
    fps = []
    for e in eps:
        if e["episode_type"] not in DELIBERATIVOS or e["episode_id"] in usados:
            continue
        if any(iou(ep_chars[e["episode_id"]], pc) >= UMBRAL for pc in p_chars):
            continue
        roles = sorted({segs[k]["rol"] + (":distractor" if segs[k].get("distractor") else "")
                        for k, sc in enumerate(seg_chars) if cobertura(ep_chars[e["episode_id"]], sc) >= UMBRAL
                        or (sc and len(ep_chars[e["episode_id"]] & sc) / max(1, len(ep_chars[e["episode_id"]])) >= UMBRAL)})
        fps.append({"episode_id": e["episode_id"], "tipo": e["episode_type"], "desenlace": e.get("outcome"),
                    "segmentos_solapados": roles,
                    "texto": " ".join(m.get("text", "") for m in e.get("members") or [])[:200]})

    # ruta verificación → corrección
    ruta = None
    iv = next((f["episode_id"] for f in filas if f["tipo"] == "VERIFICATION" and f["desenlace"] == "REFUTED" and f["detectado"]), None)
    ic = next((f["episode_id"] for f in filas if f["tipo"] == "CORRECTION" and f["detectado"]), None)
    tiene_par = any(p["tipo"] == "VERIFICATION" and p["desenlace"] == "REFUTED" for p in plantados) and any(p["tipo"] == "CORRECTION" for p in plantados)
    if tiene_par:
        ruta = bool(iv and ic and any(iv in (r.get("episode_ids") or []) and ic in (r.get("episode_ids") or []) for r in rutas))

    # línea abandonada (P07, P16): la progresión que la cubre debe quedar INTERRUPTED o ABANDONED
    abandonada = None
    idx_ab = [k for k, s in enumerate(segs) if s["rol"] == "PROGRESSION" and s["desenlace"] == "ABANDONED"]
    if idx_ab:
        sc = seg_chars[idx_ab[0]]
        prog = [e for e in eps if e["episode_type"] == "PROGRESSION" and corresponde(ep_chars[e["episode_id"]], sc)]
        abandonada = any(e.get("outcome") in ("INTERRUPTED", "ABANDONED") for e in prog)

    return {"id": v["id"], "plantilla": v["plantilla"], "dominio": v["dominio"], "estilo": v["estilo"],
            "plantados": filas, "falsos_positivos": fps, "n_episodios": len(eps),
            "n_deliberativos_detectados": sum(e["episode_type"] in DELIBERATIVOS for e in eps), "ruta_ver_corr": ruta,
            "linea_abandonada_ok": abandonada}


# ----------------------------------------------------------------------------- agregados
def proporcion(xs: list[bool]) -> dict:
    xs = [x for x in xs if x is not None]
    return {"n": len(xs), "valor": round(sum(xs) / len(xs), 3) if xs else None}


def agregar(res: list[dict]) -> dict:
    validos = [r for r in res if "error" not in r]
    filas = [(r, f) for r in validos for f in r["plantados"]]

    def por(clave):
        g = collections.defaultdict(list)
        for r, f in filas:
            k = f[clave] if clave in f else r[clave]
            g[k].append(f)
        return {str(k): {"sensibilidad_tipo": proporcion([f["detectado"] for f in fs])["valor"],
                         "sensibilidad_tipo_desenlace": proporcion([f["detectado"] and f["desenlace_correcto"] for f in fs])["valor"],
                         "n": len(fs)} for k, fs in sorted(g.items(), key=lambda kv: str(kv[0]))}

    tp = sum(f["detectado"] for _, f in filas)
    confus = sum(1 for _, f in filas if f["confundido_con"])
    fp = sum(len(r["falsos_positivos"]) for r in validos)
    nulas = [r for r in validos if r["plantilla"] in ("P00", "P12")]
    detectados = [f for _, f in filas if f["detectado"]]
    expl = [f for _, f in filas if f["explicitud"] == "explicita"]
    return {
        "cadenas_puntuadas": len(validos), "cadenas_con_error": [r["id"] for r in res if "error" in r],
        "pasajes_plantados": len(filas),
        "sensibilidad_tipo": proporcion([f["detectado"] for _, f in filas]),
        "sensibilidad_tipo_laxa": proporcion([f["detectado_laxo"] for _, f in filas]),
        "sensibilidad_tipo_desenlace": proporcion([f["detectado"] and f["desenlace_correcto"] for _, f in filas]),
        "sensibilidad_explicitos_tipo": proporcion([f["detectado"] for f in expl]),
        "sensibilidad_explicitos_tipo_desenlace": proporcion([f["detectado"] and f["desenlace_correcto"] for f in expl]),
        "desenlace_correcto_entre_detectados": proporcion([f["desenlace_correcto"] for f in detectados]),
        "confusiones_de_tipo": confus,
        "falsos_positivos_total": fp,
        "falsos_positivos_por_cadena": round(fp / len(validos), 3) if validos else None,
        "falsos_positivos_por_cadena_nulas_P00_P12": round(sum(len(r["falsos_positivos"]) for r in nulas) / len(nulas), 3) if nulas else None,
        "falsos_positivos_por_cadena_P00": (lambda xs: round(sum(len(r["falsos_positivos"]) for r in xs) / len(xs), 3) if xs else None)([r for r in validos if r["plantilla"] == "P00"]),
        "falsos_positivos_por_cadena_P12": (lambda xs: round(sum(len(r["falsos_positivos"]) for r in xs) / len(xs), 3) if xs else None)([r for r in validos if r["plantilla"] == "P12"]),
        "precision": round(tp / (tp + fp + confus), 3) if (tp + fp + confus) else None,
        "relacion_ok_entre_detectados": proporcion([f["relacion_ok"] for f in detectados]),
        "relacion_ok_explicitos": proporcion([f["relacion_ok"] for f in detectados if f["explicitud"] == "explicita"]),
        "ancla_ok_entre_detectados": proporcion([f["ancla_ok"] for f in detectados]),
        "ruta_ver_corr": proporcion([r["ruta_ver_corr"] for r in validos]),
        "linea_abandonada_ok": proporcion([r["linea_abandonada_ok"] for r in validos]),
        "por_tipo": por("tipo"), "por_explicitud": por("explicitud"), "por_acuerdo_ciego": por("acuerdo_ciego"), "por_plantilla": por("plantilla"),
        "por_dominio": por("dominio"), "por_estilo": por("estilo"),
    }


def estabilidad(corridas: list[dict[str, dict]]) -> dict:
    conteo: dict[tuple[str, int], list[bool]] = collections.defaultdict(list)
    explic: dict[tuple[str, int], str] = {}
    for res in corridas:
        for r in res.values():
            if "error" in r:
                continue
            for f in r["plantados"]:
                conteo[(r["id"], f["indice"])].append(f["detectado"])
                explic[(r["id"], f["indice"])] = f["explicitud"]
    multi = {k: v for k, v in conteo.items() if len(v) >= 2}
    if not multi:
        return {"pasajes_con_varias_corridas": 0}
    dist = collections.Counter(f"{sum(v)}/{len(v)}" for v in multi.values())
    expl = [v for k, v in multi.items() if explic[k] == "explicita"]
    return {"pasajes_con_varias_corridas": len(multi), "distribucion_detecciones": dict(sorted(dist.items())),
            "explicitos_en_al_menos_2_de_3": proporcion([sum(v) >= 2 for v in expl if len(v) >= 3]),
            "acuerdo_completo": proporcion([len(set(v)) == 1 for v in multi.values()])}


def condicion_de_paso(ag: dict, est: dict) -> dict:
    umbrales = json.load(open(PLANTILLAS, encoding="utf-8"))["condicion_de_paso_propuesta"]
    chequeos = {
        "sensibilidad_tipo_y_desenlace_explicitos": (ag["sensibilidad_explicitos_tipo_desenlace"]["valor"], ">=", umbrales["sensibilidad_tipo_y_desenlace_explicitos"]),
        "sensibilidad_tipo_explicitos": (ag["sensibilidad_explicitos_tipo"]["valor"], ">=", umbrales["sensibilidad_tipo_explicitos"]),
        "falsos_positivos_por_cadena_P00": (ag["falsos_positivos_por_cadena_P00"], "<=", umbrales["falsos_positivos_por_cadena_P00"]),
        "recuperacion_relaciones_rutas_explicitas": (ag["relacion_ok_explicitos"]["valor"], ">=", umbrales["recuperacion_relaciones_rutas_explicitas"]),
        "estabilidad_explicitos_en_2_de_3_corridas": ((est.get("explicitos_en_al_menos_2_de_3") or {}).get("valor"), ">=", umbrales["estabilidad_explicitos_en_2_de_3_corridas"]),
    }
    salida = {}
    for k, (valor, op, umbral) in chequeos.items():
        cumple = None if valor is None else (valor >= umbral if op == ">=" else valor <= umbral)
        salida[k] = {"valor": valor, "umbral": f"{op} {umbral}", "cumple": cumple}
    salida["PASA"] = all(x["cumple"] for x in salida.values() if isinstance(x, dict)) if all(
        x["cumple"] is not None for x in salida.values() if isinstance(x, dict)) else None
    return salida


def evaluar(corridas_salida: list[dict[str, dict]], verdad: dict, textos: dict) -> dict:
    resultados = []
    for salida in corridas_salida:
        resultados.append({tid: puntuar_cadena(verdad[tid], textos[tid], d) for tid, d in salida.items()})
    principal = resultados[0]
    faltan = sorted(set(verdad) - set(principal))
    ag = agregar(list(principal.values()))
    est = estabilidad(resultados)
    return {"regla_de_deteccion": f"IoU >= {UMBRAL}; {TOLERANCIA_CORRECCION}",
            "cadenas_sin_salida_en_corrida_principal": faltan, "agregado": ag, "estabilidad": est,
            "condicion_de_paso": condicion_de_paso(ag, est), "por_cadena": list(principal.values())}


def imprimir(inf: dict) -> None:
    ag = inf["agregado"]
    f = lambda x: "—" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
    print(f"cadenas puntuadas: {ag['cadenas_puntuadas']} | sin salida: {len(inf['cadenas_sin_salida_en_corrida_principal'])} | con error: {len(ag['cadenas_con_error'])}")
    print(f"sensibilidad (tipo): {f(ag['sensibilidad_tipo']['valor'])}  laxa: {f(ag['sensibilidad_tipo_laxa']['valor'])}  tipo+desenlace: {f(ag['sensibilidad_tipo_desenlace']['valor'])}  (n={ag['pasajes_plantados']})")
    print(f"explícitos: tipo {f(ag['sensibilidad_explicitos_tipo']['valor'])}  tipo+desenlace {f(ag['sensibilidad_explicitos_tipo_desenlace']['valor'])}")
    print(f"precisión: {f(ag['precision'])} | FP por cadena: {f(ag['falsos_positivos_por_cadena'])} (P00 {f(ag['falsos_positivos_por_cadena_P00'])}, P12 {f(ag['falsos_positivos_por_cadena_P12'])}) | confusiones de tipo: {ag['confusiones_de_tipo']}")
    print(f"relación correcta (detectados): {f(ag['relacion_ok_entre_detectados']['valor'])}  ancla: {f(ag['ancla_ok_entre_detectados']['valor'])}  ruta ver→corr: {f(ag['ruta_ver_corr']['valor'])}  línea abandonada: {f(ag['linea_abandonada_ok']['valor'])}")
    for clave in ("por_explicitud", "por_acuerdo_ciego", "por_tipo"):
        print(f"{clave}: " + "; ".join(f"{k} {f(v['sensibilidad_tipo'])}/{f(v['sensibilidad_tipo_desenlace'])} (n={v['n']})" for k, v in ag[clave].items()))
    print("estabilidad:", json.dumps(inf["estabilidad"], ensure_ascii=False))
    cp = inf["condicion_de_paso"]
    print("condición de paso:", " | ".join(f"{k}={f(v['valor'])} {v['umbral']} {'✓' if v['cumple'] else ('✗' if v['cumple'] is False else '?')}"
                                          for k, v in cp.items() if isinstance(v, dict)), "→ PASA" if cp["PASA"] else ("→ NO PASA" if cp["PASA"] is False else "→ incompleta"))


# ----------------------------------------------------------------------------- autoprueba (sin API)
def salida_sintetica(v: dict, texto: str, *, quitar_implicitos=False, invertir_desenlace=False, fp_en_nulas=False) -> dict:
    """Construye una salida de objetivo 3 a partir de la verdad: un episodio por segmento, con relaciones y ruta."""
    eps, rels = [], []
    idx_ep: dict[int, str] = {}
    plantado_en = {p["indice_segmento"]: p for p in v["plantados"]}
    for k, s in enumerate(v["segmentos"]):
        p = plantado_en.get(k)
        if p and quitar_implicitos and p["explicitud"] == "implicita":
            tipo, desenlace = "PROGRESSION", "ADVANCES"
        elif p:
            tipo, desenlace = p["tipo"], p["desenlace"]
            if invertir_desenlace and tipo == "VERIFICATION":
                desenlace = {"CONFIRMED": "REFUTED", "REFUTED": "CONFIRMED", "INCONCLUSIVE": "CONFIRMED"}[desenlace]
        else:
            tipo = "SUPPORT" if s["rol"] == "SUPPORT" else "PROGRESSION"
            desenlace = "ATTACHED_ACTIVE" if tipo == "SUPPORT" else ("ABANDONED" if s.get("desenlace") == "ABANDONED" else "ADVANCES")
        eid = f"ep{k}"
        idx_ep[k] = eid
        anclas = []
        if p and tipo == p["tipo"] and p.get("objetivo"):
            o = p["objetivo"]
            anclas = [{"start_char": o["start_char"], "end_char": o["end_char"], "text": texto[o["start_char"]:o["end_char"]], "anchor_role": ROL_ANCLA[tipo]}]
        eps.append({"episode_id": eid, "episode_type": tipo, "outcome": desenlace, "anchors": anclas,
                    "members": [{"start_char": s["start_char"], "end_char": s["end_char"], "text": texto[s["start_char"]:s["end_char"]]}]})
    for p in v["plantados"]:
        if p.get("objetivo") and eps[p["indice_segmento"]]["episode_type"] == p["tipo"]:
            rels.append({"relation_type": p["relacion_esperada"], "source_episode_id": idx_ep[p["indice_segmento"]],
                         "target_episode_id": idx_ep[p["objetivo"]["indice_segmento"]]})
    if fp_en_nulas and v["plantilla"] in ("P00", "P12"):
        s = v["segmentos"][1]
        eps.append({"episode_id": "fp", "episode_type": "VERIFICATION", "outcome": "CONFIRMED", "anchors": [],
                    "members": [{"start_char": s["start_char"], "end_char": s["end_char"], "text": texto[s["start_char"]:s["end_char"]]}]})
    rutas = [{"route_type": "CORRECTION", "episode_ids": [e["episode_id"] for e in eps]}]
    return {"trace_id": v["id"], "sgot_graph": {"trace_text": texto}, "episodes": eps, "episode_relations": rels, "routes": rutas}


def autoprueba() -> None:
    verdad, textos = cargar_verdad()
    perfecta = {tid: salida_sintetica(v, textos[tid]) for tid, v in verdad.items()}
    inf = evaluar([perfecta, perfecta, perfecta], verdad, textos)
    ag = inf["agregado"]
    assert ag["sensibilidad_tipo_desenlace"]["valor"] == 1.0, ag["sensibilidad_tipo_desenlace"]
    assert ag["falsos_positivos_total"] == 0 and ag["precision"] == 1.0, (ag["falsos_positivos_total"], ag["precision"])
    assert ag["relacion_ok_entre_detectados"]["valor"] == 1.0 and ag["ancla_ok_entre_detectados"]["valor"] == 1.0
    assert ag["ruta_ver_corr"]["valor"] == 1.0 and inf["condicion_de_paso"]["PASA"] is True
    n_abandonada = sum(any(s["rol"] == "PROGRESSION" and s["desenlace"] == "ABANDONED" for s in v["segmentos"]) for v in verdad.values())
    assert ag["linea_abandonada_ok"]["valor"] == 1.0 and ag["linea_abandonada_ok"]["n"] == n_abandonada, ag["linea_abandonada_ok"]
    print("autoprueba 1 (salida perfecta): OK")
    imprimir(inf)

    n_impl = sum(p["explicitud"] == "implicita" for v in verdad.values() for p in v["plantados"])
    n_tot = sum(len(v["plantados"]) for v in verdad.values())
    n_ver = sum(p["tipo"] == "VERIFICATION" for v in verdad.values() for p in v["plantados"])
    degradada = {tid: salida_sintetica(v, textos[tid], quitar_implicitos=True, invertir_desenlace=True, fp_en_nulas=True)
                 for tid, v in verdad.items()}
    inf = evaluar([degradada, perfecta, perfecta], verdad, textos)
    ag = inf["agregado"]
    assert abs(ag["sensibilidad_tipo"]["valor"] - round((n_tot - n_impl) / n_tot, 3)) < 1e-9, ag["sensibilidad_tipo"]
    assert ag["por_explicitud"]["implicita"]["sensibilidad_tipo"] == 0.0
    assert ag["falsos_positivos_por_cadena_nulas_P00_P12"] == 1.0 and ag["falsos_positivos_por_cadena_P00"] == 1.0
    ver_expl = sum(p["tipo"] == "VERIFICATION" and p["explicitud"] == "explicita" for v in verdad.values() for p in v["plantados"])
    det = n_tot - n_impl
    assert abs(ag["desenlace_correcto_entre_detectados"]["valor"] - round((det - ver_expl) / det, 3)) < 1e-9
    assert inf["estabilidad"]["distribucion_detecciones"].get("2/3") == n_impl
    assert inf["condicion_de_paso"]["PASA"] is False
    print(f"\nautoprueba 2 (sin implícitos, verificaciones con desenlace invertido, 1 FP por cadena nula): OK ({n_ver} verificaciones)")
    imprimir(inf)

    # autoprueba 3: tolerancia de frontera de CORRECTION. Se amplía cada corrección sintética hacia atrás hasta el inicio
    # de la verificación refutadora anterior (la lectura alternativa del anotador); debe seguir detectada, sin crear
    # falsos positivos, y la IoU estricta por sí sola debe perder una parte (prueba de que la tolerancia actúa).
    ampliada = {}
    n_ampliadas = 0
    for tid, v in verdad.items():
        d = salida_sintetica(v, textos[tid])
        for e in d["episodes"]:
            if e["episode_type"] != "CORRECTION":
                continue
            ini = e["members"][0]["start_char"]
            previas = [s for s in v["segmentos"] if s["rol"] == "VERIFICATION" and s["desenlace"] == "REFUTED" and s["end_char"] <= ini]
            if previas:
                e["members"][0]["start_char"] = max(previas, key=lambda s: s["end_char"])["start_char"]
                n_ampliadas += 1
        ampliada[tid] = d
    ag = evaluar([ampliada], verdad, textos)["agregado"]
    solo_iou = sum(f["iou_mejor"] >= UMBRAL for r in evaluar([ampliada], verdad, textos)["por_cadena"] for f in r["plantados"] if f["tipo"] == "CORRECTION")
    n_corr = sum(p["tipo"] == "CORRECTION" for v in verdad.values() for p in v["plantados"])
    assert n_ampliadas > 0 and solo_iou < n_corr, (solo_iou, n_corr)
    assert ag["por_tipo"]["CORRECTION"]["sensibilidad_tipo"] == 1.0, ag["por_tipo"]["CORRECTION"]
    assert ag["falsos_positivos_total"] == 0
    print(f"\nautoprueba 3 (corrección ampliada hasta la verificación en {n_ampliadas} casos): IoU sola {solo_iou}/{n_corr}, con tolerancia {n_corr}/{n_corr}, sin FP: OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corridas", nargs="+", type=Path, help="raíces de corrida; la primera es la principal")
    ap.add_argument("--informe", type=Path, help="ruta del informe JSON")
    ap.add_argument("--autoprueba", action="store_true", help="comprueba el puntuador con salidas sintéticas")
    args = ap.parse_args()
    if args.autoprueba:
        autoprueba()
        return
    if not args.corridas:
        ap.error("indique --corridas o --autoprueba")
    verdad, textos = cargar_verdad()
    salidas = []
    for raiz in args.corridas:
        s = cargar_corrida(raiz, set(verdad))
        print(f"{raiz}: {len(s)} cadenas con salida")
        salidas.append(s)
    inf = evaluar(salidas, verdad, textos)
    imprimir(inf)
    if args.informe:
        args.informe.write_text(json.dumps(inf, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"informe → {args.informe}")


if __name__ == "__main__":
    main()
