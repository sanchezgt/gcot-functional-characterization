# Inferencia de RQ2 (generada por inferencia.py)

## 1. Regresión de recuentos: episodios ~ log(nodos) + modelo + dominio, errores robustos agrupados por problema

Razón de tasas [IC 95 %] (p). Referencias: Qwen y medicina. Elasticidad = exp(β de log(nodos)): cuánto crece el recuento al duplicar los nodos sería 2^β.

| Recuento | Familia | β log(nodos) | Gemma / Qwen | Olmo / Qwen | DeepSeek / Qwen | derecho / medicina | Interacción modelo × dominio (Wald, p) |
|---|---|---|---|---|---|---|---|
| VERIFICATION | binomial negativa (NB2) | 0,28 (< 0,001) | 3,81 [2,79–5,21] (< 0,001) | 1,12 [0,81–1,54] (0,496) | 1,05 [0,77–1,43] (0,742) | 0,93 [0,73–1,18] (0,565) | χ²(3) = 8,5, p = 0,037 |
| CORRECTION | binomial negativa (NB2) | 0,90 (< 0,001) | 0,97 [0,56–1,69] (0,915) | 1,20 [0,78–1,84] (0,405) | 0,92 [0,48–1,73] (0,789) | 2,18 [1,43–3,34] (< 0,001) | χ²(3) = 4,0, p = 0,266 |
| ALTERNATIVE | binomial negativa (NB2) | 0,46 (< 0,001) | 2,92 [1,98–4,29] (< 0,001) | 3,08 [1,94–4,91] (< 0,001) | 1,42 [0,82–2,44] (0,209) | 0,87 [0,60–1,25] (0,453) | χ²(3) = 6,4, p = 0,094 |
| SUPPORT | binomial negativa (NB2) | 0,16 (0,003) | 3,36 [2,67–4,22] (< 0,001) | 3,99 [3,18–5,00] (< 0,001) | 1,09 [0,82–1,46] (0,556) | 0,97 [0,81–1,16] (0,743) | χ²(3) = 2,4, p = 0,497 |
| deliberativos (VER+COR+ALT) | binomial negativa (NB2) | 0,49 (< 0,001) | 2,61 [2,12–3,21] (< 0,001) | 1,88 [1,45–2,44] (< 0,001) | 1,13 [0,84–1,53] (0,410) | 1,04 [0,82–1,32] (0,755) | χ²(3) = 5,4, p = 0,144 |
| episodios (todos) | binomial negativa (NB2) | 0,27 (< 0,001) | 2,77 [2,41–3,19] (< 0,001) | 2,55 [2,17–3,01] (< 0,001) | 1,07 [0,88–1,30] (0,515) | 1,02 [0,88–1,18] (0,813) | χ²(3) = 6,1, p = 0,109 |

Contrastes entre modelos (razón de tasas A / B [IC 95 %], p de Holm sobre los 6 pares):

**VERIFICATION**

| A | B | A / B | IC 95 % | p Holm |
|---|---|---|---|---|
| Qwen | Gemma | 0,26 | 0,19–0,36 | < 0,001 |
| Qwen | Olmo | 0,89 | 0,65–1,23 | 1,000 |
| Qwen | DeepSeek | 0,95 | 0,70–1,29 | 1,000 |
| Gemma | Olmo | 3,41 | 2,59–4,50 | < 0,001 |
| Gemma | DeepSeek | 3,62 | 2,84–4,61 | < 0,001 |
| Olmo | DeepSeek | 1,06 | 0,76–1,48 | 1,000 |

**CORRECTION**

| A | B | A / B | IC 95 % | p Holm |
|---|---|---|---|---|
| Qwen | Gemma | 1,03 | 0,59–1,80 | 1,000 |
| Qwen | Olmo | 0,83 | 0,54–1,28 | 1,000 |
| Qwen | DeepSeek | 1,09 | 0,58–2,06 | 1,000 |
| Gemma | Olmo | 0,81 | 0,49–1,33 | 1,000 |
| Gemma | DeepSeek | 1,06 | 0,54–2,09 | 1,000 |
| Olmo | DeepSeek | 1,31 | 0,73–2,34 | 1,000 |

**ALTERNATIVE**

| A | B | A / B | IC 95 % | p Holm |
|---|---|---|---|---|
| Qwen | Gemma | 0,34 | 0,23–0,50 | < 0,001 |
| Qwen | Olmo | 0,32 | 0,20–0,52 | < 0,001 |
| Qwen | DeepSeek | 0,71 | 0,41–1,22 | 0,418 |
| Gemma | Olmo | 0,95 | 0,58–1,55 | 0,826 |
| Gemma | DeepSeek | 2,06 | 1,26–3,35 | 0,015 |
| Olmo | DeepSeek | 2,18 | 1,19–3,99 | 0,036 |

**SUPPORT**

| A | B | A / B | IC 95 % | p Holm |
|---|---|---|---|---|
| Qwen | Gemma | 0,30 | 0,24–0,37 | < 0,001 |
| Qwen | Olmo | 0,25 | 0,20–0,31 | < 0,001 |
| Qwen | DeepSeek | 0,92 | 0,68–1,23 | 0,556 |
| Gemma | Olmo | 0,84 | 0,72–0,99 | 0,064 |
| Gemma | DeepSeek | 3,08 | 2,57–3,69 | < 0,001 |
| Olmo | DeepSeek | 3,65 | 2,94–4,54 | < 0,001 |

**deliberativos (VER+COR+ALT)**

| A | B | A / B | IC 95 % | p Holm |
|---|---|---|---|---|
| Qwen | Gemma | 0,38 | 0,31–0,47 | < 0,001 |
| Qwen | Olmo | 0,53 | 0,41–0,69 | < 0,001 |
| Qwen | DeepSeek | 0,88 | 0,65–1,19 | 0,410 |
| Gemma | Olmo | 1,39 | 1,04–1,84 | 0,048 |
| Gemma | DeepSeek | 2,30 | 1,84–2,88 | < 0,001 |
| Olmo | DeepSeek | 1,66 | 1,22–2,26 | 0,004 |

### Medicina con el acierto (episodios ~ log(nodos) + modelo + acierta)

| Recuento | Familia | β log(nodos) | acierta / falla | Gemma / Qwen | Olmo / Qwen | DeepSeek / Qwen |
|---|---|---|---|---|---|---|
| VERIFICATION | binomial negativa (NB2) | 0,20 (0,115) | 1,03 [0,69–1,54] (0,875) | 4,90 [3,18–7,53] (< 0,001) | 1,62 [1,10–2,37] (0,015) | 1,07 [0,71–1,64] (0,738) |
| CORRECTION | binomial negativa (NB2) | 1,20 (< 0,001) | 1,11 [0,49–2,49] (0,804) | 0,67 [0,33–1,37] (0,274) | 1,35 [0,65–2,80] (0,417) | 1,21 [0,40–3,69] (0,732) |
| ALTERNATIVE | binomial negativa (NB2) | 0,22 (0,227) | 1,28 [0,77–2,10] (0,341) | 2,01 [1,17–3,45] (0,012) | 4,72 [2,61–8,53] (< 0,001) | 1,48 [0,78–2,82] (0,230) |
| deliberativos (VER+COR+ALT) | binomial negativa (NB2) | 0,35 (0,008) | 1,13 [0,81–1,58] (0,469) | 2,65 [1,97–3,57] (< 0,001) | 2,75 [1,81–4,19] (< 0,001) | 1,33 [0,87–2,03] (0,183) |

## 2. Intervalos por bootstrap (remuestreo de problemas; percentil 2,5–97,5)

### Medianas por celda [IC 95 %]

**Medicina**

| Métrica | Qwen | Gemma | Olmo | DeepSeek |
|---|---|---|---|---|
| delib_100n | 3,28 [1,82–5,00] | 8,22 [6,70–11,69] | 5,12 [3,55–8,79] | 5,80 [1,39–9,09] |
| ver_100n | 1,05 [0,45–1,82] | 5,20 [3,45–6,35] | 1,10 [0,45–1,66] | 0,81 [0,38–2,13] |
| cor_100n | 0,00 [0,00–0,91] | 0,00 [0,00–0,52] | 0,62 [0,45–1,20] | 0,00 [0,00–0,00] |
| alt_100n | 0,91 [0,27–1,47] | 2,56 [1,30–3,16] | 3,11 [1,70–6,04] | 1,45 [0,64–4,55] |
| sup_100n | 1,82 [1,18–3,03] | 10,26 [7,30–11,90] | 6,90 [4,97–10,80] | 2,91 [1,45–6,45] |
| cobertura | 0,27 [0,10–0,38] | 0,47 [0,39–0,55] | 0,41 [0,34–0,46] | 0,34 [0,16–0,48] |
| episodios | 7,00 [5,00–8,00] | 21,00 [17,00–25,00] | 19,00 [16,00–25,00] | 7,00 [5,00–9,00] |

**Derecho**

| Métrica | Qwen | Gemma | Olmo | DeepSeek |
|---|---|---|---|---|
| delib_100n | 5,56 [3,80–6,90] | 12,86 [7,52–20,34] | 4,48 [3,69–7,22] | 5,00 [3,55–8,33] |
| ver_100n | 1,62 [0,36–3,12] | 4,87 [3,28–6,36] | 1,24 [0,46–1,47] | 2,15 [1,32–4,55] |
| cor_100n | 1,08 [0,00–2,66] | 0,95 [0,00–2,25] | 1,04 [0,00–2,55] | 0,00 [0,00–2,22] |
| alt_100n | 0,62 [0,00–1,49] | 2,86 [1,12–9,73] | 2,22 [1,49–3,18] | 0,00 [0,00–2,22] |
| sup_100n | 3,49 [2,04–5,56] | 10,53 [6,86–11,76] | 7,01 [5,02–11,11] | 4,82 [3,45–8,57] |
| cobertura | 0,38 [0,26–0,50] | 0,51 [0,46–0,67] | 0,42 [0,31–0,48] | 0,46 [0,38–0,56] |
| episodios | 7,00 [5,00–9,00] | 20,00 [17,00–23,00] | 17,00 [14,00–22,00] | 6,00 [5,00–9,00] |

### δ de Cliff entre modelos [IC 95 %] (positivo: A mayor)

**Medicina**

| Métrica | Qwen − Gemma | Qwen − Olmo | Qwen − DeepSeek | Gemma − Olmo | Gemma − DeepSeek | Olmo − DeepSeek |
|---|---|---|---|---|---|---|
| delib_100n | -0,63 [-0,84–-0,40] | -0,35 [-0,60–-0,07] | -0,11 [-0,44–0,21] | 0,36 [0,06–0,64] | 0,32 [0,08–0,58] | 0,10 [-0,21–0,44] |
| ver_100n | -0,69 [-0,90–-0,45] | 0,02 [-0,27–0,31] | -0,00 [-0,32–0,29] | 0,83 [0,65–0,96] | 0,69 [0,46–0,89] | -0,05 [-0,33–0,28] |
| cor_100n | 0,19 [-0,10–0,47] | -0,18 [-0,46–0,11] | 0,19 [-0,12–0,47] | -0,41 [-0,68–-0,13] | 0,02 [-0,24–0,28] | 0,43 [0,14–0,71] |
| alt_100n | -0,43 [-0,71–-0,12] | -0,51 [-0,72–-0,32] | -0,26 [-0,54–-0,00] | -0,14 [-0,45–0,18] | 0,12 [-0,16–0,40] | 0,17 [-0,16–0,50] |

**Derecho**

| Métrica | Qwen − Gemma | Qwen − Olmo | Qwen − DeepSeek | Gemma − Olmo | Gemma − DeepSeek | Olmo − DeepSeek |
|---|---|---|---|---|---|---|
| delib_100n | -0,58 [-0,81–-0,31] | 0,01 [-0,33–0,34] | -0,04 [-0,39–0,34] | 0,55 [0,26–0,80] | 0,48 [0,24–0,71] | -0,03 [-0,34–0,26] |
| ver_100n | -0,59 [-0,80–-0,34] | 0,25 [-0,08–0,57] | -0,20 [-0,46–0,08] | 0,88 [0,77–0,98] | 0,46 [0,21–0,70] | -0,49 [-0,75–-0,20] |
| cor_100n | 0,05 [-0,24–0,33] | -0,05 [-0,35–0,25] | 0,15 [-0,11–0,42] | -0,11 [-0,35–0,14] | 0,09 [-0,14–0,34] | 0,20 [0,00–0,41] |
| alt_100n | -0,47 [-0,69–-0,21] | -0,43 [-0,70–-0,12] | 0,02 [-0,27–0,33] | 0,16 [-0,15–0,47] | 0,43 [0,12–0,71] | 0,36 [0,02–0,66] |

## 3. Fiabilidad por cadena entre corridas (20 cadenas × 3 corridas)

ICC(1): fiabilidad de una sola corrida; ICC(1,3): de la media de tres. Atenuación: factor por el que una correlación observada con la métrica de una corrida subestima la verdadera (√ICC).

| Métrica | ICC(1) [IC 95 %] | ICC(1,3) | atenuación |
|---|---|---|---|
| nodos | 1,00 [0,99–1,00] | 1,00 | 1,00 |
| episodios | 0,84 [0,55–0,93] | 0,94 | 0,92 |
| delib_100n | 0,46 [0,13–0,85] | 0,72 | 0,68 |
| ver_100n | 0,51 [0,19–0,74] | 0,75 | 0,71 |
| cor_100n | 0,62 [-0,06–0,88] | 0,83 | 0,79 |
| alt_100n | 0,38 [0,11–0,85] | 0,65 | 0,62 |
| cobertura | 0,81 [0,69–0,88] | 0,93 | 0,90 |

## 4. Sensibilidad a la longitud

### Pares del mismo problema con cociente de nodos ≤ 1.5

Wilcoxon sobre los pares conservados; δ de Cliff sobre esas cadenas.

**Medicina**

| Métrica | A | B | pares | A > B | B > A | mediana A | mediana B | δ | p |
|---|---|---|---|---|---|---|---|---|---|
| delib_100n | Qwen | Gemma | 9 | 1 | 8 | 3,30 | 9,32 | -0,70 | 0,008 |
| delib_100n | Qwen | Olmo | 11 | 1 | 10 | 3,28 | 6,35 | -0,64 | 0,005 |
| delib_100n | Qwen | DeepSeek | 7 | 2 | 5 | 1,82 | 6,37 | -0,12 | 0,297 |
| delib_100n | Gemma | Olmo | 10 | 8 | 2 | 7,20 | 5,41 | 0,32 | 0,131 |
| delib_100n | Gemma | DeepSeek | 10 | 8 | 2 | 10,12 | 4,38 | 0,38 | 0,020 |
| delib_100n | Olmo | DeepSeek | 8 | 6 | 2 | 4,89 | 1,83 | 0,56 | 0,148 |
| ver_100n | Qwen | Gemma | 9 | 0 | 9 | 1,25 | 6,09 | -0,89 | 0,004 |
| ver_100n | Qwen | Olmo | 11 | 2 | 8 | 0,78 | 1,18 | -0,31 | 0,037 |
| ver_100n | Qwen | DeepSeek | 7 | 3 | 4 | 1,25 | 1,82 | -0,10 | 0,938 |
| ver_100n | Gemma | Olmo | 10 | 8 | 2 | 3,91 | 1,81 | 0,59 | 0,020 |
| ver_100n | Gemma | DeepSeek | 10 | 9 | 0 | 3,99 | 1,27 | 0,65 | 0,004 |
| ver_100n | Olmo | DeepSeek | 8 | 6 | 2 | 0,93 | 0,47 | 0,42 | 0,148 |
| cor_100n | Qwen | Gemma | 9 | 2 | 2 | 0,00 | 0,00 | 0,00 | — |
| cor_100n | Qwen | Olmo | 11 | 2 | 7 | 0,00 | 0,65 | -0,28 | 0,164 |
| cor_100n | Qwen | DeepSeek | 7 | 1 | 3 | 0,00 | 0,00 | -0,14 | — |
| cor_100n | Gemma | Olmo | 10 | 3 | 4 | 0,00 | 0,51 | -0,26 | 0,688 |
| cor_100n | Gemma | DeepSeek | 10 | 1 | 5 | 0,00 | 0,20 | -0,25 | 0,094 |
| cor_100n | Olmo | DeepSeek | 8 | 4 | 3 | 0,59 | 0,20 | 0,16 | 0,812 |
| alt_100n | Qwen | Gemma | 9 | 3 | 6 | 1,25 | 2,36 | -0,09 | 0,496 |
| alt_100n | Qwen | Olmo | 11 | 1 | 10 | 1,25 | 3,25 | -0,52 | 0,005 |
| alt_100n | Qwen | DeepSeek | 7 | 2 | 4 | 0,00 | 0,99 | -0,43 | 0,156 |
| alt_100n | Gemma | Olmo | 10 | 5 | 4 | 2,46 | 2,83 | -0,12 | 1,000 |
| alt_100n | Gemma | DeepSeek | 10 | 7 | 2 | 3,55 | 1,53 | 0,37 | 0,074 |
| alt_100n | Olmo | DeepSeek | 8 | 6 | 2 | 2,41 | 0,92 | 0,56 | 0,109 |

**Derecho**

| Métrica | A | B | pares | A > B | B > A | mediana A | mediana B | δ | p |
|---|---|---|---|---|---|---|---|---|---|
| delib_100n | Qwen | Gemma | 5 | 3 | 2 | 7,61 | 6,36 | 0,20 | — |
| delib_100n | Qwen | Olmo | 7 | 3 | 4 | 4,50 | 9,38 | -0,55 | 0,219 |
| delib_100n | Qwen | DeepSeek | 9 | 5 | 4 | 6,25 | 4,35 | 0,27 | 0,652 |
| delib_100n | Gemma | Olmo | 9 | 6 | 3 | 15,93 | 5,97 | 0,41 | 0,055 |
| delib_100n | Gemma | DeepSeek | 8 | 7 | 1 | 15,41 | 7,18 | 0,56 | 0,023 |
| delib_100n | Olmo | DeepSeek | 5 | 4 | 1 | 8,00 | 3,55 | 0,60 | — |
| ver_100n | Qwen | Gemma | 5 | 3 | 2 | 2,74 | 3,37 | 0,04 | — |
| ver_100n | Qwen | Olmo | 7 | 5 | 2 | 1,60 | 0,78 | 0,43 | 0,375 |
| ver_100n | Qwen | DeepSeek | 9 | 4 | 5 | 2,04 | 2,22 | -0,07 | 0,652 |
| ver_100n | Gemma | Olmo | 9 | 9 | 0 | 5,17 | 1,47 | 0,75 | 0,004 |
| ver_100n | Gemma | DeepSeek | 8 | 8 | 0 | 5,55 | 1,68 | 0,75 | 0,008 |
| ver_100n | Olmo | DeepSeek | 5 | 1 | 4 | 1,82 | 2,15 | -0,60 | — |
| cor_100n | Qwen | Gemma | 5 | 2 | 2 | 1,27 | 0,00 | 0,12 | — |
| cor_100n | Qwen | Olmo | 7 | 5 | 2 | 2,66 | 1,82 | 0,27 | 0,375 |
| cor_100n | Qwen | DeepSeek | 9 | 5 | 2 | 1,02 | 0,00 | 0,26 | 0,469 |
| cor_100n | Gemma | Olmo | 9 | 1 | 5 | 0,00 | 1,04 | -0,26 | 0,438 |
| cor_100n | Gemma | DeepSeek | 8 | 4 | 2 | 2,59 | 1,71 | 0,25 | 0,438 |
| cor_100n | Olmo | DeepSeek | 5 | 4 | 0 | 1,82 | 0,00 | 0,48 | — |
| alt_100n | Qwen | Gemma | 5 | 3 | 2 | 1,16 | 1,11 | 0,04 | — |
| alt_100n | Qwen | Olmo | 7 | 0 | 7 | 1,06 | 6,06 | -0,92 | 0,016 |
| alt_100n | Qwen | DeepSeek | 9 | 4 | 2 | 0,62 | 0,00 | 0,16 | 0,688 |
| alt_100n | Gemma | Olmo | 9 | 6 | 2 | 8,57 | 2,67 | 0,42 | 0,078 |
| alt_100n | Gemma | DeepSeek | 8 | 4 | 3 | 2,49 | 2,27 | 0,09 | 0,688 |
| alt_100n | Olmo | DeepSeek | 5 | 4 | 0 | 2,30 | 0,00 | 0,76 | — |

### Regresión restringida a la banda común de 50–200 nodos (n = 115; por celda: Qwen|medicina 17, Qwen|derecho 12, Gemma|medicina 19, Gemma|derecho 20, Olmo|medicina 14, Olmo|derecho 15, DeepSeek|medicina 7, DeepSeek|derecho 11)

| Recuento | Familia | β log(nodos) | Gemma / Qwen | Olmo / Qwen | DeepSeek / Qwen | derecho / medicina |
|---|---|---|---|---|---|---|
| VERIFICATION | binomial negativa (NB2) | 0,54 (0,001) | 4,11 [2,61–6,49] (< 0,001) | 1,26 [0,79–2,01] (0,339) | 1,33 [0,85–2,09] (0,214) | 0,82 [0,63–1,07] (0,152) |
| CORRECTION | binomial negativa (NB2) | 1,07 (< 0,001) | 0,71 [0,39–1,29] (0,258) | 0,98 [0,58–1,67] (0,948) | 1,17 [0,53–2,56] (0,696) | 1,97 [1,22–3,18] (0,005) |
| ALTERNATIVE | binomial negativa (NB2) | 0,26 (0,285) | 2,09 [1,33–3,28] (0,001) | 1,98 [1,26–3,10] (0,003) | 1,17 [0,66–2,07] (0,596) | 1,00 [0,66–1,50] (0,995) |
| deliberativos (VER+COR+ALT) | binomial negativa (NB2) | 0,56 (< 0,001) | 2,30 [1,71–3,09] (< 0,001) | 1,44 [1,07–1,94] (0,017) | 1,18 [0,80–1,76] (0,407) | 1,06 [0,84–1,33] (0,641) |

## 5. Corrección múltiple global sobre las 156 comparaciones pareadas de analizar.py

Significativas sin corregir: 73; con Benjamini-Hochberg (q < 0,05): 61; con Holm global: 28.

Las que sobreviven a Benjamini-Hochberg:

| Dominio | Métrica | A | B | δ | p | q BH |
|---|---|---|---|---|---|---|
| medicina | episodios | Qwen | Gemma | -0,90 | < 0,001 | 0,000 |
| medicina | episodios | Qwen | Olmo | -0,95 | < 0,001 | 0,000 |
| medicina | episodios | Olmo | DeepSeek | 0,92 | < 0,001 | 0,000 |
| derecho | palabras | Olmo | DeepSeek | 0,70 | < 0,001 | 0,000 |
| derecho | episodios | Qwen | Olmo | -0,82 | < 0,001 | 0,000 |
| derecho | episodios | Gemma | DeepSeek | 0,94 | < 0,001 | 0,000 |
| derecho | episodios | Olmo | DeepSeek | 0,89 | < 0,001 | 0,000 |
| derecho | ver_100n | Gemma | Olmo | 0,88 | < 0,001 | 0,000 |
| medicina | episodios | Gemma | DeepSeek | 0,85 | < 0,001 | 0,000 |
| derecho | episodios | Qwen | Gemma | -0,90 | < 0,001 | 0,000 |
| medicina | ver_100n | Gemma | Olmo | 0,83 | < 0,001 | 0,000 |
| derecho | nodos | Olmo | DeepSeek | 0,62 | < 0,001 | 0,000 |
| derecho | componentes_100n | Qwen | Gemma | -0,71 | < 0,001 | 0,000 |
| medicina | alt_100n | Qwen | Olmo | -0,51 | < 0,001 | 0,000 |
| medicina | componentes_100n | Qwen | Olmo | -0,72 | < 0,001 | 0,001 |
| derecho | componentes_100n | Gemma | DeepSeek | 0,72 | < 0,001 | 0,001 |
| derecho | delib_100n | Qwen | Gemma | -0,58 | < 0,001 | 0,001 |
| derecho | delib_100n | Gemma | DeepSeek | 0,48 | < 0,001 | 0,001 |
| medicina | ver_100n | Gemma | DeepSeek | 0,69 | < 0,001 | 0,001 |
| derecho | sup_100n | Gemma | DeepSeek | 0,58 | < 0,001 | 0,001 |
| medicina | sup_100n | Qwen | Olmo | -0,67 | < 0,001 | 0,001 |
| medicina | componentes_100n | Qwen | Gemma | -0,68 | < 0,001 | 0,001 |
| medicina | delib_100n | Qwen | Gemma | -0,63 | < 0,001 | 0,001 |
| derecho | ver_100n | Qwen | Gemma | -0,59 | < 0,001 | 0,002 |
| medicina | cobertura | Qwen | Gemma | -0,55 | < 0,001 | 0,002 |
| medicina | componentes_100n | Olmo | DeepSeek | 0,59 | < 0,001 | 0,002 |
| medicina | ver_100n | Qwen | Gemma | -0,69 | < 0,001 | 0,002 |
| derecho | sup_100n | Qwen | Gemma | -0,62 | < 0,001 | 0,002 |
| medicina | sup_100n | Qwen | Gemma | -0,62 | < 0,001 | 0,003 |
| medicina | componentes_100n | Gemma | DeepSeek | 0,57 | < 0,001 | 0,004 |
| derecho | palabras | Gemma | DeepSeek | 0,52 | < 0,001 | 0,004 |
| derecho | cobertura_delib | Gemma | Olmo | 0,58 | < 0,001 | 0,004 |
| derecho | componentes_100n | Qwen | Olmo | -0,54 | < 0,001 | 0,004 |
| derecho | alt_100n | Qwen | Gemma | -0,47 | 0,001 | 0,005 |
| medicina | cobertura_delib | Qwen | Gemma | -0,54 | 0,001 | 0,006 |
| derecho | delib_100n | Gemma | Olmo | 0,55 | 0,001 | 0,006 |
| derecho | componentes_100n | Olmo | DeepSeek | 0,62 | 0,001 | 0,006 |
| derecho | cobertura_delib | Qwen | Gemma | -0,54 | 0,002 | 0,007 |
| derecho | ver_100n | Olmo | DeepSeek | -0,49 | 0,004 | 0,016 |
| derecho | palabras | Qwen | Olmo | -0,45 | 0,004 | 0,016 |
| derecho | sup_100n | Olmo | DeepSeek | 0,33 | 0,004 | 0,016 |
| derecho | cobertura | Qwen | Gemma | -0,48 | 0,004 | 0,016 |
| medicina | sup_100n | Olmo | DeepSeek | 0,48 | 0,005 | 0,016 |
| derecho | nodos | Gemma | Olmo | -0,38 | 0,005 | 0,016 |
| derecho | ver_100n | Gemma | DeepSeek | 0,46 | 0,005 | 0,016 |
| medicina | palabras | Qwen | Olmo | -0,42 | 0,005 | 0,016 |
| medicina | sup_100n | Gemma | DeepSeek | 0,49 | 0,005 | 0,016 |
| medicina | cobertura_delib | Gemma | DeepSeek | 0,44 | 0,005 | 0,016 |
| medicina | nodos | Gemma | Olmo | -0,41 | 0,006 | 0,018 |
| derecho | alt_100n | Gemma | DeepSeek | 0,43 | 0,007 | 0,021 |
| derecho | cobertura | Gemma | Olmo | 0,46 | 0,008 | 0,025 |
| derecho | nodos | Qwen | Olmo | -0,36 | 0,010 | 0,028 |
| derecho | sup_100n | Qwen | Olmo | -0,45 | 0,010 | 0,028 |
| derecho | prop_delib | Qwen | Olmo | 0,40 | 0,011 | 0,030 |
| medicina | cobertura | Qwen | Olmo | -0,37 | 0,014 | 0,038 |
| derecho | palabras | Gemma | Olmo | -0,28 | 0,014 | 0,038 |
| medicina | alt_100n | Qwen | Gemma | -0,43 | 0,014 | 0,038 |
| medicina | palabras | Olmo | DeepSeek | 0,37 | 0,015 | 0,040 |
| medicina | palabras | Gemma | Olmo | -0,30 | 0,017 | 0,046 |
| medicina | alt_100n | Qwen | DeepSeek | -0,26 | 0,018 | 0,046 |
| medicina | delib_100n | Gemma | Olmo | 0,36 | 0,019 | 0,048 |
