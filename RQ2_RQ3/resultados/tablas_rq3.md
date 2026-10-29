# RQ3 — tablas (generadas por rq3.py)

## 0. Fiabilidad de los rasgos (ICC(1), 20 cadenas × 3 corridas) y rasgos funcionales primarios (ICC ≥ 0,7)

| Rasgo | Capa | ICC(1) | primario |
|---|---|---|---|
| cuota_nodo_METHOD | estructural | 0,88 |  |
| cuota_des_CORRECTION_RECOVERED | funcional | 0,87 | sí |
| nodos_100pal | estructural | 0,87 |  |
| cuota_ep_CORRECTION | funcional | 0,87 | sí |
| cuota_nodo_CHECK | estructural | 0,82 |  |
| cobertura | funcional | 0,81 | sí |
| cobertura_PROGRESSION | funcional | 0,80 | sí |
| cuota_des_ALTERNATIVE_ADOPTED | funcional | 0,79 | sí |
| cuota_nodo_EVID | estructural | 0,78 |  |
| aristas_por_nodo | estructural | 0,71 |  |
| solapados_por_ep | funcional | 0,70 | sí |
| cuota_nodo_PROP | estructural | 0,70 |  |
| cobertura_VERIFICATION | funcional | 0,70 |  |
| episodios_100n | funcional | 0,70 |  |
| cobertura_SUPPORT | funcional | 0,69 |  |
| cuota_arista_SUPPORTS | estructural | 0,56 |  |
| cuota_arista_ALTERNATIVE_TO | estructural | 0,53 |  |
| cuota_nodo_OUT | estructural | 0,51 |  |
| cuota_nodo_ALT | estructural | 0,50 |  |
| cobertura_CORRECTION | funcional | 0,50 |  |
| no_contiguos_frac | funcional | 0,48 |  |
| cuota_rel_CONTINUES | funcional | 0,48 |  |
| cuota_arista_LEADS_TO | estructural | 0,45 |  |
| cobertura_ALTERNATIVE | funcional | 0,45 |  |
| cuota_des_ALTERNATIVE_REJECTED | funcional | 0,42 |  |
| cuota_ep_PROGRESSION | funcional | 0,42 |  |
| lineas_por_ep | funcional | 0,41 |  |
| cuota_ep_ALTERNATIVE | funcional | 0,39 |  |
| nodos_ambiguos_frac | estructural | 0,37 |  |
| cuota_arista_CHECKS | estructural | 0,37 |  |
| cuota_ep_SUPPORT | funcional | 0,33 |  |
| cuota_des_ALTERNATIVE_UNRESOLVED | funcional | 0,31 |  |
| cuota_des_VERIFICATION_CONFIRMED | funcional | 0,30 |  |
| componentes_por_ep | funcional | 0,29 |  |
| diversidad_funcional | funcional | 0,28 |  |
| cuota_nodo_REVISE | estructural | 0,27 |  |
| cuota_arista_REVISES | estructural | 0,25 |  |
| cuota_rel_SUPPORTS_EPISODE | funcional | 0,25 |  |
| cuota_ep_VERIFICATION | funcional | 0,25 |  |
| anidados_por_ep | funcional | 0,24 |  |
| cuota_des_CORRECTION_PARTIAL | funcional | 0,24 |  |
| pos_delib | funcional | 0,23 |  |
| cuota_des_VERIFICATION_REFUTED | funcional | 0,21 |  |
| cuota_des_PROGRESSION_REACHES_OUTPUT | funcional | 0,21 |  |
| cuota_rel_REPLACES | funcional | 0,18 |  |
| ruta_episodios_media | funcional | 0,16 |  |
| cuota_rel_EVALUATES | funcional | 0,14 |  |
| relaciones_por_ep | funcional | 0,09 |  |
| abiertos_frac | funcional | 0,05 |  |
| cuota_rel_COMPETES_WITH | funcional | 0,05 |  |
| sin_delib | funcional | -0,00 |  |
| cuota_des_VERIFICATION_INCONCLUSIVE | funcional | -0,01 |  |
| cuota_rel_EMBEDS | funcional | -0,26 |  |

## Modelo (n = 200; clases: Qwen, Gemma, Olmo, DeepSeek; azar 0,25; base: log_palabras, log_palabras2, acierto, dominio_derecho)

| Conjunto de rasgos | rasgos | exactitud [IC 95 %] | ganancia vs base [IC 95 %] | F1 macro | log-pérdida [IC] | McNemar vs base (solo base / solo conjunto; p) | Wilcoxon log-pérdida por problema (mejora/empeora; p) |
|---|---|---|---|---|---|---|---|
| base | 4 | 0,490 [0,440–0,540] | — | 0,444 | 1,223 [1,166–1,280] | — | — |
| base+funcional_primario | 10 | 0,550 [0,500–0,600] | 0,060 [0,010–0,115] | 0,549 | 1,096 [1,013–1,188] | 23 / 35; 0,148 | 35/15; 0,002 |
| base+funcional | 42 | 0,670 [0,620–0,720] | 0,180 [0,110–0,245] | 0,669 | 0,781 [0,700–0,862] | 18 / 54; < 0,001 | 48/2; < 0,001 |
| base+estructural | 19 | 0,615 [0,555–0,675] | 0,125 [0,060–0,190] | 0,608 | 0,855 [0,777–0,942] | 15 / 40; 0,001 | 46/4; < 0,001 |
| base+todo | 57 | 0,720 [0,665–0,775] | 0,230 [0,160–0,305] | 0,716 | 0,659 [0,575–0,744] | 9 / 55; < 0,001 | 49/1; < 0,001 |
| funcional_solo | 38 | 0,625 [0,570–0,680] | 0,135 [0,055–0,215] | 0,623 | 0,902 [0,806–1,005] | 28 / 55; 0,004 | 39/11; < 0,001 |
| todo_solo | 53 | 0,715 [0,665–0,770] | 0,225 [0,150–0,295] | 0,715 | 0,676 [0,593–0,763] | 12 / 57; < 0,001 | 49/1; < 0,001 |

Sensibilidad por clase (fracción de cadenas de cada clase reconocidas):

| Conjunto | Qwen | Gemma | Olmo | DeepSeek |
|---|---|---|---|---|
| base | 0,06 | 0,72 | 0,70 | 0,48 |
| base+funcional_primario | 0,48 | 0,58 | 0,60 | 0,54 |
| base+funcional | 0,58 | 0,78 | 0,88 | 0,44 |
| base+estructural | 0,42 | 0,78 | 0,78 | 0,48 |
| base+todo | 0,50 | 0,88 | 0,94 | 0,56 |
| funcional_solo | 0,56 | 0,68 | 0,82 | 0,44 |
| todo_solo | 0,54 | 0,84 | 0,92 | 0,56 |

Bosque aleatorio (500 árboles, misma validación):

| Conjunto | exactitud [IC 95 %] | F1 macro | log-pérdida |
|---|---|---|---|
| base | 0,360 [0,305–0,415] | 0,354 | 1,506 |
| base+todo | 0,745 [0,685–0,805] | 0,743 | 0,791 |

Ganancia del bosque base+todo sobre base: 0,385 [0,310–0,460]; McNemar 8 / 85, p = < 0,001.

Matriz de confusión (filas: clase real; columnas: predicha) para base y base+todo:

*base*

| real \ predicha | Qwen | Gemma | Olmo | DeepSeek |
|---|---|---|---|---|
| Qwen | 3 | 17 | 14 | 16 |
| Gemma | 4 | 36 | 10 | 0 |
| Olmo | 3 | 10 | 35 | 2 |
| DeepSeek | 2 | 16 | 8 | 24 |

*base+todo*

| real \ predicha | Qwen | Gemma | Olmo | DeepSeek |
|---|---|---|---|---|
| Qwen | 25 | 2 | 2 | 21 |
| Gemma | 2 | 44 | 3 | 1 |
| Olmo | 0 | 2 | 47 | 1 |
| DeepSeek | 18 | 2 | 2 | 28 |

### Permutaciones (300)

- Ganancia de base+todo sobre base: observada 0,230; nula (perfil barajado dentro de cada problema) media -0,131, percentil 95 -0,080; p = 0,003.
- Exactitud de todo_solo: observada 0,715; nula (etiquetas barajadas dentro de cada problema) media 0,250, percentil 95 0,305; p = 0,003.

### Ablación por grupos de rasgos (regresión logística, validación por problema)

| Grupo | rasgos | exactitud sin el grupo (base+todo = 0,720) | McNemar vs base+todo (p) | exactitud base + solo el grupo (base = 0,490) | McNemar vs base (p) |
|---|---|---|---|---|---|
| verificación | 6 | 0,720 | 1,000 | 0,495 | 1,000 |
| alternativas | 6 | 0,710 | 0,688 | 0,535 | 0,200 |
| correcciones | 5 | 0,715 | 1,000 | 0,490 | 1,000 |
| apoyo | 3 | 0,710 | 0,688 | 0,550 | 0,155 |
| progresión y organización | 18 | 0,690 | 0,263 | 0,600 | 0,009 |
| estructura | 15 | 0,670 | 0,132 | 0,615 | 0,001 |

### Coeficientes estandarizados (modelo base+todo ajustado a las 200 cadenas, C = 0.1); los ocho mayores en valor absoluto por clase

- **Qwen:** cuota_nodo_METHOD (0,35); cobertura_SUPPORT (-0,33); cuota_des_PROGRESSION_REACHES_OUTPUT (0,28); cuota_des_ALTERNATIVE_UNRESOLVED (-0,25); cuota_nodo_ALT (-0,24); nodos_100pal (0,23); cuota_des_CORRECTION_RECOVERED (0,23); episodios_100n (-0,23)
- **Gemma:** episodios_100n (0,41); cuota_nodo_METHOD (-0,40); cuota_nodo_OUT (-0,36); nodos_100pal (-0,36); no_contiguos_frac (-0,34); cuota_nodo_CHECK (0,28); cuota_rel_EVALUATES (0,24); aristas_por_nodo (0,23)
- **Olmo:** cuota_nodo_OUT (0,44); cuota_ep_SUPPORT (0,38); cuota_ep_VERIFICATION (-0,38); cuota_des_VERIFICATION_CONFIRMED (-0,34); acierto (-0,33); cuota_arista_ALTERNATIVE_TO (0,29); nodos_100pal (-0,27); cuota_rel_SUPPORTS_EPISODE (0,26)
- **DeepSeek:** nodos_100pal (0,40); no_contiguos_frac (0,32); diversidad_funcional (-0,28); solapados_por_ep (0,27); cuota_nodo_OUT (-0,26); cobertura_CORRECTION (-0,25); cuota_nodo_METHOD (0,21); cuota_ep_SUPPORT (-0,20)

## Dominio (n = 200; clases: medicina, derecho; azar 0,50; base: log_palabras, log_palabras2, acierto, modelo_Gemma, modelo_Olmo, modelo_DeepSeek)

| Conjunto de rasgos | rasgos | exactitud [IC 95 %] | ganancia vs base [IC 95 %] | F1 macro | log-pérdida [IC] | McNemar vs base (solo base / solo conjunto; p) | Wilcoxon log-pérdida por problema (mejora/empeora; p) | AUC [IC] |
|---|---|---|---|---|---|---|---|---|
| base | 6 | 0,705 [0,600–0,805] | — | 0,691 | 0,597 [0,513–0,685] | — | — | 0,632 [0,509–0,753] |
| base+funcional_primario | 12 | 0,775 [0,715–0,835] | 0,070 [-0,030–0,165] | 0,775 | 0,481 [0,410–0,555] | 20 / 34; 0,076 | 37/13; 0,001 | 0,845 [0,782–0,900] |
| base+funcional | 44 | 0,730 [0,665–0,790] | 0,025 [-0,065–0,110] | 0,730 | 0,576 [0,498–0,662] | 25 / 30; 0,590 | 28/22; 0,438 | 0,783 [0,707–0,851] |
| base+estructural | 21 | 0,790 [0,710–0,865] | 0,085 [-0,005–0,180] | 0,790 | 0,484 [0,371–0,617] | 13 / 30; 0,014 | 34/16; 0,002 | 0,852 [0,770–0,922] |
| base+todo | 59 | 0,780 [0,715–0,840] | 0,075 [-0,025–0,185] | 0,780 | 0,517 [0,439–0,607] | 22 / 37; 0,067 | 31/19; 0,032 | 0,850 [0,780–0,912] |
| funcional_solo | 38 | 0,685 [0,620–0,755] | -0,020 [-0,145–0,105] | 0,685 | 0,608 [0,561–0,654] | 44 / 40; 0,744 | 26/24; 0,848 | 0,724 [0,649–0,796] |
| todo_solo | 53 | 0,755 [0,695–0,810] | 0,050 [-0,065–0,170] | 0,755 | 0,519 [0,465–0,575] | 34 / 44; 0,308 | 28/22; 0,166 | 0,829 [0,763–0,885] |

Sensibilidad por clase (fracción de cadenas de cada clase reconocidas):

| Conjunto | medicina | derecho |
|---|---|---|
| base | 0,49 | 0,92 |
| base+funcional_primario | 0,78 | 0,77 |
| base+funcional | 0,69 | 0,77 |
| base+estructural | 0,77 | 0,81 |
| base+todo | 0,79 | 0,77 |
| funcional_solo | 0,70 | 0,67 |
| todo_solo | 0,78 | 0,73 |

Bosque aleatorio (500 árboles, misma validación):

| Conjunto | exactitud [IC 95 %] | F1 macro | log-pérdida |
|---|---|---|---|
| base | 0,630 [0,550–0,705] | 0,630 | 0,706 |
| base+todo | 0,775 [0,720–0,825] | 0,775 | 0,520 |

Ganancia del bosque base+todo sobre base: 0,145 [0,065–0,225]; McNemar 19 / 48, p = < 0,001.

Matriz de confusión (filas: clase real; columnas: predicha) para base y base+todo:

*base*

| real \ predicha | medicina | derecho |
|---|---|---|
| medicina | 49 | 51 |
| derecho | 8 | 92 |

*base+todo*

| real \ predicha | medicina | derecho |
|---|---|---|
| medicina | 79 | 21 |
| derecho | 23 | 77 |

## Acierto (n = 100; clases: falla, acierta; azar 0,50; base: log_palabras, log_palabras2, modelo_Gemma, modelo_Olmo, modelo_DeepSeek)

| Conjunto de rasgos | rasgos | exactitud [IC 95 %] | ganancia vs base [IC 95 %] | F1 macro | log-pérdida [IC] | McNemar vs base (solo base / solo conjunto; p) | Wilcoxon log-pérdida por problema (mejora/empeora; p) | AUC [IC] |
|---|---|---|---|---|---|---|---|---|
| base | 5 | 0,650 [0,550–0,750] | — | 0,646 | 0,636 [0,547–0,749] | — | — | 0,729 [0,619–0,837] |
| base+funcional_primario | 11 | 0,640 [0,540–0,740] | -0,010 [-0,070–0,050] | 0,639 | 0,666 [0,604–0,737] | 5 / 4; 1,000 | 6/19; 0,007 | 0,659 [0,550–0,786] |
| base+funcional | 43 | 0,580 [0,480–0,670] | -0,070 [-0,150–0,000] | 0,577 | 0,720 [0,635–0,824] | 13 / 6; 0,167 | 6/19; 0,002 | 0,582 [0,464–0,702] |
| base+estructural | 20 | 0,560 [0,460–0,660] | -0,090 [-0,150–-0,030] | 0,560 | 0,688 [0,628–0,757] | 10 / 1; 0,012 | 5/20; < 0,001 | 0,604 [0,481–0,744] |
| base+todo | 58 | 0,540 [0,430–0,650] | -0,110 [-0,190–-0,030] | 0,537 | 0,744 [0,651–0,856] | 15 / 4; 0,019 | 4/21; < 0,001 | 0,553 [0,431–0,683] |
| funcional_solo | 38 | 0,520 [0,420–0,620] | -0,130 [-0,240–-0,030] | 0,518 | 0,748 [0,677–0,834] | 23 / 10; 0,035 | 4/21; 0,002 | 0,499 [0,385–0,610] |
| todo_solo | 53 | 0,490 [0,390–0,590] | -0,160 [-0,250–-0,070] | 0,490 | 0,775 [0,695–0,872] | 23 / 7; 0,005 | 3/22; < 0,001 | 0,466 [0,349–0,590] |

Sensibilidad por clase (fracción de cadenas de cada clase reconocidas):

| Conjunto | falla | acierta |
|---|---|---|
| base | 0,55 | 0,75 |
| base+funcional_primario | 0,59 | 0,69 |
| base+funcional | 0,51 | 0,65 |
| base+estructural | 0,55 | 0,57 |
| base+todo | 0,47 | 0,61 |
| funcional_solo | 0,47 | 0,57 |
| todo_solo | 0,47 | 0,51 |

Bosque aleatorio (500 árboles, misma validación):

| Conjunto | exactitud [IC 95 %] | F1 macro | log-pérdida |
|---|---|---|---|
| base | 0,540 [0,430–0,650] | 0,539 | 0,707 |
| base+todo | 0,570 [0,470–0,670] | 0,570 | 0,713 |

Ganancia del bosque base+todo sobre base: 0,030 [-0,080–0,150]; McNemar 10 / 13, p = 0,678.

Matriz de confusión (filas: clase real; columnas: predicha) para base y base+todo:

*base*

| real \ predicha | falla | acierta |
|---|---|---|
| falla | 27 | 22 |
| acierta | 13 | 38 |

*base+todo*

| real \ predicha | falla | acierta |
|---|---|---|
| falla | 23 | 26 |
| acierta | 20 | 31 |
