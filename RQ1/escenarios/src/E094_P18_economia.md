---
id: E094
plantilla: P18
dominio: economia
estilo: escolar
respuesta: 830
enunciado: In a given year, a country's consumption is 500, investment is 200, government purchases are 150, exports are 100, and imports are 120, all in billions of dollars. What is its GDP by the expenditure approach, in billions of dollars?
---
## PROGRESSION:ADVANCES
We are asked for GDP by the expenditure approach, given consumption 500, investment 200, government purchases 150, exports 100, and imports 120, in billions of dollars.

## ERROR | id=e1
Net exports are exports plus imports: 100 + 120 = 220.

## PROGRESSION:ADVANCES
GDP = C + I + G + NX = 500 + 200 + 150 + 220 = 1,070.

## PROGRESSION:ADVANCES
The share of consumption in GDP is then 500 / 1,070, about 46.7%.

## VERIFICATION:REFUTED | explicita | objetivo=e1
Let me go back and check the net exports step. Imports are goods produced abroad; they are already included in C, I, and G, and they must be subtracted so that GDP counts only domestic production. Net exports are exports minus imports, so the 220 used above is wrong, and so is everything built on it.

## CORRECTION:RECOVERED | explicita | objetivo=e1
Correcting it: NX = 100 - 120 = -20, so GDP = 500 + 200 + 150 - 20 = 830, and the share of consumption is 500 / 830, about 60.2%.

## PROGRESSION:REACHES_OUTPUT
GDP is 830 billion dollars.

Answer: 830.
