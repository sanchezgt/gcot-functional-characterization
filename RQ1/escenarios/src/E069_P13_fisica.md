---
id: E069
plantilla: P13
dominio: fisica
estilo: terso
respuesta: 6
enunciado: A 6-ohm resistor and a 3-ohm resistor are connected in parallel across a 12-volt battery. What is the total current supplied by the battery, in amperes?
---
## PROGRESSION:ADVANCES
Need total current: 6 ohm and 3 ohm in parallel across 12 V.

## ALTERNATIVE:REJECTED | explicita
Alternatively: compute each branch current and add them. Workable, but the equivalent resistance gives the total in one step. Not using the branch sum.

## ERROR | id=e1
Equivalent resistance: 6 + 3 = 9 ohm. Current: 12 / 9 = 1.33 A.

## VERIFICATION:REFUTED | explicita
Check: 9 ohm is larger than either resistor, but a parallel combination always has less resistance than its smallest branch. Adding the resistances is the series rule. Wrong.

## CORRECTION:RECOVERED | explicita | id=c1
Correcting it with the parallel rule: 1/R = 1/6 + 1/3 = 1/6 + 2/6 = 3/6, so R = 2 ohm, not 9. Current: 12 / 2 = 6 A.

## VERIFICATION:CONFIRMED | explicita | objetivo=c1
Check by branches: 12 / 6 = 2 A and 12 / 3 = 4 A; 2 + 4 = 6 A. Confirmed.

## PROGRESSION:REACHES_OUTPUT
Total current: 6 A. Answer: 6.
