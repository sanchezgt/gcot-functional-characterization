---
id: E071
plantilla: P14
dominio: medicina
estilo: deepseek
respuesta: 4.5
enunciado: A 4-year-old child weighing 16 kg is prescribed amoxicillin at 45 mg/kg/day, divided into doses every 12 hours. The oral suspension contains 400 mg per 5 mL. How many milliliters should be given per dose?
---
## PROGRESSION:ADVANCES
We are given: "A 4-year-old child weighing 16 kg is prescribed amoxicillin at 45 mg/kg/day, divided every 12 hours. The suspension contains 400 mg per 5 mL. How many milliliters per dose?"

We need the volume of suspension for a single dose.

## ALTERNATIVE:REJECTED | implicita
A per-dose amount could be read off a weight-band dosing table, but this prescription fixes the dose at 45 mg/kg/day, so it has to be worked out from the child's 16 kg.

## ERROR | id=e1
Daily dose: 45 x 16 = 720 mg. Per dose, divided every 12 hours: 720 / 12 = 60 mg.

## VERIFICATION:REFUTED | implicita
Every 12 hours means two doses a day, so each dose is half of the daily amount, whereas 60 mg is one twelfth of it.

## CORRECTION:RECOVERED | implicita | id=c1
Each dose is 720 / 2 = 360 mg.

## VERIFICATION:CONFIRMED | implicita | objetivo=c1
Two doses of 360 mg make 720 mg a day, which is 45 mg for each of the child's 16 kg.

## PROGRESSION:ADVANCES
The suspension has 400 mg in 5 mL, which is 80 mg per mL. Volume per dose: 360 / 80 = 4.5 mL.

## PROGRESSION:REACHES_OUTPUT
Thus final answer: 4.5 mL per dose.
