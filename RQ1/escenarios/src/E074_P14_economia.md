---
id: E074
plantilla: P14
dominio: economia
estilo: terso
respuesta: -4
enunciado: A stock's price rises 20% on Monday and falls 20% on Tuesday. What is its net percentage change over the two days?
---
## PROGRESSION:ADVANCES
Need the net % change after +20% on Monday and -20% on Tuesday.

## ALTERNATIVE:REJECTED | implicita
A general starting price P could be carried through both days, but symbols add nothing when only a percentage is asked; a starting price of 100 keeps the arithmetic in plain numbers.

## PROGRESSION:ADVANCES
Start at 100.

## ERROR | id=e1
+20% then -20%: the changes cancel. Net change 0%.

## VERIFICATION:REFUTED | implicita
After Monday the price is 120, and Tuesday's 20% is taken from 120, which is 24, more than the 20 gained on Monday.

## CORRECTION:RECOVERED | implicita | id=c1
Tuesday close: 120 - 24 = 96. Net change: -4%.

## VERIFICATION:CONFIRMED | implicita | objetivo=c1
From a start of 50: Monday takes it to 60, Tuesday takes 12 off, leaving 48, which is 4% below 50.

## PROGRESSION:REACHES_OUTPUT
Net change: -4%. Answer: -4.
