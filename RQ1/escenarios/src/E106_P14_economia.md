---
id: E106
plantilla: P14
dominio: economia
estilo: narrativo
respuesta: 250
enunciado: In a simple Keynesian model of a closed economy with no taxes, the marginal propensity to consume is 0.8. If government purchases increase by $50 billion, by how much does equilibrium GDP increase, in billions of dollars?
---
## PROGRESSION:ADVANCES
First, I set out what is given: a closed economy without taxes, a marginal propensity to consume of 0.8, and a $50 billion increase in government purchases. I need the resulting increase in equilibrium GDP.

## ALTERNATIVE:REJECTED | implicita
The effect could be traced through the rounds of spending one at a time, but those rounds never end, so I would only approach the total; the spending multiplier gives the whole effect in one step.

## ERROR | id=e1
The spending multiplier is 1 / MPC = 1 / 0.8 = 1.25, so GDP rises by 1.25 x 50 = $62.5 billion.

## VERIFICATION:REFUTED | implicita
The first round alone adds the full $50 billion to income, and households then spend 0.8 of it, another $40 billion, so the first two rounds already raise GDP by $90 billion, well above $62.5 billion.

## CORRECTION:RECOVERED | implicita | id=c1
So 1 / 0.8 cannot be the multiplier. The multiplier is 1 / (1 - MPC) = 1 / 0.2 = 5, and GDP rises by 5 x 50 = $250 billion.

## VERIFICATION:CONFIRMED | implicita | objetivo=c1
Adding up the rounds, 50 + 40 + 32 + 25.6 + ... is a geometric series with ratio 0.8, and its sum is 50 / (1 - 0.8) = 250.

## PROGRESSION:REACHES_OUTPUT
Equilibrium GDP increases by $250 billion.

Final answer: 250.
