---
id: E126
plantilla: P10
dominio: matematicas
estilo: qwen
respuesta: 78π
enunciado: A closed right circular cylinder has a radius of 3 cm and a height of 10 cm. What is its total surface area, in square centimeters? Give the answer in terms of π.
---
## PROGRESSION:ADVANCES
Okay, let's see. A closed cylinder with radius 3 cm and height 10 cm, and I need the total surface area. The surface is the two circular ends plus the curved side, so SA = 2πr^2 + 2πrh.

## VERIFICATION:CONFIRMED | explicita
Let me double-check the side term in that formula. If I unroll the curved side, it becomes a rectangle whose width is the circumference 2πr and whose height is h, so its area is 2πrh. Yes, the formula is right.

## PROGRESSION:ADVANCES
The two ends: 2π(3)^2 = 2π x 9 = 18π.

## ERROR
The curved side: 2π(3)(10) = 30π.

## VERIFICATION:REFUTED | explicita
Wait, let me check that product. 2 x 3 x 10 is 60, not 30, so I dropped the factor of 2. The 30π above is wrong.

## CORRECTION:RECOVERED | explicita
Fixing it: the curved side is 2π(3)(10) = 60π.

## PROGRESSION:ADVANCES
Total surface area: 18π + 60π = 78π.

## PROGRESSION:REACHES_OUTPUT
So the answer should be 78π square centimeters.
