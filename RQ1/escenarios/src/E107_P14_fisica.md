---
id: E107
plantilla: P14
dominio: fisica
estilo: terso
respuesta: 15
enunciado: An object is placed 30 cm in front of a thin converging lens with a focal length of 10 cm. How far from the lens does the image form, in cm?
---
## PROGRESSION:ADVANCES
Need the image distance. Object distance do = 30 cm, focal length f = 10 cm, converging lens.

## ALTERNATIVE:REJECTED | implicita
A scale ray diagram would locate the image, but only to drawing precision; the thin-lens equation 1/f = 1/do + 1/di gives the distance exactly.

## ERROR | id=e1
1/di = 1/10 + 1/30 = 4/30, so di = 7.5 cm.

## VERIFICATION:REFUTED | implicita
The object sits beyond 2f = 20 cm, and an object beyond 2f gives a real image between f and 2f, from 10 to 20 cm; 7.5 cm lies inside the focal length.

## CORRECTION:RECOVERED | implicita | id=c1
Solving 1/f = 1/do + 1/di for 1/di subtracts, it does not add: 1/di = 1/10 - 1/30 = 2/30, so di = 15 cm.

## VERIFICATION:CONFIRMED | implicita | objetivo=c1
1/30 + 1/15 = 1/30 + 2/30 = 3/30 = 1/10, the focal length, and 15 cm lies between f and 2f.

## PROGRESSION:REACHES_OUTPUT
Image 15 cm from the lens, on the far side. Answer: 15.
