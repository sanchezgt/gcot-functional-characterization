---
id: E098
plantilla: P19
dominio: matematicas
estilo: narrativo
respuesta: 6
enunciado: A triangle has sides of length 3, 4, and 5. What is its area?
---
## PROGRESSION:ADVANCES
First, I note the sides: 3, 4, and 5. I want the area of the triangle.

## ALTERNATIVE:REJECTED | implicita
Heron's formula works for any three sides, but these sides satisfy 3^2 + 4^2 = 5^2, so the triangle is right-angled and its two legs give the area directly.

## PROGRESSION:ADVANCES
With legs 3 and 4 meeting at a right angle, the area is half the product of the legs: (1/2) x 3 x 4 = 6.

## VERIFICATION:CONFIRMED | implicita
The inradius of a right triangle is half of (leg + leg - hypotenuse), (3 + 4 - 5) / 2 = 1, and the area also equals the inradius times the semiperimeter, 1 x 6 = 6.

## PROGRESSION:REACHES_OUTPUT
The area of the triangle is 6.

Final answer: 6.
