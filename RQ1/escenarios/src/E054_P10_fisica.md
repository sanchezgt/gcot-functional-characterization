---
id: E054
plantilla: P10
dominio: fisica
estilo: qwen
respuesta: 2
enunciado: A spring with spring constant 200 N/m is compressed by 0.10 m and then released, pushing a 0.50-kg cart along a frictionless horizontal track. What is the speed of the cart, in m/s, when it leaves the spring?
---
## PROGRESSION:ADVANCES
Okay, let's see. Spring constant 200 N/m, compression 0.10 m, cart of 0.50 kg, no friction. I need the cart's speed when it leaves the spring, so the plan is to find the energy stored in the spring and convert it to kinetic energy.

## PROGRESSION:ADVANCES
Energy stored in the spring: (1/2) k x^2 = (1/2)(200)(0.10)^2 = (1/2)(200)(0.01) = 1 J.

## VERIFICATION:CONFIRMED | explicita
Let me double-check the formula I used: the elastic potential energy of a spring is one half k x squared, with x the compression from the natural length. Yes, and 0.10 squared is 0.01, so 1 J is right.

## PROGRESSION:ADVANCES
With no friction, all of that energy becomes kinetic energy of the cart: (1/2) m v^2 = 1 J.

## PROGRESSION:ADVANCES
So v^2 = 2 x 1 / 0.50 = 4.

## ERROR
So v = 4 m/s.

## VERIFICATION:REFUTED | explicita
Wait, that's not right. Let me check: 4 is v squared, not v, so the square root was never taken. The value above is wrong.

## CORRECTION:RECOVERED | explicita
Fixing it: v = sqrt(4) = 2 m/s.

## PROGRESSION:REACHES_OUTPUT
So the answer should be 2 m/s.
