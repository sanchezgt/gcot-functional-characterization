---
id: E026
plantilla: P03
dominio: fisica
estilo: qwen
respuesta: 3
enunciado: A ball is dropped from rest from a height of 45 m. Ignoring air resistance and taking g = 10 m/s^2, how many seconds does it take to reach the ground?
---
## PROGRESSION:ADVANCES
Okay, let's see. A ball dropped from rest from 45 m, g = 10 m/s^2, no air resistance. I need the fall time.

## ERROR
For free fall the height equals g times the time, so h = g t, which gives t = 45 / 10 = 4.5 s.

## VERIFICATION:REFUTED | explicita
Hmm, wait. Let me check that formula. The speed grows as g t, but the distance fallen from rest is the average speed times the time, which is (1/2) g t, times t, so h = (1/2) g t^2, not g t. If I plug t = 4.5 s into the correct formula I get (1/2)(10)(4.5^2), about 101 m, not 45 m. So the step above is wrong.

## CORRECTION:RECOVERED | explicita
Correcting it with h = (1/2) g t^2: t^2 = 2h / g = 90 / 10 = 9, so t = 3 s.

## SUPPORT
The factor of one half appears because the speed grows steadily from zero, so the average speed over the fall is half the final speed.

## PROGRESSION:REACHES_OUTPUT
So the ball takes 3 seconds to reach the ground.

The answer should be 3.
