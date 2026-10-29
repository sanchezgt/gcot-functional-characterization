---
id: E031
plantilla: P04
dominio: logica
estilo: qwen
respuesta: 4
enunciado: In a class of 30 students, 18 play soccer, 15 play basketball, and 7 play both sports. How many students play neither sport?
---
## PROGRESSION:ADVANCES
Okay, let's see. 30 students total. 18 play soccer, 15 play basketball, 7 play both. I need how many play neither.

## ERROR
Students who play at least one sport: 18 + 15 = 33. That's already more than 30, so every student plays something and the number who play neither is 0.

## VERIFICATION:REFUTED | implicita
Adding 18 and 15 counts the 7 students who play both sports twice, once in each group, so 33 overstates the number of players and cannot be compared with 30 as it stands.

## CORRECTION:RECOVERED | implicita
The number playing at least one sport is 18 + 15 - 7 = 26, with the double-counted students removed once.

## PROGRESSION:ADVANCES
Then the students who play neither are 30 - 26 = 4.

## SUPPORT
This is the inclusion-exclusion principle: the size of a union is the sum of the sizes of the two sets minus the size of their overlap.

## PROGRESSION:REACHES_OUTPUT
So the answer should be 4.
