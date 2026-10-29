---
id: E073
plantilla: P14
dominio: logica
estilo: qwen
respuesta: B
enunciado: Each of three people, A, B, and C, either always tells the truth or always lies. A says, "B is lying." B says, "C is lying." C says, "A and B are both lying." Who tells the truth?
---
## PROGRESSION:ADVANCES
Okay, let's see. Three statements: A says B is lying; B says C is lying; C says A and B are both lying. Each person is either always truthful or always lying. Who is truthful?

## ALTERNATIVE:REJECTED | implicita
Going through all eight truth assignments one by one would settle it, but each statement fixes another person's status, so starting from one assumption and following the chain needs at most two tries.

## ERROR | id=e1
Start with A truthful. Then B is lying, so B's claim that C is lying is false, which makes C truthful. C says that A and B are both lying, and B is indeed lying, so C's statement holds. So A and C tell the truth and B lies.

## VERIFICATION:REFUTED | implicita
C's statement needs both A and B to be lying, and A was taken to be truthful, so C's statement is false, which a truthful C could not say.

## CORRECTION:RECOVERED | implicita | id=c1
The solution with A and C truthful cannot stand, so A is lying. Then A's claim about B is false, so B is truthful, and B's claim makes C a liar.

## VERIFICATION:CONFIRMED | implicita | objetivo=c1
With A lying, B truthful, and C lying, each statement has the truth value its speaker requires: A's is false, B's is true, and C's is false.

## PROGRESSION:REACHES_OUTPUT
So the answer should be B: only B tells the truth.
