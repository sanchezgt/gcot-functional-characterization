---
id: E058
plantilla: P11
dominio: logica
estilo: narrativo
respuesta: knave
enunciado: On an island, every inhabitant is either a knight, who always tells the truth, or a knave, who always lies. You meet two inhabitants, A and B. A says, "B is a knave." B says, "A and I are of the same type." What is B?
---
## PROGRESSION:ADVANCES
First, I set out the rules: knights always tell the truth, knaves always lie, and I have two statements, A's claim that B is a knave and B's claim that the two of them are of the same type.

## PROGRESSION:ADVANCES
I start by supposing that A is a knight. Then A's statement is true, so B is a knave.

## ALTERNATIVE:REJECTED | explicita
Alternatively, A could be a knave. Then A's statement is false, so B is a knight, and B's statement must be true, which would make A a knight as well. That contradicts the supposition that A is a knave, so this possibility is ruled out.

## PROGRESSION:ADVANCES
A cannot be a knave, so A is a knight, and A's true statement makes B a knave.

## VERIFICATION:CONFIRMED | explicita
Let me double-check the solution against both statements. A, a knight, says B is a knave: true, as a knight's statement must be. B, a knave, says they are of the same type: false, as a knave's statement must be. Both hold, so the solution is confirmed.

## PROGRESSION:REACHES_OUTPUT
B is a knave.

Final answer: knave.
