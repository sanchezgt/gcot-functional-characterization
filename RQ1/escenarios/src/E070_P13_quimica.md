---
id: E070
plantilla: P13
dominio: quimica
estilo: escolar
respuesta: 4
enunciado: Hydrogen and oxygen react according to 2 H2 + O2 -> 2 H2O. If 4 mol of H2 react with 3 mol of O2, what is the maximum amount of water that can be produced, in moles?
---
## PROGRESSION:ADVANCES
We have 4 mol of H2 and 3 mol of O2 reacting by 2 H2 + O2 -> 2 H2O, and we want the maximum amount of water. The amount produced is set by the limiting reagent.

## ALTERNATIVE:REJECTED | explicita
One alternative is to take the reactant present in the smaller number of moles, here O2, as the limiting one. But the coefficients in the equation are different, so the amounts must be compared through the reaction ratio; that shortcut is set aside.

## ERROR | id=e1
Each mole of H2 needs one mole of O2, so 4 mol of H2 need 4 mol of O2. Only 3 mol are available, so O2 is limiting, and 3 mol of O2 produce 2 x 3 = 6 mol of water.

## VERIFICATION:REFUTED | explicita
Let me check that ratio against the equation: 2 H2 react with 1 O2, so each mole of H2 needs half a mole of O2, not one. The step above used the wrong ratio.

## CORRECTION:RECOVERED | explicita | id=c1
Correcting it: 4 mol of H2 need 4 x 1/2 = 2 mol of O2. With 3 mol available, O2 is in excess and H2 is limiting. Since 2 mol of H2 give 2 mol of H2O, 4 mol of H2 give 4 mol of water.

## VERIFICATION:CONFIRMED | explicita | objetivo=c1
Let me double-check by counting atoms: 4 mol of H2O contain 8 mol of H atoms, which is the 4 mol of H2, and 4 mol of O atoms, which is 2 mol of O2, leaving 1 mol of O2 unreacted. Everything balances, so the result is confirmed.

## PROGRESSION:REACHES_OUTPUT
The maximum amount of water is 4 mol.

Answer: 4.
