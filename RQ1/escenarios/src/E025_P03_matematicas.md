---
id: E025
plantilla: P03
dominio: matematicas
estilo: narrativo
respuesta: 2
enunciado: Pipe A can fill an empty tank in 6 hours and pipe B can fill the same tank in 3 hours. If both pipes are opened at the same time on the empty tank, how many hours does it take to fill the tank?
---
## PROGRESSION:ADVANCES
First, I note what each pipe does on its own: pipe A fills the tank in 6 hours and pipe B fills it in 3 hours. I need the time with both pipes open together.

## ERROR
With both pipes open, I take the average of the two times: (6 + 3) / 2 = 4.5 hours.

## VERIFICATION:REFUTED | explicita
Wait, let me check that against a simple bound. Pipe B alone fills the tank in 3 hours, so adding pipe A can only make it faster: the answer must be less than 3 hours, and 4.5 hours is more than 3. So averaging the times is wrong.

## CORRECTION:RECOVERED | explicita
Correcting the approach: it is the rates that add, not the times. Pipe A fills 1/6 of the tank per hour and pipe B fills 1/3 per hour, so together they fill 1/6 + 1/3 = 1/6 + 2/6 = 3/6 = 1/2 of the tank per hour.

## PROGRESSION:ADVANCES
At half a tank per hour, a full tank takes 1 / (1/2) = 2 hours.

## SUPPORT
Rates add because each pipe pours in its own share of the tank every hour, whether or not the other pipe is running.

## PROGRESSION:REACHES_OUTPUT
Together the pipes fill the tank in 2 hours.

Final answer: 2.
