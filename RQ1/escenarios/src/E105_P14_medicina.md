---
id: E105
plantilla: P14
dominio: medicina
estilo: qwen
respuesta: 20
enunciado: A hospitalized patient is to receive a continuous heparin infusion at 1,000 units per hour. The pharmacy supplies 25,000 units of heparin in 500 mL of 5% dextrose. At what rate, in mL per hour, should the infusion pump be set?
---
## PROGRESSION:ADVANCES
Okay, let's see. The order is heparin at 1,000 units per hour, and the bag holds 25,000 units in 500 mL. I need the pump rate in mL per hour.

## ALTERNATIVE:REJECTED | implicita
A weight-based heparin nomogram could set the rate, but this order fixes the dose at 1,000 units per hour whatever the patient weighs, so the rate follows from the bag's concentration alone.

## ERROR | id=e1
The concentration of the bag is 25,000 / 50 = 500 units per mL. So 1,000 units per hour is 1,000 / 500 = 2 mL per hour.

## VERIFICATION:REFUTED | implicita
At 500 units in each of its 500 mL, the bag would hold 250,000 units, ten times the 25,000 units it contains.

## CORRECTION:RECOVERED | implicita | id=c1
So the bag has 50 units per mL, not 500: 25,000 / 500 = 50. The rate is 1,000 / 50 = 20 mL per hour.

## VERIFICATION:CONFIRMED | implicita | objetivo=c1
At 20 mL per hour the 500-mL bag runs for 25 hours, and 25,000 units spread over 25 hours is 1,000 units each hour, the ordered dose.

## PROGRESSION:ADVANCES
So the pump is set to 20 mL/h, and the bag will need replacing after about a day.

## PROGRESSION:REACHES_OUTPUT
So the answer should be 20 mL per hour.
