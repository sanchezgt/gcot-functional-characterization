---
id: E119
plantilla: P03
dominio: informatica
estilo: escolar
respuesta: 15000
enunciado: An uncompressed image is 100 pixels wide and 50 pixels high, and each pixel is stored with 24 bits of color. How many bytes are needed to store the pixel data? Ignore any file header.
---
## PROGRESSION:ADVANCES
The image is 100 pixels wide and 50 pixels high, with 24 bits per pixel. We must find how many bytes the pixel data takes.

## PROGRESSION:ADVANCES
Number of pixels: 100 x 50 = 5,000.

## ERROR
Each pixel takes 24, so the pixel data takes 5,000 x 24 = 120,000 bytes.

## VERIFICATION:REFUTED | explicita
Wait, let me check the units of that product. The 24 counts bits per pixel, not bytes, so 5,000 x 24 = 120,000 is a number of bits. Calling it 120,000 bytes overstates the size eight times, so that step is wrong.

## CORRECTION:RECOVERED | explicita
Correcting the units: a byte is 8 bits, so the pixel data takes 120,000 / 8 = 15,000 bytes.

## SUPPORT
24-bit color stores 8 bits, one byte, for each of the red, green, and blue channels of a pixel.

## PROGRESSION:REACHES_OUTPUT
The pixel data needs 15,000 bytes.

Answer: 15000.
