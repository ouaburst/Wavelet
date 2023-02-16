# Inlämning 1 - HAAR
## Maskininlärning och neurala nätverk A518TA 17931 VT2023 - Oualid Burström

## Syftet med inlämningsuppgiten
Uppgiften går ut på att göra en Wavelet transform och inverstransform med en HAAR wavelet. Syftet med uppgiften är att visa hur data kan komprimeras samt filtreras.
<br>Koden ska utföra transform respektive inverstransform i ett antal svep enligt beskrivning i boken "wavelet made easy" samt föreläsningsanteckningar för matrisformulering.
<br>Genom matrisformulering kan transformen utföras mha matrismultiplikation, koden ska även vara körbar i biblioteket cupy för att visa fördelen med att använda en GPU.
<br>En enkel filter definition ska även skapas för att ta bort wavelet koefficienter som understiger ett bestämt värde (valfritt).

## Utvecklingsmiljön
Koden har skrivits och testats i Spyder som är en utvecklingsmiljö i Anaconda.
<br>Anaconda är en Open Source plattform som inkluderar applikationer och bibliotek i Python.
<br>Operativsystem: Windows 10.
<br>GPU: NVIDIA GeForce GTX 1060 6GB.

## Implementation
Koden är välkommenterad och den består av flera delar:

Importera nödvändiga bibliotek:

numpy för numeriska beräkningar och array-manipulationer
PIL för att ladda och bearbeta bilder
matplotlib för visualiseringar
Ladda en bild med PIL och konvertera den till gråskala format.

Definiera en funktion haar_matrix(n) som returnerar en n x n Haar-matris med hjälp av Haar wavelet-formeln (Eq 7.1, 7.5).

Utför en Haar wavelet-transform på bilden i två svep. I varje svep omvandlas bilden först med Haar-matrisen i horisontell riktning, sedan i vertikal riktning. De resulterande koefficienterna lagras i horizontal_matrix_mult samt vertical_matrix_mult.

Tillämpa en tröskelvärdering på de transformerade koefficienterna genom att sätta alla värden med absolutbelopp mindre än tröskeln C till noll.

Utför en omvänd Haar wavelet-transform på de filtrerade koefficienterna för att erhålla den brusreducerade bilden.

Plotta original samt brusreducerade bilderna med hjälp av matplotlib.

Det finns två implementationer. 
<br>Den ena använder numpy (HAAR_2D_student2023.py) och den andra använder cupy som utnyttjatr GPU (HAAR_2D_student2023_cupy.py).
