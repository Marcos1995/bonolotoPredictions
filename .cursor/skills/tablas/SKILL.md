---
name: tablas
description: >
  Any HTML table or data grid, or "tabla", "ordenable", "columnas".
  Every column header sorts on click. First click is descending.
  A frozen column must hide the cells scrolling underneath.
---

# Tablas

Toda tabla se ordena al pulsar el encabezado. Cualquier columna. El primer clic ordena de mayor a menor; el segundo, al revés. El control es un botón dentro del `th`, con foco visible. El puesto 1…n, si existe, es la primera columna y se renumera después de ordenar.

El valor de orden va en `data-n` (número o texto). No ordenes el texto visible si lleva el símbolo del euro o un porcentaje.

Si una columna se queda fija al hacer scroll horizontal, el fondo es opaco en claro y en oscuro, el `z-index` tapa lo que se mueve (también los encabezados) y el `left` es el ancho medido de la columna anterior, no un rem fijo. Si eso no cierra la junta, no dejes columnas fijas: mejor scroll limpio que ver la columna de al lado por debajo.
