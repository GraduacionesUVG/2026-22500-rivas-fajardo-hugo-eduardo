# Análisis, normalización y detector

Instrumentos del trabajo, en dos bloques. El primero prepara el corpus real que
sirve a la vez como clase auténtica y como entrada del pipeline de
manipulación. El segundo construye y evalúa el detector.

| Archivo | Papel |
|---|---|
| `auditoria_corpus.py` | Caracteriza el corpus crudo |
| `normalizar.py` | Produce la clase real normalizada |
| `recodificacion.py` | Decide el método de medición del OE2 |
| `baseline_umbral.py` | Piso de rendimiento del OE3 |
| `dataset_parches.py` | Muestreo de parches pareados |
| `transformaciones.py` | Aumentaciones permitidas |
| `modelo.py` | Detector, con sus tres inicializaciones |
| `preentrenamiento_cifake.py` | Punto de control de la rama CIFAKE |
| `entrenamiento.py` | Bucle de ajuste y evaluación |
| `evaluacion_ciega.py` | Métricas a nivel de imagen sin usar la máscara |
| `inferencia.py` | Atiende una imagen suelta: puntaje, mapa y región señalada |
| `reproducibilidad.py` | Semilla única y procedencia de cada tabla |
| `gradientes.py` | Métricas de gradiente y contraste pareado (OE2) |
| `intervalos.py` | Intervalos de confianza por remuestreo |
| `figuras_gradiente.py` | Figuras comparativas y resumen del efecto |
| `robustez.py` | Caída del detector ante reescalado y recompresión |
| `explicacion.py` | Descripción de la región señalada y su evaluación (OE4) |

## Por qué importa la normalización

Cada imagen manipulada se produce editando su imagen real correspondiente.
Eso hace que el par comparta origen, escena, cámara e iluminación, y que la
única diferencia entre ambas sea la edición, que es exactamente lo que el
detector debe aprender.

Esa propiedad se pierde si las dos mitades del par no pasan por el mismo
tubo. Si la real se guarda con la resolución original de Wikimedia y la
manipulada sale a 1152×768 del editor, el clasificador distingue las clases
por su tamaño y alcanza un F1 alto sin haber aprendido nada sobre
manipulación. Por eso **la imagen normalizada es la clase real**; el archivo
descargado queda solo como respaldo y trazabilidad.

## Decisiones y su motivo

| Decisión | Valor | Motivo |
|---|---|---|
| Proporción | 3:2 | Es la dominante del corpus (mediana 1.49, p75 1.50); recortar hacia ella descarta el mínimo de píxeles. |
| Lado menor | 768 px | A 1024 px habría que escalar hacia arriba el 27.6% del corpus, y la interpolación suaviza los gradientes justo donde el OE2 busca señal. |
| Orientación | Preservada | Recortar un retrato vertical a formato horizontal decapita al sujeto. Verticales quedan en 768×1152. |
| Calidad JPEG | Muestreada del corpus | Un factor fijo le quitaría a la clase real la variabilidad de compresión del material auténtico, y esa diferencia sería aprendible. |
| Semilla | 22500 | Repetir la corrida reproduce las mismas calidades. |

## Uso

```bash
../scraper/.venv/bin/python auditoria_corpus.py --dataset ../dataset
```

```bash
../scraper/.venv/bin/python normalizar.py --dataset ../dataset --salida ../corpus/real
```

Salida en `corpus/real/` más `corpus/metadata_real.csv`, que agrega a los
metadatos originales la orientación, las dimensiones finales, el factor de
calidad aplicado, el SHA-256 del archivo normalizado y una columna
`escalada` que marca las imágenes que hubo que agrandar.

## Pruebas

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Las pruebas fijan contratos y no valores concretos del corpus, de modo que
recalcular una medición no las rompe. Varias existen para que un descuido se
vuelva visible en lugar de degradar los resultados en silencio:

| Suite | Lo que protege |
|---|---|
| `test_normalizar.py` | Recorte en las cuatro orientaciones, conversión de modos de color, orientación EXIF aplicada antes de medir, y la paridad de formato, tamaño y modo entre una salida simulada del editor y su real |
| `test_recodificacion.py` | El veredicto del experimento y el rechazo de entradas con pérdida |
| `test_baseline_umbral.py` | Que el clasificador degenerado quede expuesto: F1 de 0.667 con exactitud balanceada de 0.500 |
| `test_dataset_parches.py` | Que la real y su manipulada reciban las mismas coordenadas, y que ninguna pareja caiga en dos particiones |
| `test_transformaciones.py` | Que el pipeline conserve el alto frecuencia, midiendo la energía del Laplaciano |
| `test_evaluacion_ciega.py` | Que la rejilla cubra cada píxel de la imagen, y que la caja no intervenga en elegir dónde mirar |
| `test_modelo.py` | Que las tres inicializaciones produzcan la misma arquitectura |
| `test_preentrenamiento_cifake.py` | Que la clase positiva sea la sintética y no la que imponga el orden alfabético |

## Experimento de recodificación

Decide si una herramienta de edición reescribe toda la imagen o solo compone
la región editada sobre los píxeles originales. De esa respuesta depende el
método de medición del OE2.

```bash
../scraper/.venv/bin/python recodificacion.py \
    --original ../pruebas/real/par1.png \
    --editada ../pruebas/editada/par1_compuesta_obvio.png
```

**Protocolo.** Para cada herramienta candidata:

1. Tomar una imagen de `corpus/real/` y guardarla como PNG — esa es la
   entrada al editor.
2. Pedir una edición pequeña y localizada, en una esquina o en un objeto
   secundario.
3. Guardar la salida **en PNG**, sin pasar por JPEG.
4. Correr el comando de arriba.

El paso 3 no es opcional: JPEG altera píxeles fuera de la región editada por
su propia compresión, y el veredicto saldría siempre «recodificación
global». El módulo rechaza entradas con pérdida salvo que se pase
`--permitir-con-perdida`.

**Cómo leer el veredicto.**

| Veredicto | Qué significa | Consecuencia para el OE2 |
|---|---|---|
| Recodificación global | Más del 5% de los píxeles fuera de la región cambiaron | Métricas de gradiente sobre la imagen completa; Wilcoxon pareada directa. |
| Composición con máscara | El exterior de la región quedó intacto | Métricas por bloques, reportando el máximo por imagen. |
| Sin cambios | La salida es idéntica a la entrada | La edición no se aplicó; revisar la llamada al editor. |

## Muestreo de parches

El análisis del OE2 y el baseline del OE3 mostraron que la señal de
manipulación es local y relativa: existe dentro de la región editada y se diluye
al promediar sobre la imagen completa. Un detector que recibiera la fotografía
reescalada a la entrada de una red convencional la perdería dos veces, primero
por el promedio y después por la interpolación. Por eso `dataset_parches.py`
entrega parches a resolución nativa, sin reescalar.

**Control de contenido.** De cada pareja se extraen dos parches en las mismas
coordenadas, uno de la real y otro de su manipulada. Las dos muestras comparten
escena, encuadre, cámara e iluminación, de modo que lo único que las separa es
la edición. Sin esa simetría el modelo podría aprender que los podios y los
micrófonos anuncian manipulación, porque son los objetos que el pipeline de
generación eligió con más frecuencia.

**Tamaño fijo.** Escalar el parche con el tamaño de la edición introduciría un
factor de remuestreo distinto por nivel, que el modelo podría leer como atajo
para inferir el nivel en lugar de detectar la manipulación.

**Cobertura de la edición.** Cuando la región editada cabe en el parche, la
ventana la contiene entera; cuando la desborda, la ventana cae dentro de ella.
Exigir solo que se toquen produce positivos con una fracción despreciable de
edición, indistinguibles de un parche auténtico. Los tres niveles quedan en
escalas muy distintas, lo cual es parte de la tarea y no un defecto del
muestreo:

| Nivel | Lado mayor de la caja (mediana) | Fracción del parche que es edición (mediana) |
|---|---|---|
| Obvio | 668 px | 1.000 |
| Normal | 224 px | 0.316 |
| Sutil | 76 px | 0.050 |

## Transformaciones permitidas

La señal vive en el alto frecuencia. Cualquier operación que suavice la textura
antes de que el modelo la vea borra lo único que distingue a las clases, y lo
hace sin producir ningún error. Por eso el catálogo de aumentaciones es
deliberadamente corto y lo excluido importa tanto como lo permitido.

**Permitido.** Volteo horizontal, que reordena los píxeles sin interpolar ni
promediar, de modo que los valores sobreviven exactos. Normalización por media y
desviación de ImageNet, que es una transformación afín por canal: reescala los
valores pero no mezcla píxeles vecinos.

**Excluido.**

| Operación | Motivo |
|---|---|
| Reescalado, incluido `RandomResizedCrop` | La interpolación promedia píxeles vecinos y suaviza el alto frecuencia. Además, como el tamaño de la edición difiere por nivel, un factor de remuestreo variable permitiría inferir el nivel en lugar de la manipulación |
| Desenfoque | Destruye la señal de forma directa |
| Jitter de color, brillo y contraste | Alteran la magnitud de los gradientes, que es la cantidad discriminante según el hallazgo 1 |
| Recompresión JPEG | El corpus guarda cada manipulada con la calidad exacta de su pareja real para que la compresión no sea aprendible; recomprimir reintroduciría esa variable |
| Borrado aleatorio | Puede tapar la región editada y convertir un positivo en un parche sin manipulación, lo que es ruido de etiqueta |

`test_transformaciones.py` no se limita a declarar esta lista: mide la energía
del Laplaciano antes y después del pipeline y verifica que sobreviva. Las
aumentaciones excluidas reducen esa energía a menos de la mitad.

## Preentrenamiento sobre CIFAKE

La comparación del OE1 entrena tres ramas con la misma arquitectura y cambia
solo el origen de los pesos: `imagenet`, `cifake` y `ninguna`. Esa igualdad es
lo que hace que el experimento mida el preentrenamiento y no la capacidad del
modelo.

**El salto de dominio.** CIFAKE son imágenes de 32 por 32 píxeles derivadas de
CIFAR, donde la clase sintética proviene de síntesis completa. El corpus de este
trabajo son parches de 256 píxeles con manipulación local sobre fotografía
auténtica. Entre un conjunto y otro cambian la escala, la resolución, el tipo de
contenido y la naturaleza de la falsificación. Que esos pesos sirvan de punto de
partida es la hipótesis que la comparación pone a prueba, no un supuesto.

**La resolución del preentrenamiento.** A 32 píxeles el tronco de entrada reduce
el mapa de activación muy rápido y las últimas etapas operan sobre mapas casi
puntuales, de modo que sus filtros quedan aprendidos sobre estructuras distintas
de las que verán después. Subir la resolución de entrada alivia eso pero
interpola, lo que agrega un suavizado que ambas clases de CIFAKE reciben por
igual y por lo tanto no es fuga, aunque sí altera la banda de frecuencias del
preentrenamiento. No hay opción libre de costo; el parámetro `--tam` deja
registrar cuál se usó y repetir con la otra.

**La exactitud sobre CIFAKE se guarda en el punto de control.** Si la
transferencia resulta pobre, ese número distingue dos conclusiones muy
distintas: que el preentrenamiento no aprendió, o que aprendió y no transfiere.

## Evaluación a ciegas

`dataset_parches.py` usa la caja del catálogo para decidir dónde recortar. Para
entrenar es correcto, porque el modelo necesita ver parches que contengan la
edición. Para evaluar no lo es: un detector en uso recibe una fotografía y no
sabe qué región se alteró, de modo que medir con la caja reportaría la capacidad
de un oráculo.

Esa distancia ya está medida dos veces en este trabajo. El hallazgo 1 la
documentó sobre los gradientes y el hallazgo 2 la cuantificó sobre el baseline,
donde la exactitud balanceada pasa de 0.600 con máscara a 0.533 sin ella.

`evaluacion_ciega.py` recorre la imagen con una rejilla de parches solapados, 40
por imagen a 1152×768 con paso 128, puntúa cada uno y agrega a una decisión de
imagen. La agregación por máximo corresponde a la hipótesis de que basta una
región anómala para delatar la manipulación; la agregación por promedio de los k
más altos es su versión robusta frente a un parche atípico.

**Localización.** Como el catálogo conserva la caja verdadera, el mismo recorrido
permite medir si el parche mejor puntuado cae sobre la región que de verdad se
editó, exigiendo al menos un 10 % de solapamiento para contar el acierto. Esa
tasa mide si el detector mira donde debe, y es el punto de partida cuantitativo
del OE4. Una imagen puede clasificarse como manipulada por la razón equivocada,
y esta métrica lo revela en lugar de dejarlo supuesto.

## Inferencia sobre una imagen suelta

`experimento_oe1.py` recorre conjuntos y produce tablas; el despliegue del OE4
necesita lo contrario, atender una imagen a la vez. `inferencia.py` cubre ese
caso y devuelve el puntaje agregado, el veredicto contra el umbral, el mapa de
sospecha del tamaño de la imagen y el recorte de la región más sospechosa.

**El detector se reconstruye desde el punto de control.** La forma de su primera
convolución indica si se entrenó con tres canales o con cuatro, y de ahí se elige
la transformación correspondiente. Quien despliega no necesita recordar la
configuración del entrenamiento, que es el dato que más fácil se pierde entre una
etapa y la siguiente.

**El umbral se lee de la tabla del experimento** y no se fija en el código. Usar
el 0.5 de fábrica sobre el puntaje agregado por imagen produce el clasificador
degenerado, que ya ocurrió una vez en este trabajo.

**El mapa promedia las ventanas que cubren cada píxel.** Como la rejilla se
solapa, un píxel recibe el promedio de todas las ventanas que lo contienen, y
ninguna zona de la imagen queda sin puntuar.

## Reparto entre el detector y la explicación

El detector decide y localiza. El modelo de visión y lenguaje recibe la imagen y
el recorte de la región señalada, y describe qué hay ahí y si su textura,
iluminación y bordes resultan coherentes con el resto de la escena. No emite su
propio juicio sobre autenticidad.

La separación evita que dos sistemas juzguen lo mismo por caminos distintos y se
contradigan. Un texto fluido resulta más convincente que un porcentaje aunque sea
menos confiable, y los modelos de visión y lenguaje no son instrumentos forenses.

Como el catálogo conserva la caja verdadera de cada manipulación, la explicación
admite evaluación cuantitativa: la tasa de localización mide con qué frecuencia la
región descrita es la que efectivamente se editó.

## Semilla y procedencia

Una tabla de resultados sin constancia de cómo se produjo no se puede atribuir a
una corrida concreta, y dos mediciones de la misma rama que difieren unas
décimas no se pueden distinguir de un cambio real en el método.

**La semilla.** `fijar_semilla` siembra random, numpy y torch, y desactiva la
búsqueda de algoritmo de cuDNN. Sembrar solo torch no alcanza: el barajado del
cargador y la inicialización de la red consumen el generador global, los
procesos trabajadores derivan el suyo al bifurcar, y las rutinas de convolución
eligen implementación midiendo tiempos, de modo que dos corridas con la misma
semilla sobre la misma GPU no coinciden mientras esa búsqueda siga activa. El
cargador recibe además su propio generador, para que agregar una llamada
aleatoria en otra parte del programa no cambie el orden de las muestras.

**La procedencia.** Cada tabla queda acompañada de un `.procedencia.json` con la
fecha, el commit y si el árbol estaba sucio, las versiones de Python, numpy y
torch, el equipo y la GPU, la semilla, los hiperparámetros, el indicador de
residuo, las parejas de cada partición y el SHA-256 del catálogo. Ese archivo es
lo que permite citar una cifra y poder volver a producirla.

`test_reproducibilidad.py` fija que dos corridas con la misma semilla den una
tabla idéntica y que ninguna salga sin su procedencia.

## Módulo de explicación

`explicacion.py` recibe la imagen y el recorte que entrega `inferencia.py`, y
devuelve una descripción de la región señalada. El reparto con el detector está
descrito arriba; este apartado cubre cómo se sostiene.

**La instrucción deja la decisión fuera de la tarea.** El turno de sistema
delimita qué se describe —objeto, textura, iluminación, bordes, perspectiva— y
qué no corresponde: decidir sobre autenticidad, estimar probabilidades, nombrar a
las personas que aparecen o atribuirles intenciones. Lo último es el compromiso
ético del OE4; lo primero evita que dos componentes juzguen lo mismo por caminos
distintos.

**La instrucción no basta.** Un texto que llama falsa a la fotografía contradice
al detector cuando este la clasificó como auténtica, y el usuario atiende al
texto antes que al número. `fuga_de_veredicto` devuelve los términos de
autenticidad que aparecieron en la respuesta. No reescribe el texto: mide con qué
frecuencia el modelo desobedece, y deja al despliegue la decisión de advertirlo.

**La decodificación es voraz.** Sin muestreo, la misma imagen y la misma región
producen el mismo texto, de modo que la tabla del OE4 se puede volver a producir
igual que las demás del trabajo.

**La evaluación recorre las dos clases.** El prototipo también produce texto
sobre una fotografía auténtica, y un falso positivo explicado con aplomo es el
peor resultado posible. La tabla reporta cobertura, tasa de región acertada por
nivel —que solo tiene sentido sobre la manipulada, porque en la real no hay caja
contra la que comparar— y tasa de fuga. La columna `revision` queda vacía: la
meta del OE4 sobre artefactos mencionados se verifica a mano.

## Medición del gradiente

`gradientes.py` es el instrumento del OE2. Calcula las tres métricas que el
objetivo declara —magnitud media, varianza y entropía— con los tres operadores,
sobre tres zonas: la imagen completa, el bloque de magnitud mínima y el interior
de la región editada.

**La entropía usa un rango fijo y no uno por imagen.** Normalizar el histograma
al máximo de cada imagen volvería la entropía insensible a la escala del
gradiente, que es justamente la cantidad que el objetivo pone a prueba.

**El bloque extremo se elige por la magnitud y las tres métricas se reportan
sobre ese mismo bloque**, de modo que las tres describen la misma región. Si
cada métrica eligiera su propio extremo, las tres hablarían de lugares distintos
de la imagen.

**El contraste es pareado.** La prueba de Wilcoxon de los rangos signados se
aplica sobre las diferencias dentro de cada pareja, y se acompaña de la
correlación rango-biserial como tamaño del efecto: con 494 parejas la prueba
alcanza significancia ante diferencias pequeñas, de modo que un valor de
probabilidad por sí solo no dice de qué tamaño es la diferencia.

**El resultado se desagrega** por nivel de manipulación, por región de origen y
por referente de la edición. La región responde al compromiso del enunciado del
OE3 de reportar el subconjunto guatemalteco por separado.

**El control de compresión es una pasada aparte.** Con `--pasada recompresion`
la imagen auténtica recibe la pasada JPEG adicional que su pareja ya acumula; si
el efecto sobrevive a ese control, no procede del recodificado.

## Intervalos de confianza

`intervalos.py` acompaña cada cifra de su incertidumbre. El remuestreo es
estratificado por clase, porque un remuestreo plano puede devolver una muestra
de una sola clase, para la que el área bajo la curva no está definida.

El remuestreo opera sobre los puntajes de imagen ya agregados y no rehace el
recorrido por cada repetición: lo que se quiere describir es la incertidumbre
del tamaño de la muestra, y rehacer el recorrido costaría horas sin cambiarla.

Con 60 parejas en el conjunto de prueba, el intervalo del área bajo la curva
ronda los nueve puntos. Esa es la razón por la que las comparaciones entre ramas
se reportan también sobre el conjunto de estrés.

## Robustez a la degradación

El corpus se recolectó con un filtro de lado menor igual o mayor a 768 píxeles
sin escalar, y cada imagen acumula una sola pasada de compresión. Ese filtro
**seleccionó fuera el caso difícil**: una fotografía difundida por una
plataforma llega reescalada y recomprimida, y las dos operaciones destruyen la
banda de alta frecuencia donde vive la señal que el detector lee. El detector
está validado en un solo punto de la distribución y todo lo demás es
extrapolación.

`robustez.py` mide esa extrapolación en lugar de declararla. Evalúa el mismo
punto de control sobre los conjuntos de prueba y de estrés bajo varias
condiciones: lado menor a 720, 600 y 480, recompresión a calidad 75 y 60, y la
combinación que aplica una aplicación de mensajería.

**Condición no es aumentación.** Las aumentaciones que borran el alto frecuencia
siguen excluidas del entrenamiento y `test_transformaciones.py` lo hace cumplir:
son azarosas, por imagen, y destruyen la diferencia entre las clases. Una
degradación aplicada por igual a los dos miembros de una pareja es otra cosa:
mueve la pareja entera a otro punto de operación y pregunta si la diferencia
entre sus dos miembros sobrevive ahí. Esa distinción es la que permite medir la
robustez sin contradecir el catálogo de aumentaciones.

**La caja viaja con la imagen.** Cuando una condición cambia el tamaño, la
región editada se lleva al nuevo sistema de coordenadas antes de medir la
localización. Sin eso se estaría comparando contra una caja que ya no
corresponde a ningún lugar de lo que el detector vio.

**Solo se reduce, nunca se amplía.** Ampliar no es algo que ocurra en difusión, y
es además la operación que el corpus evitó por construcción.

## La escena, no solo la región

La tasa de localización es 0.40: el detector acierta la región en cuatro de cada
diez imágenes manipuladas. En las otras seis, la única salida textual del sistema
describía con aplomo un recorte que no contiene la edición, y el lector no tenía
con qué notarlo.

El módulo pide ahora dos descripciones en una sola llamada, con rótulos fijos:
qué muestra la fotografía completa, y qué ocupa la región señalada. Cuando la
región cae mal, el desajuste entre las dos descripciones es visible sin
conocimiento técnico, que es justo lo que un detector de 0.63 de AUC necesita
para no leerse como certeza.

Tres decisiones del reparto. Va en **una sola llamada**, no dos, así que el costo
de inferencia del OE4 no cambia. Una respuesta **sin los rótulos** no se descarta:
se conserva entera como descripción de la región, que es lo que el módulo producía
antes, y se cuenta en el indicador de cobertura de escena, porque medir la
desobediencia del modelo es un dato. Y la **fuga de veredicto se mide sobre la
respuesta entera**, no solo sobre la parte de la región, porque la escena es texto
del modelo igual que la otra y puede invadir la decisión lo mismo.

La prohibición de nombrar a las personas que aparecen pesa más aquí que antes: la
descripción de la escena completa es donde más tienta romperla, de modo que el
turno del usuario la repite al pedirla.

## Síntesis completa: la segunda cabeza

El corpus pareado fijó el objeto de estudio en la manipulación local, y el
detector del OE1 responde a eso: puntúa la anomalía de una región frente al
resto de la imagen. El objetivo general, en cambio, dice «generadas o
manipuladas». Una prueba del prototipo con una imagen enteramente sintética
devolvió 20.2 % de indicio, por debajo del corte de 92.1 %, que es el
comportamiento previsto y no un fallo: en una síntesis completa no hay región
anómala contra la que contrastar.

El no transferir entre las dos tareas está medido en las dos direcciones. La
rama CIFAKE alcanzó 0.950 de exactitud balanceada sobre CIFAKE y 0.493 de AUC
sobre este corpus, por debajo de inicializar al azar: el preentrenamiento
aprendió su tarea y no sirvió para esta. El recíproco es el 20.2 % de arriba.
Por eso son dos clasificadores y no uno con dos salidas.

**Dos condiciones, y la distinción es metodológica.** La condición `pareada`
regenera cada fotografía auténtica completa con difusión de imagen a imagen a
fuerza 0.95: a esa fuerza el ruido inicial borra la textura de partida y
conserva la composición, de modo que todos los píxeles salen del decodificador
y cada imagen sintética mantiene su fotografía de origen. Es la primaria porque
conserva el control de contenido que sostiene todo el diseño del trabajo. La
condición `libre` fabrica la imagen desde una descripción de escena, no admite
pareo, y se reserva como conjunto de generalización: no participa del
entrenamiento ni del ajuste del corte.

**Ninguna descripción nombra a una persona real.** Generar el rostro de un
político identificable sería producir la desinformación que el trabajo estudia,
y lo que el clasificador debe aprender es la firma del generador, no una
identidad. `test_corpus_sintetico.py` verifica esa restricción sobre el banco de
descripciones.

**El muestreo no se condiciona a ninguna caja.** La firma del generador cubre
toda la superficie, así que no hay región privilegiada que buscar: las ventanas
se sortean en cualquier posición, y la fotografía y su versión sintética
entregan la misma ventana.

**La agregación es el promedio y no el máximo.** Bajo el máximo, un solo parche
atípico de una fotografía auténtica bastaría para elevar su indicio. El máximo
se conserva como contraste en la tabla, de modo que la elección quede medida.

**El resultado es una matriz, no una cifra.** Los dos clasificadores, cada uno
con su corte y su agregación, se cruzan contra las cuatro clases de imagen de la
partición de prueba: auténtica, manipulada localmente, sintética pareada y
sintética libre. Lo que la matriz decide es si cada cabeza responde a lo suyo;
una fila que lo contradiga indicaría que una de las dos lee contenido en lugar
de generación.
