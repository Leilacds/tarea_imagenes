# Clasificador de lápices y lapiceras

Aplicación local que clasifica una foto con un solo objeto visible como
**lápiz de grafito** o **lapicera común**. La página muestra la categoría
predicha y la confianza estimada por el modelo.

La aplicación y la predicción se ejecutan en tu computadora. Las fotos no se
envían a un servicio de IA. La primera vez que se entrena el modelo, TensorFlow
descarga los pesos preentrenados de ImageNet; después quedan guardados en la
computadora.

## Requisitos

- Windows y PowerShell.
- [Git](https://git-scm.com/) para clonar el repositorio.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) para gestionar
  Python, el entorno virtual y las dependencias.
- Python 3.12. Si no está instalado, `uv` puede descargarlo al sincronizar el
  proyecto.

## Descargar y preparar el proyecto

En PowerShell, elegí dónde guardar el repositorio y ejecutá:

```powershell
git clone --branch desarrollo https://github.com/Leilacds/tarea_imagenes.git
Set-Location .\tarea_imagenes
uv sync --python 3.12
```

`uv sync` crea o actualiza el entorno virtual local (`.venv`) e instala las
versiones registradas en `uv.lock`. No hace falta activar el entorno: los
comandos `uv run` lo usan automáticamente.

### Agregar las fotos

Las imágenes no se incluyen en Git: el archivo `.gitignore` excluye `data/` y
`models/`. Copiá las fotos al proyecto con esta estructura:

```text
data/
├── train/
│   ├── lapicera/    # Fotos para entrenar
│   └── lapiz/
└── validation/
    ├── lapicera/    # Fotos reservadas para validar
    └── lapiz/
```

Cada subcarpeta debe contener fotos de su clase. Para entrenar, usá JPG/JPEG o
PNG; el cargador de directorios de Keras no admite WEBP directamente. Las
copias WEBP del conjunto preparado ya se convirtieron a JPEG. La aplicación sí
permite subir JPG, PNG y WEBP. Si tenés varias fotos del mismo objeto,
mantenelas en una sola partición para evitar que el modelo parezca rendir
mejor por reconocer un objeto ya visto.

En el repositorio original, `data/lapiz/` y `data/lapicera/` contienen las fotos
originales. Las carpetas `data/train/` y `data/validation/` son copias
separadas para el entrenamiento actual; las imágenes originales se conservan.

## Entrenar el modelo

Desde la carpeta del proyecto, ejecutá:

```powershell
uv run python train.py
```

El entrenamiento se ejecuta en la computadora, normalmente usando CPU en
Windows. Al terminar, crea:

```text
models/
├── clasificador.keras
└── clases.json
```

Esos archivos son locales y están excluidos de Git. Si clonás el proyecto en
otra computadora, necesitás copiar las fotos y volver a entrenar el modelo.

## Iniciar la aplicación

Después de entrenar:

```powershell
uv run streamlit run app.py
```

Abrí en el navegador la dirección local que indique Streamlit (normalmente
`http://localhost:8501`). Subí un archivo JPG, PNG o WEBP que muestre un solo
objeto. Para detener la aplicación, presioná `Ctrl+C` en la terminal.

La confianza es la puntuación que asigna el modelo a su categoría ganadora;
no está calibrada como probabilidad y no garantiza que la clasificación sea
correcta.

## Tecnologías y bibliotecas

Las versiones exactas quedan fijadas en `uv.lock`; el proyecto declara:

- **Python 3.12**: lenguaje de la aplicación y del entrenamiento.
- **TensorFlow con `tf.keras`**: carga de imágenes, entrenamiento, evaluación y
  predicción del modelo.
- **Keras MobileNetV2**: red convolucional preentrenada con ImageNet, usada como
  extractor de características. Se congela esa base y se entrena una cabeza de
  clasificación nueva para las dos categorías. Es *transfer learning*; no se
  entrena la red completa desde cero.
- **Streamlit**: interfaz web local para subir una imagen y mostrar el
  resultado.
- **Pillow**: lectura, corrección de orientación y conversión de imágenes a
  RGB.
- **uv**: creación del entorno virtual y sincronización de dependencias.

El script redimensiona las fotos a 160 × 160 píxeles, usa aumentos aleatorios
de imagen durante el entrenamiento y detiene el ajuste si la pérdida de
validación deja de mejorar. Guarda el modelo en formato Keras y el orden de las
clases en JSON para que la aplicación interprete correctamente la predicción.

## Datos actuales y evaluación

En la primera ejecución se usaron **63 fotos para entrenar** (31 lápices y
32 lapiceras) y **16 para validar** (8 por clase). El resultado observado fue
15 aciertos de 16 (93,8%) en esa ejecución. Es una referencia inicial, no una
medida precisa del rendimiento real: con solo 16 fotos, cada error cambia el
porcentaje en 6,25 puntos.

Además, el conjunto de validación se usa durante el entrenamiento para decidir
cuándo detenerlo y restaurar los mejores pesos. Por eso, esta medición no es
una prueba final independiente. El proyecto todavía no tiene una partición
`test`; una evaluación final debería hacerse con fotos nuevas que no se usaron
para entrenar, seleccionar ni ajustar el modelo.

### ¿Cuántas fotos hacen falta?

No existe una cantidad mínima universal establecida por un estándar para
obtener una “buena predicción”. Depende de la variedad de objetos, fondos,
iluminación y ángulos, y de qué nivel de error sea aceptable. Las guías de
aprendizaje automático recomiendan que los conjuntos de evaluación sean
representativos, suficientemente grandes y separados de los datos de
entrenamiento; una división como 70% entrenamiento, 15% validación y 15% prueba
es una referencia posible, no una regla obligatoria.

Como **meta práctica para este prototipo**, apuntá a reunir entre 100 y 200
fotos distintas por clase para entrenar y, cuando haya suficientes datos,
reservá 100 fotos nuevas por clase para una prueba final. Con alrededor de 100
ejemplos de prueba por clase, el margen de incertidumbre de una proporción
todavía puede rondar ±10 puntos porcentuales en el caso conservador; para
medir con más precisión hacen falta más ejemplos. Estas cantidades son una
orientación, no una garantía ni un requisito oficial.

Priorizá la diversidad y la separación por objeto antes que acumular ráfagas o
copias casi iguales. Las fotos de prueba deberían parecerse a las imágenes que
se usarán en la aplicación, e incluir distintos tamaños, marcas, ángulos,
fondos e iluminación.

## Referencias

- [TensorFlow: separar datos de entrenamiento, validación y prueba](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets)
- [TensorFlow: transferencia de aprendizaje y ajuste fino](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Keras: MobileNetV2](https://keras.io/api/applications/mobilenet/)
