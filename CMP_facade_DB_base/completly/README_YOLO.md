# Dataset CMP Facade - Anotaciones YOLO para Detección de Objetos

## Descripción

Este dataset contiene anotaciones en formato YOLO para detección de objetos de elementos arquitectónicos en fachadas. Las anotaciones fueron extraídas automáticamente de las máscaras de segmentación del dataset original CMP Facade Database.

## Contenido

- **images/**: Imágenes originales en formato JPG (378 imágenes)
- **masks/**: Máscaras de segmentación en formato PNG (original)
- **labels_yolo/**: Anotaciones en formato YOLO (378 archivos .txt)
- **classes.txt**: Nombres de las 12 clases
- **extract_bboxes.py**: Script para extraer bounding boxes de máscaras
- **validate_annotations.py**: Script para validar las anotaciones
- **visualize_annotations.py**: Script para visualizar las anotaciones en las imágenes

## Clases

El dataset contiene 12 clases de elementos arquitectónicos:

| ID | Nombre | Descripción |
|----|--------|-------------|
| 0 | background | Fondo (no se usa en detección) |
| 1 | facade | Fachada |
| 2 | window | Ventana |
| 3 | door | Puerta |
| 4 | cornice | Cornisa |
| 5 | sill | Alféizar |
| 6 | balcony | Balcón |
| 7 | blind | Persiana |
| 8 | deco | Decoración |
| 9 | molding | Moldura |
| 10 | pillar | Pilar |
| 11 | shop | Tienda |

## Estadísticas

- **Total de imágenes**: 378
- **Total de objetos detectados**: 30,765
- **Promedio de objetos por imagen**: 81.39
- **Formato de anotaciones**: YOLO (class_id x_center y_center width height)

### Distribución por clase

| Clase | Cantidad |
|-------|----------|
| background | 1,229 |
| facade | 367 |
| window | 12,075 |
| door | 1,940 |
| cornice | 2,798 |
| sill | 1,091 |
| balcony | 956 |
| blind | 3,317 |
| deco | 1,843 |
| molding | 2,270 |
| pillar | 2,488 |
| shop | 391 |

## Formato de Anotaciones

Cada archivo .txt en `labels_yolo/` contiene una línea por objeto detectado con el siguiente formato:

```
<class_id> <x_center> <y_center> <width> <height>
```

Donde:
- `class_id`: ID de la clase (0-11)
- `x_center`: Coordenada X del centro normalizada (0-1)
- `y_center`: Coordenada Y del centro normalizada (0-1)
- `width`: Ancho del bounding box normalizado (0-1)
- `height`: Alto del bounding box normalizado (0-1)

### Ejemplo

```
2 0.456789 0.234567 0.123456 0.234567
3 0.654321 0.456789 0.098765 0.123456
```

## Uso

### 1. Validar las anotaciones

```bash
python validate_annotations.py
```

Este script verifica que todos los archivos de anotaciones sean válidos y que los valores estén en los rangos correctos.

### 2. Visualizar las anotaciones

```bash
python visualize_annotations.py
```

Este script genera imágenes con las bounding boxes dibujadas para verificar visualmente la calidad de las anotaciones.

### 3. Usar con YOLOv8

```python
from ultralytics import YOLO

# Cargar modelo
model = YOLO('yolov8n.pt')

# Entrenar
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    device=0
)
```

Primero, crea un archivo `dataset.yaml`:

```yaml
path: /ruta/al/dataset
train: images
val: images
test: images

nc: 12
names: ['background', 'facade', 'window', 'door', 'cornice', 'sill', 
        'balcony', 'blind', 'deco', 'molding', 'pillar', 'shop']
```

### 4. Usar con OpenCV

```python
import cv2
import numpy as np

# Leer imagen
image = cv2.imread('images/cmp_b0001.jpg')
height, width = image.shape[:2]

# Leer anotaciones
with open('labels_yolo/cmp_b0001.txt', 'r') as f:
    for line in f:
        class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())
        
        # Convertir a píxeles
        x_center_px = int(x_center * width)
        y_center_px = int(y_center * height)
        width_px = int(bbox_width * width)
        height_px = int(bbox_height * height)
        
        # Calcular esquinas
        x1 = x_center_px - width_px // 2
        y1 = y_center_px - height_px // 2
        x2 = x_center_px + width_px // 2
        y2 = y_center_px + height_px // 2
        
        # Dibujar
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('Detections', image)
cv2.waitKey(0)
```

## Mapeo de Píxeles a Clases

Las máscaras originales usan valores de píxel específicos para cada clase:

| Valor de píxel | Clase |
|---|---|
| 29 | background |
| 19 | facade |
| 78 | window |
| 126 | door |
| 178 | cornice |
| 210 | sill |
| 50 | balcony |
| 194 | blind |
| 76 | deco |
| 176 | molding |
| 225 | pillar |
| 128 | shop |

## Extracción de Bounding Boxes

El script `extract_bboxes.py` realiza los siguientes pasos:

1. Lee cada máscara PNG
2. Identifica los píxeles de cada clase
3. Encuentra los contornos de cada objeto
4. Calcula el bounding box mínimo para cada contorno
5. Normaliza las coordenadas
6. Guarda en formato YOLO

### Parámetros de extracción

- **Tamaño mínimo de bbox**: 5 píxeles (para filtrar ruido)
- **Método de contornos**: RETR_EXTERNAL (solo contornos externos)
- **Aproximación de contornos**: CHAIN_APPROX_SIMPLE

## Calidad de las Anotaciones

- ✓ Todas las anotaciones han sido validadas
- ✓ Todos los valores están en los rangos correctos
- ✓ Todas las imágenes tienen archivos de anotaciones correspondientes
- ✓ Se han visualizado muestras para verificar la calidad

## Referencia Original

```
@INPROCEEDINGS{Tylecek13
  author = {Radim Tyle{\v c}ek Radim {\v S}{\' a}ra}
  title = {Spatial Pattern Templates for Recognition of Objects with Regular Structure}
  booktitle = {Proc. GCPR}
  year = {2013}
  address = {Saarbrucken Germany}
}
```

Website: cmp.felk.cvut.cz/~tylecr1/facade/

## Notas

- Las anotaciones fueron extraídas automáticamente de las máscaras de segmentación
- Algunos objetos muy pequeños (< 5 píxeles) fueron filtrados para reducir ruido
- La clase "background" se incluye en las estadísticas pero generalmente no se usa en entrenamiento
- Las coordenadas están normalizadas (0-1) para ser independientes de la resolución de la imagen

