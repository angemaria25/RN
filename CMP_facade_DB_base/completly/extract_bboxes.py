"""
Script para extraer bounding boxes de máscaras de segmentación
y generar archivos de anotaciones en formato YOLO.

El script procesa máscaras PNG donde cada valor de píxel representa
una clase diferente, extrae las bounding boxes de cada objeto y
genera archivos de texto en formato YOLO (class_id x_center y_center width height).
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def get_class_names():
    # Mapeo corregido basado en la inspección visual de las cajas:
    # Valor de Píxel -> Nombre Real del Objeto
    mapping_real = {
        78:  "window",    # Ventana (Amarillo)
        50:  "door",      # Puerta / Arco de entrada (Antes salía como balcony)
        126: "cornice",   # Cornisa / Franja superior (Antes salía como door)
        210: "sill",      # Alféizar (Bajo la ventana)
        194: "balcony",   # Balcón
        178: "blind",     # Persiana (Dentro de la ventana)
        76:  "deco",      # Decoración
        176: "molding",   # Moldura
        225: "pillar",    # Pilar / Columna
        128: "shop"       # Tienda (Arco comercial)
    }

    # Creamos los diccionarios para YOLO (IDs del 0 al 9)
    pixel_to_class = {}
    class_names = {}
    
    # Ordenamos las clases para que siempre tengan el mismo ID
    for i, pix_val in enumerate(sorted(mapping_real.keys())):
        name = mapping_real[pix_val]
        pixel_to_class[pix_val] = i
        class_names[i] = name
        
    return pixel_to_class, class_names

def extract_bboxes_from_mask(mask_path, image_width, image_height, pixel_to_class):
    """
    Extrae bounding boxes de una máscara de segmentación.
    
    Args:
        mask_path: Ruta a la imagen de máscara PNG
        image_width: Ancho de la imagen original
        image_height: Alto de la imagen original
        pixel_to_class: Diccionario que mapea valores de píxel a IDs de clase YOLO
    
    Returns:
        Lista de tuplas (class_id, x_center, y_center, width, height) normalizadas
    """
    # Leer la máscara
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Error al leer la máscara: {mask_path}")
        return []
    
    bboxes = []
    
    # Obtener valores únicos (clases) en la máscara
    unique_pixels = np.unique(mask)
    
    
    for pixel_val in unique_pixels:
        # Solo procesamos si el píxel está en nuestro mapeo (excluye background/facade)
        if pixel_val in pixel_to_class:
            yolo_class_id = pixel_to_class[pixel_val]
            binary_mask = (mask == pixel_val).astype(np.uint8)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filtro para imágenes pequeñas: si la imagen es pequeña, 
                # tal vez quieras bajar este 5 a 2 o 3.
                if w < 4 or h < 4: continue 
                
                x_center = (x + w / 2) / image_width
                y_center = (y + h / 2) / image_height
                width_norm = w / image_width
                height_norm = h / image_height
                
                bboxes.append((yolo_class_id, x_center, y_center, width_norm, height_norm))
    return bboxes

# El resto de la función process_dataset y main se mantienen igual, 
# pero asegúrate de que usen estas funciones actualizadas.
    
    
def process_dataset(images_dir, masks_dir, output_dir, pixel_to_class, class_names):
    """
    Procesa todo el dataset extrayendo bounding boxes.
    
    Args:
        images_dir: Directorio con imágenes JPG
        masks_dir: Directorio con máscaras PNG
        output_dir: Directorio de salida para archivos de anotaciones
        pixel_to_class: Diccionario que mapea valores de píxel a IDs de clase YOLO
        class_names: Diccionario de nombres de clases
    """
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtener lista de imágenes
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    print(f"Procesando {len(image_files)} imágenes...")
    
    stats = {
        'total_images': len(image_files),
        'images_with_objects': 0,
        'total_objects': 0,
        'objects_per_class': {class_id: 0 for class_id in class_names.keys()}
    }
    
    for image_file in tqdm(image_files, desc="Extrayendo bounding boxes"):
        # Obtener nombre base sin extensión
        base_name = os.path.splitext(image_file)[0]
        
        # Rutas
        image_path = os.path.join(images_dir, image_file)
        mask_path = os.path.join(masks_dir, base_name + '.png')
        output_path = os.path.join(output_dir, base_name + '.txt')
        
        # Verificar que exista la máscara
        if not os.path.exists(mask_path):
            print(f"Advertencia: No se encontró máscara para {image_file}")
            continue
        
        # Leer imagen para obtener dimensiones
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error al leer imagen: {image_path}")
            continue
        
        height, width = image.shape[:2]
        
        # Extraer bounding boxes
        bboxes = extract_bboxes_from_mask(mask_path, width, height, pixel_to_class)
        
        # Guardar anotaciones
        if bboxes:
            stats['images_with_objects'] += 1
            stats['total_objects'] += len(bboxes)
            
            with open(output_path, 'w') as f:
                for class_id, x_center, y_center, width_norm, height_norm in bboxes:
                    # Formato YOLO: class_id x_center y_center width height (todos normalizados)
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}\n")
                    stats['objects_per_class'][class_id] += 1
        else:
            # Crear archivo vacío si no hay objetos
            open(output_path, 'w').close()
    
    return stats


def print_statistics(stats, class_names):
    """Imprime estadísticas del procesamiento."""
    print("\n" + "="*60)
    print("ESTADÍSTICAS DEL PROCESAMIENTO")
    print("="*60)
    print(f"Total de imágenes: {stats['total_images']}")
    print(f"Imágenes con objetos: {stats['images_with_objects']}")
    print(f"Total de objetos detectados: {stats['total_objects']}")
    print(f"Promedio de objetos por imagen: {stats['total_objects'] / stats['total_images']:.2f}")
    
    print("\nObjetos por clase:")
    print("-" * 40)
    for class_id in sorted(class_names.keys()):
        count = stats['objects_per_class'][class_id]
        class_name = class_names[class_id]
        print(f"  {class_id:2d} - {class_name:15s}: {count:5d}")
    print("="*60 + "\n")


def main():
    """Función principal."""
    # Configurar rutas
    base_dir = r"d:\Documentos\3RO\1ER Semestre\RN\RN\CMP_facade_DB_base\completly"
    images_dir = os.path.join(base_dir, "images")
    masks_dir = os.path.join(base_dir, "masks")
    output_dir = os.path.join(base_dir, "labels_yolo")
    
    # Obtener nombres de clases y mapeo de píxeles
    pixel_to_class, class_names = get_class_names()
    
    # Verificar que existan los directorios
    if not os.path.exists(images_dir):
        print(f"Error: No se encontró el directorio de imágenes: {images_dir}")
        return
    
    if not os.path.exists(masks_dir):
        print(f"Error: No se encontró el directorio de máscaras: {masks_dir}")
        return
    
    print(f"Directorio de imágenes: {images_dir}")
    print(f"Directorio de máscaras: {masks_dir}")
    print(f"Directorio de salida: {output_dir}")
    print(f"Clases detectadas: {len(class_names)}\n")
    
    # Procesar dataset
    stats = process_dataset(images_dir, masks_dir, output_dir, pixel_to_class, class_names)
    
    # Imprimir estad��sticas
    print_statistics(stats, class_names)
    
    print(f"✓ Anotaciones guardadas en: {output_dir}")


if __name__ == "__main__":
    main()
