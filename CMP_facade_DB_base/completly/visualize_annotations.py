"""
Script para visualizar las anotaciones YOLO en las imágenes.
"""

import os
import cv2
import numpy as np
from pathlib import Path


def load_classes(classes_file):
    """Carga los nombres de las clases."""
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes


def draw_bboxes(image, label_path, classes, output_path=None):
    """
    Dibuja las bounding boxes en la imagen.
    
    Args:
        image: Imagen numpy
        label_path: Ruta al archivo de anotaciones
        classes: Lista de nombres de clases
        output_path: Ruta para guardar la imagen (opcional)
    
    Returns:
        Imagen con bounding boxes dibujadas
    """
    height, width = image.shape[:2]
    
    # Colores para cada clase
    colors = [
    (0, 0, 255),      # Window - Rojo
    (255, 128, 0),    # Door - Naranja
    (255, 0, 255),    # Cornice - Rosa
    (0, 255, 255),    # Sill - Amarillo
    (200, 200, 0),    # Balcony - Turquesa
    (0, 255, 0),      # Blind - Verde
    (128, 0, 128),    # Deco - Púrpura
    (255, 255, 0),    # Molding - Cian
    (0, 128, 255),    # Pillar - Azul claro
    (0, 100, 100),    # Shop - Oliva
    ]
    
    # Leer anotaciones
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                bbox_width = float(parts[3])
                bbox_height = float(parts[4])
                
                # Convertir de coordenadas normalizadas a píxeles
                x_center_px = int(x_center * width)
                y_center_px = int(y_center * height)
                width_px = int(bbox_width * width)
                height_px = int(bbox_height * height)
                
                # Calcular esquinas
                x1 = max(0, x_center_px - width_px // 2)
                y1 = max(0, y_center_px - height_px // 2)
                x2 = min(width, x_center_px + width_px // 2)
                y2 = min(height, y_center_px + height_px // 2)
                
                # Dibujar rectángulo
                color = colors[class_id % len(colors)]
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Dibujar etiqueta
                class_name = classes[class_id] if class_id < len(classes) else f"Class {class_id}"
                label_text = f"{class_name}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
                
                # Fondo para el texto
                cv2.rectangle(image, (x1, y1 - text_size[1] - 4), 
                            (x1 + text_size[0], y1), color, -1)
                cv2.putText(image, label_text, (x1, y1 - 2), font, 
                          font_scale, (255, 255, 255), thickness)
    
    # Guardar imagen si se especifica
    if output_path:
        cv2.imwrite(output_path, image)
    
    return image


def visualize_samples(images_dir, labels_dir, classes_file, output_dir, num_samples=5):
    """
    Visualiza algunos ejemplos de anotaciones.
    
    Args:
        images_dir: Directorio con imágenes
        labels_dir: Directorio con anotaciones
        classes_file: Archivo con nombres de clases
        output_dir: Directorio para guardar imágenes visualizadas
        num_samples: Número de muestras a visualizar
    """
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar clases
    classes = load_classes(classes_file)
    
    # Obtener lista de imágenes
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    # Seleccionar muestras
    step = max(1, len(image_files) // num_samples)
    sample_files = image_files[::step][:num_samples]
    
    print(f"Visualizando {len(sample_files)} muestras...\n")
    
    for image_file in sample_files:
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, base_name + '.txt')
        output_path = os.path.join(output_dir, f"visualized_{image_file}")
        
        # Leer imagen
        image = cv2.imread(image_path)
        if image is None:
            print(f"✗ No se puede leer: {image_file}")
            continue
        
        # Dibujar bounding boxes
        image_with_bboxes = draw_bboxes(image.copy(), label_path, classes, output_path)
        
        # Contar anotaciones
        num_bboxes = 0
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                num_bboxes = len([l for l in f.readlines() if l.strip()])
        
        print(f"✓ {image_file}: {num_bboxes} objetos detectados")
    
    print(f"\n✓ Imágenes visualizadas guardadas en: {output_dir}")


def main():
    """Función principal."""
    base_dir = r"d:\Documentos\3RO\1ER Semestre\RN\RN\CMP_facade_DB_base\completly"
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels_yolo")
    classes_file = os.path.join(base_dir, "classes.txt")
    output_dir = os.path.join(base_dir, "visualized_samples")
    
    visualize_samples(images_dir, labels_dir, classes_file, output_dir, num_samples=10)


if __name__ == "__main__":
    main()
