"""
Script de validación para verificar que los archivos de anotaciones YOLO sean correctos.
"""

import os
import cv2
from pathlib import Path


def validate_annotations(images_dir, labels_dir, classes_file):
    """
    Valida que los archivos de anotaciones sean correctos.
    
    Args:
        images_dir: Directorio con imágenes JPG
        labels_dir: Directorio con archivos de anotaciones TXT
        classes_file: Archivo con nombres de clases
    """
    # Leer nombres de clases
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    num_classes = len(classes)
    print(f"Clases encontradas: {num_classes}")
    print(f"Clases: {', '.join(classes)}\n")
    
    # Validar archivos
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg')])
    
    errors = []
    valid_count = 0
    
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, base_name + '.txt')
        
        # Verificar que exista el archivo de anotaciones
        if not os.path.exists(label_path):
            errors.append(f"Falta archivo de anotaciones para: {image_file}")
            continue
        
        # Leer imagen
        image = cv2.imread(image_path)
        if image is None:
            errors.append(f"No se puede leer imagen: {image_file}")
            continue
        
        height, width = image.shape[:2]
        
        # Leer anotaciones
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                
                if len(parts) != 5:
                    errors.append(f"{image_file} línea {line_num}: formato incorrecto (esperado 5 valores)")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    bbox_width = float(parts[3])
                    bbox_height = float(parts[4])
                    
                    # Validar rango de valores
                    if class_id < 0 or class_id >= num_classes:
                        errors.append(f"{image_file} línea {line_num}: class_id {class_id} fuera de rango [0, {num_classes-1}]")
                    
                    if not (0 <= x_center <= 1):
                        errors.append(f"{image_file} línea {line_num}: x_center {x_center} fuera de rango [0, 1]")
                    
                    if not (0 <= y_center <= 1):
                        errors.append(f"{image_file} línea {line_num}: y_center {y_center} fuera de rango [0, 1]")
                    
                    if not (0 <= bbox_width <= 1):
                        errors.append(f"{image_file} línea {line_num}: width {bbox_width} fuera de rango [0, 1]")
                    
                    if not (0 <= bbox_height <= 1):
                        errors.append(f"{image_file} línea {line_num}: height {bbox_height} fuera de rango [0, 1]")
                
                except ValueError as e:
                    errors.append(f"{image_file} línea {line_num}: error al parsear valores - {e}")
            
            valid_count += 1
        
        except Exception as e:
            errors.append(f"{image_file}: error al leer anotaciones - {e}")
    
    # Imprimir resultados
    print("="*60)
    print("RESULTADOS DE VALIDACIÓN")
    print("="*60)
    print(f"Total de imágenes: {len(image_files)}")
    print(f"Imágenes validadas correctamente: {valid_count}")
    print(f"Errores encontrados: {len(errors)}")
    
    if errors:
        print("\nPrimeros 10 errores:")
        print("-" * 60)
        for error in errors[:10]:
            print(f"  ✗ {error}")
        if len(errors) > 10:
            print(f"  ... y {len(errors) - 10} errores más")
    else:
        print("\n✓ Todas las anotaciones son válidas")
    
    print("="*60 + "\n")


def main():
    """Función principal."""
    base_dir = r"d:\Documentos\3RO\1ER Semestre\RN\RN\CMP_facade_DB_base\completly"
    images_dir = os.path.join(base_dir, "images")
    labels_dir = os.path.join(base_dir, "labels_yolo")
    classes_file = os.path.join(base_dir, "classes.txt")
    
    validate_annotations(images_dir, labels_dir, classes_file)


if __name__ == "__main__":
    main()
