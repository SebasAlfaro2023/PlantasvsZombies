from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)


# Carpeta donde se almacenarán las imágenes subidas
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Asegúrate de que la carpeta uploads existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Cargar el modelo de IA previamente entrenado
model = tf.keras.models.load_model('mejor_modelo.keras')

# Definir las clases del modelo
clases = ['Apple-Apple-scab', 'Apple-Black-rot', 'Apple-Cedar-apple-rust', 'Apple-healthy', 'Blueberry-healthy', 
          'Cherry-sour-Powdery-mildew', 'Cherry-sour-healthy', 'Corn-Cercospora-leaf-spot-Gray-leaf-spot', 
          'Corn-Common-rust', 'Corn-Northern-Leaf-Blight', 'Corn-healthy', 'Grape-Black-rot', 
          'Grape-Esca-Black_Measles', 'Grape-Leaf-blight-Isariopsis', 'Grape-healthy', 
          'Orange-Haunglongbing-Citrus-greening', 'Peach-Bacterial-spot', 'Peach-healthy', 
          'Pepper-bell-Bacterial-spot', 'Pepper-bell-healthy', 'Potato-Early-blight', 'Potato-Late-blight', 
          'Potato-healthy', 'Raspberry-healthy', 'Soybean-healthy', 'Squash-Powdery-mildew', 
          'Strawberry-Leaf-scorch', 'Strawberry-healthy', 'Tomato-Bacterial-spot', 'Tomato-Early-blight', 
          'Tomato-Late-blight', 'Tomato-Leaf-Mold', 'Tomato-Septoria-leaf-spot', 
          'Tomato-Spider-mites-Two-spotted-spider-mite', 'Tomato-Target-Spot', 
          'Tomato-Tomato-Yellow-Leaf-Curl-Virus', 'Tomato-Tomato-mosaic-virus', 'Tomato-healthy']
# Clases saludables
healthy_classes = [
    'Apple-healthy',
    'Blueberry-healthy',
    'Cherry-sour-healthy',
    'Corn-healthy',
    'Grape-healthy',
    'Peach-healthy',
    'Pepper-bell-healthy',
    'Potato-healthy',
    'Raspberry-healthy',
    'Soybean-healthy',
    'Strawberry-healthy',
    'Tomato-healthy'
    ]
# Clases enfermas
unhealthy_classes = [
    'Apple-Apple-scab',
    'Apple-Black-rot',
    'Apple-Cedar-apple-rust',
    'Cherry-sour-Powdery-mildew',
    'Corn-Cercospora-leaf-spot-Gray-leaf-spot',
    'Corn-Common-rust',
    'Corn-Northern-Leaf-Blight',
    'Grape-Black-rot',
    'Grape-Esca-Black_Measles',
    'Grape-Leaf-blight-Isariopsis',
    'Orange-Haunglongbing-Citrus-greening',
    'Peach-Bacterial-spot',
    'Pepper-bell-Bacterial-spot',
    'Potato-Early-blight',
    'Potato-Late-blight',
    'Squash-Powdery-mildew',
    'Tomato-Bacterial-spot',
    'Tomato-Early-blight',
    'Tomato-Late-blight',
    'Tomato-Leaf-Mold',
    'Tomato-Septoria-leaf-spot',
    'Tomato-Spider-mites-Two-spotted-spider-mite',
    'Tomato-Target-Spot',
    'Tomato-Tomato-Yellow-Leaf-Curl-Virus',
    'Tomato-Tomato-mosaic-virus']

# Función para hacer predicción con el modelo
def predecir_imagen(ruta_imagen):
    img = image.load_img(ruta_imagen, target_size=(256, 256))  # imagen de 256x256
    img_array = image.img_to_array(img)  # Convertir la imagen a un array de píxeles
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra para el batch
    img_array /= 255.0  # Escalar los valores de la imagen entre 0 y 1
    # Realizar la predicción
    prediccion = model.predict(img_array)
    clase_predicha = np.argmax(prediccion)  # Índice de la clase con mayor probabilidad
    return clases[clase_predicha], prediccion[0]  # Retorna la clase predicha y las probabilidades

# Ruta para servir archivos estáticos (imágenes)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def index():
    img_path = None
    prediccion = None
    aviso = ''
    resultado=''
    color=''
    predicho=''
    
    recomendaciones = {
        'Apple-Apple-scab': "La planta presenta mancha de manzana. Aplica fungicidas a base de cobre o productos que contengan azufre en forma de polvo o líquido. Repite cada 10-14 días durante la temporada de crecimiento, especialmente si hay lluvias.",
        'Apple-Black-rot': "La planta tiene óxido negro. Utiliza fungicidas con captan o miclobutanil al inicio de la temporada y después de lluvias fuertes. También retira las hojas o frutos infectados.",
        'Apple-Cedar-apple-rust': "Se observa óxido de manzana. Aplica un fungicida con ingrediente activo de miclobutanil o mancozeb antes de que aparezcan los primeros síntomas y durante la primavera, en intervalos de 7-10 días.",
        'Apple-healthy': "¡La planta de manzana está sana! Sigue cuidándola con riego adecuado y suficiente luz solar.",
        
        'Blueberry-healthy': "¡Los arándanos están sanos! Asegúrate de mantener el suelo bien drenado y proporciona suficiente luz.",
        
        'Cherry-sour-Powdery-mildew': "La planta tiene moho polvoriento. Aplica fungicidas con azufre, bicarbonato de potasio o neem. También puedes usar una solución casera de leche diluida al 10-20% (una parte de leche y 9 de agua) para prevenir.",
        'Cherry-sour-healthy': "¡La planta de cerezo está sana! Mantén un riego regular y asegúrate de que reciba luz adecuada.",
        
        'Corn-Cercospora-leaf-spot-Gray-leaf-spot': "Se presenta mancha de Cercospora. Aplica fungicidas que contengan clorotalonil o cobre y elimina las hojas afectadas para evitar el avance.",
        'Corn-Common-rust': "La planta tiene óxido común. Usa fungicidas que contengan azoxistrobina, propiconazol o miclobutanil y elimina las hojas infectadas para protegerla.",
        'Corn-Northern-Leaf-Blight': "Se observa tizón de hoja norteña. Utiliza fungicidas con clorotalonil, cobre o miclobutanil. Asegúrate de aplicarlos temprano en la temporada y después de lluvias intensas.",
        'Corn-healthy': "¡El maíz está sano! Asegúrate de que reciba suficiente agua y nutrientes.",
        
        'Grape-Black-rot': "La planta presenta óxido negro. Usa fungicidas con captan, cobre o miclobutanil y elimina las partes infectadas para mantenerla sana.",
        'Grape-Esca-Black_Measles': "Se observa Esca. Aplica tratamientos específicos y controla el riego para ayudar a la planta.",
        'Grape-Leaf-blight-Isariopsis': "La planta tiene moho de hoja. Mejora la circulación de aire y aplica fungicidas con clorotalonil o mancozeb para controlarlo.",
        'Grape-healthy': "¡Las uvas están sanas! Mantén un riego regular y proporciona buena luz.",
        
        'Orange-Haunglongbing-Citrus-greening': "La planta presenta Huanglongbing, una enfermedad grave. Busca ayuda profesional para tratarla y controla la plaga de psílidos con insecticidas.",
        
        'Peach-Bacterial-spot': "La planta tiene mancha bacteriana. Aplica tratamientos antibacterianos como bactericidas a base de cobre y mejora la circulación de aire.",
        'Peach-healthy': "¡El durazno está sano! Asegúrate de mantener un riego adecuado y buen drenaje.",
        
        'Pepper-bell-Bacterial-spot': "La planta tiene mancha bacteriana. Aplica tratamientos antibacterianos y elimina las hojas infectadas.",
        'Pepper-bell-healthy': "¡Los pimientos están en buen estado! Sigue con su cuidado regular.",
        
        'Potato-Early-blight': "La planta presenta tizón temprano. Usa fungicidas con clorotalonil o mancozeb y mejora la aireación de las plantas.",
        'Potato-Late-blight': "Se observa tizón tardío. Aplica fungicidas sistémicos como los que contienen clorotalonil, azoxistrobina o fosetil-aluminio y elimina las partes infectadas para proteger la planta.",
        'Potato-healthy': "¡Las papas están sanas! Mantén un riego regular y asegúrate de que el suelo esté bien drenado.",
        
        'Raspberry-healthy': "¡Las frambuesas están sanas! Mantén un riego adecuado y proporciona suficiente luz.",
        
        'Soybean-healthy': "¡Los frijoles de soya están en buen estado! Revisa el drenaje del suelo y el riego.",
        
        'Squash-Powdery-mildew': "La planta tiene moho polvoriento. Controla este problema aplicando fungicidas con azufre, bicarbonato de potasio o neem, y mejorando la circulación de aire.",
        
        'Strawberry-Leaf-scorch': "La planta presenta quemaduras en las hojas. Mejora la gestión del riego para evitar el estrés hídrico.",
        'Strawberry-healthy': "¡Las fresas están sanas! Asegúrate de que reciban suficiente luz solar.",
        
        'Tomato-Bacterial-spot': "La planta tiene mancha bacteriana. Aplica tratamientos antibacterianos y elimina las hojas infectadas.",
        'Tomato-Early-blight': "La planta presenta tizón temprano. Controla este problema aplicando fungicidas con clorotalonil y eliminando hojas infectadas.",
        'Tomato-Late-blight': "Se observa tizón tardío. Aplica fungicidas específicos y elimina las partes infectadas para proteger la planta.",
        'Tomato-Leaf-Mold': "La planta tiene moho en las hojas. Mejora la circulación de aire y aplica fungicidas con clorotalonil o miclobutanil.",
        'Tomato-Septoria-leaf-spot': "La planta presenta manchas de Septoria. Aplica fungicidas a base de cobre o azoxistrobina y elimina las hojas infectadas.",
        'Tomato-Spider-mites-Two-spotted-spider-mite': "La planta tiene ácaros. Controla las plagas aplicando acaricidas a base de aceite de neem, azufre o aceites hortícolas.",
        'Tomato-Target-Spot': "Se observa mancha objetivo. Aplica fungicidas con clorotalonil y mejora el drenaje del suelo.",
        'Tomato-Tomato-Yellow-Leaf-Curl-Virus': "La planta presenta el virus de curvatura amarilla. Busca ayuda profesional, ya que esta enfermedad es grave.",
        'Tomato-Tomato-mosaic-virus': "La planta tiene virus mosaico. Asegúrate de eliminar las plantas afectadas para evitar la propagación.",
        'Tomato-healthy': "¡El tomate está sano! Mantén un riego regular y asegúrate de que el suelo esté bien drenado."
    }


    try:
        # Busca la imagen más reciente en la carpeta uploads
        files = os.listdir(app.config['UPLOAD_FOLDER'])
        files = [f for f in files if f.endswith(('jpg', 'jpeg', 'png'))]
        if files:
            # Selecciona el archivo más reciente
            latest_file = max(files, key=lambda f: os.path.getctime(os.path.join(app.config['UPLOAD_FOLDER'], f)))
            img_path = url_for('uploaded_file', filename=latest_file)  # Genera la URL correcta para la imagen
            
            # Ejecuta el modelo para predecir la clase de la imagen
            prediccion = predecir_imagen(os.path.join(app.config['UPLOAD_FOLDER'], latest_file))
            
            # Obtiene la clase predicha
            predicho, _ = prediccion  # Solo queremos la clase predicha, no las probabilidades

            if predicho in healthy_classes:
                resultado = 'sana'
                color='#a4d3a2'
            elif predicho in unhealthy_classes:
                resultado = 'enferma'
                color=' #f4a460'
            
            # Obtiene la recomendación basada en la predicción
            aviso = recomendaciones.get(predicho, "No hay recomendaciones disponibles.")
                    # Clasificar entre sano y enfermo
            

    except Exception as e:
        print(f"Error al buscar la imagen: {e}")

    return render_template('index.html', img_path=img_path, aviso=aviso, resultado=resultado, color=color, predicho=predicho)

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        # Guarda la imagen con un nombre único (usando timestamp para evitar colisiones)
        filename = datetime.now().strftime('%Y%m%d%H%M%S') + '.jpg'
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
