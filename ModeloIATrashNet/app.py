from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
reciclable_info = {
    'cardboard': (True, "El cartón es reciclable y debe colocarse en el contenedor azul."),
    'glass': (True, "El vidrio es reciclable, pero debe estar limpio y sin tapas."),
    'metal': (True, "Los metales son reciclables y se pueden depositar en puntos específicos."),
    'paper': (True, "El papel es reciclable siempre que no esté muy sucio."),
    'plastic': (True, "El plástico es reciclable, pero algunos tipos requieren separación."),
    'trash': (False, "Este material no es reciclable y debe ir a la basura común.")
}

model = tf.keras.models.load_model('modelo_basura.h5')

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert('RGB')
    image = image.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)
    img_array = preprocess_input(img_array)
    predicciones = model.predict(img_array)
    indice = np.argmax(predicciones)
    clase = class_names[indice]
    confianza = float(tf.nn.softmax(predicciones[0])[indice].numpy())
    es_reciclable, mensaje = reciclable_info.get(clase, (False, "No hay información sobre reciclabilidad."))
    return JSONResponse({
        "clase": clase,
        "confianza": confianza,
        "es_reciclable": es_reciclable,
        "mensaje": mensaje
    })