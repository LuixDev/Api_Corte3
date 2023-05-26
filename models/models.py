from config import *
from flask import jsonify, request
import jwt
import pickle
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import json
from werkzeug.utils import secure_filename
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart 
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor

from sklearn.utils import shuffle





class Models():
    
    

    
   def login(self):
        data = request.get_json()
        username = data['username']
        password = data['password']

        # Verificar el nombre de usuario y la contraseña en la base de datos
        cursor = connection.cursor()
        cursor.execute("SELECT * FROM login WHERE usuario=%s AND contraseña=%s", (username, password))
        user = cursor.fetchone()

        if not user:
            response = {
                'success': False,
                'message': 'Usuario o contraseña incorrectos'
            }
            return jsonify(response), 401

        # Generar el token JWT
        token_payload = {
            'sub': user[0],
            'exp': datetime.utcnow() + timedelta(days=1)
        }
        token = jwt.encode(token_payload, 'secret', algorithm='HS256')

        response = {
            'success': True,
            'message': 'Login exitoso',
            'token': token
        }
        return jsonify(response), 200
   

   def getn(self):
        cur = connection.cursor()
        cur.execute('SELECT * FROM contacto')
        rv = cur.fetchall()
        payload = []
        content = {}
        for result in rv:
         content = {'id': result[0], 'name': result[1], 'email': result[2], 'message': result[3]}
         payload.append(content)
        
        return jsonify(payload)


   def eliminar_notificacion(self):
     
     data = request.get_json()
     id = data['id']
     responder = data['responder']
     email = data['email']
     cur = connection.cursor()
     cur.execute('DELETE FROM  contacto WHERE id = %s', (id,))
     connection.commit()
     cur.execute('SELECT * FROM contacto')
     cur.fetchall()
     smtp_host = 'smtp-mail.outlook.com'
     smtp_port = 587
     username = 'gwermk@hotmail.com'
     password = 'Ht56848*'

     from_email = 'gwermk@hotmail.com'
     to_email =   email
     subject = 'RegresionLogistica'
     body = responder

     # Crea el objeto MIMEText con el cuerpo del mensaje
     message = MIMEMultipart()
     message['From'] = from_email
     message['To'] = to_email
     message['Subject'] = subject
     message.attach(MIMEText(body, 'plain'))

    # Establece la conexión SMTP y envía el correo electrónico
     with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.send_message(message)
     
     return jsonify("Mensaje enviado")



   def celcius(self):
      data = request.get_json()
      dato = float(data['mensaje'])
      celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
      fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
      oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
      oculta2 = tf.keras.layers.Dense(units=3)
      salida = tf.keras.layers.Dense(units=1)
      modelo = tf.keras.Sequential([oculta1, oculta2, salida])
      modelo.compile(
      optimizer=tf.keras.optimizers.Adam(0.1),
      loss='mean_squared_error'
)
      modelo.fit(celsius, fahrenheit, epochs=800, verbose=False)
      resultado = modelo.predict([dato])
      resultado_list = resultado.tolist()  # Convert ndarray to list
    
      return json.dumps(resultado_list)


  
   def predecir(self):
    archivo = request.files['file']
    text1 = request.form['text1']
    nombre_archivo = secure_filename(archivo.filename)
 
    archivo.save(nombre_archivo)
    try: 
     
     df = pd.read_csv(nombre_archivo, delimiter=',' ,encoding='utf-8')
     X = df.drop(text1, axis=1)
     y = df[text1]
    except:

        df = pd.read_csv(nombre_archivo, delimiter=';' ,encoding='utf-8')
        X = df.drop(text1, axis=1)
        y = df[text1]
  
    X = df.drop(text1, axis=1)
    y = df[text1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    
    rf = RandomForestRegressor(n_estimators=200, random_state=42)

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    score = r2_score(y_test, y_pred)
    y_pred_entero = [int(elemento) for elemento in y_pred]

  
    resultado1 = sum(y_pred_entero) / len(y_pred_entero)
   
    resultados = {
        "----SCORE": score,
        "----MSE": mse,
        "----Predicciones": resultado1
    }
    cursor = connection.cursor()
    cursor.execute("INSERT INTO respuesta  (menssage) VALUES (%s)", (resultado1,))
    connection.commit()
    
   
    return jsonify(resultados)


   def getPre(self):
        cur = connection.cursor()
        cur.execute('SELECT * FROM respuesta')
        rv = cur.fetchall()
        payload = []
        content = {}
        for result in rv:
         content = {'id': result[0], 'menssage': result[1]}
         payload.append(content)
        return jsonify(payload)


   def enviar(self):
        data = request.get_json()
        name = data['name']
        message = data['message']
        email = data['email']

        # Verificar el nombre de usuario y la contraseña en la base de datos
        cursor = connection.cursor()
        cursor.execute("INSERT INTO contacto  (name,message,email) VALUES (%s, %s,%s)", (name,message,email))
        connection.commit()
        
        smtp_host = 'smtp-mail.outlook.com'
        smtp_port = 587
        username = 'gwermk@hotmail.com'
        password = 'Ht56848*'

        # Configura los detalles del correo electrónico
        from_email = 'gwermk@hotmail.com'
        to_email =   email
        subject = 'Gracias por contactarnos'
        body = 'Hemos recibido tu mensajes'

        # Crea el objeto MIMEText con el cuerpo del mensaje
        message = MIMEMultipart()
        message['From'] = from_email
        message['To'] = to_email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        # Establece la conexión SMTP y envía el correo electrónico
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(username, password)
            server.send_message(message)

        print('Correo electrónico enviado')
        response = {
        'success': True,
        'message': 'Dato creado exitosamente'
    }
        
               
            
        
        return jsonify(response)

