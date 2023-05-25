from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
from functools import wraps
from config import *
from routes.route import usuarios

key='secret_key_parking'


app = Flask(__name__)
#CORS(app, resources={r"/*": {"origins": "*"}})
cors = CORS(app, resources={r"/*": {"origins": "http://localhost:4200"}})

app.register_blueprint(usuarios)

@app.route('/')
@cross_origin()  
def index():
    return jsonify({'message': 'welcome'})

    
def pagina_no_encontrada(error):
    
    return "<h1>La pagina a la que intentas acceder no existe...</h1>"



if __name__=="__main__":
    app.register_error_handler(404 , pagina_no_encontrada)
    ##app.run(debug=True, host="0.0.0.0")
    app.run(port=5000, debug=True)
