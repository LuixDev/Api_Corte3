from flask import Flask, jsonify, request, Blueprint
from flask_cors import  cross_origin
from controllers.controller import *


conexion= controller()



usuarios = Blueprint('usuarios', __name__)

@usuarios.route('/login', methods=['POST'])
@cross_origin()  
def login():
   return conexion.login()


@usuarios.route('/enviar', methods=['POST'])
@cross_origin()
def enviar():
   return conexion.enviar()

@usuarios.route('/predecir', methods=['POST'])
@cross_origin()
def predecir():
   return conexion.predecir()

@usuarios.route('/celcius', methods=['POST'])
@cross_origin()
def celcius():
   return conexion.celcius()


@usuarios.route('/getn', methods=['GET'])
@cross_origin()
def getn():
   return conexion.getn()

@usuarios.route('/getpre', methods=['GET'])
@cross_origin()
def getPre():
   return conexion.getPre()

@usuarios.route('/eliminar', methods=['POST'])
@cross_origin()
def eliminar_notificacion():
   return conexion.eliminar_notificacion()

