from flask import Flask, jsonify, request, Blueprint
from flask_cors import  cross_origin
from controllers.controller import *


conexion= controller()



usuarios = Blueprint('usuarios', __name__)

@usuarios.route('/login', methods=['POST'])
@cross_origin()  
def login():
   return conexion.login()


@usuarios.route('/consultar', methods=['GET'])
@cross_origin()
def getAll():
   return conexion.getAll()


@usuarios.route('/alcohol', methods=['GET'])
@cross_origin()
def getAlcohol():
   return conexion.getAlcohol()

@usuarios.route('/quality', methods=['GET'])
@cross_origin()
def getQuality():
   return conexion.getQuality()

@usuarios.route('/predecir', methods=['POST'])
@cross_origin()
def predecir():
   return conexion.predecir()