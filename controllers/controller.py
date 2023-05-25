from flask import jsonify, request

from models.models import *

mod_admin= Models()

class controller():

    def  login(self):
        query=mod_admin.login()
        return query
    
    def  enviar(self):
        query=mod_admin.enviar()
        return query
    
    def  getn(self):
        query=mod_admin.getn()
        return query
    
    def  predecir(self):
        query=mod_admin.predecir()
        return query
    
    def  celcius(self):
        query=mod_admin.celcius()
        return query
        
    
    def  eliminar_notificacion(self):
        query=mod_admin.eliminar_notificacion()
        return query
    
    def  getPre(self):
        query=mod_admin.getPre()
        return query
        
        
        
        
    
    