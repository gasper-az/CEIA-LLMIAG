import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import openai
import streamlit as st           # Framework para crear aplicaciones web interactivas
from groq import Groq           # Cliente oficial de Groq para acceso a LLMs
import random                   # Para funcionalidades aleatorias (si se necesitan)
import os                      # Para acceso a variables de entorno
from langchain_groq import ChatGroq  

# ================================
# 1. CONFIGURACIÓN INICIAL
# ================================

def configurar_pinecone():
    """
    Configura la conexión con Pinecone usando variables de entorno.
    
    Variables necesarias:
    - PINECONE_API_KEY: Tu clave API de Pinecone
    - PINECONE_ENVIRONMENT: El entorno de Pinecone (ej: 'us-west1-gcp')
    """
    
    # Obtener credenciales desde variables de entorno
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")
    
    if not api_key:
        raise ValueError("PINECONE_API_KEY no está configurada en las variables de entorno")
    
    # Inicializar Pinecone
    pc = Pinecone(
        api_key=os.environ.get("PINECONE_API_KEY")
    )
    
    print(f"✅ Pinecone configurado correctamente en {environment}")
    return True

def configurar_groq():
    try:
        groq_api_key = os.getenv('GROQ_API_KEY')
        model = 'llama-3.1-8b-instant'
		# Verificar si la clave API está configurada
        groq_chat = ChatGroq(
            groq_api_key=groq_api_key,     # Clave API para autenticación
            model_name=model,              # Modelo seleccionado por el usuario
            temperature=0.7,               # Creatividad de las respuestas (0=determinista, 1=creativo)
            max_tokens=1000,               # Máximo número de tokens en la respuesta
        )
        print("✅ Modelo conectado correctamente")
    except Exception as e:
        print(f"❌ Error al conectar con Groq: {str(e)}")

configurar_pinecone()