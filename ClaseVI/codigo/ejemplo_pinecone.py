"""
Ejemplo de uso de Pinecone Vector Database
==========================================

Este archivo demuestra cómo:
1. Crear un índice en Pinecone
2. Poblar el índice con vectores y metadatos
3. Realizar búsquedas por similitud

Autor: Clase VI - CEIA LLMIAG
Documentación en español
"""

import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import pinecone
from pinecone import Pinecone, PodSpec
from sentence_transformers import SentenceTransformer
import openai


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
        api_key=api_key,
        environment=environment
    )
    
    print(f"✅ Pinecone configurado correctamente en {environment}")
    return pc


def crear_indice(pinecone_client: Pinecone, nombre_indice: str, dimension: int = 384, metrica: str = "cosine"):
    """
    Crea un nuevo índice en Pinecone.
    
    Args:
        nombre_indice (str): Nombre del índice a crear
        dimension (int): Dimensión de los vectores (depende del modelo de embedding)
        metrica (str): Métrica de similitud ('cosine', 'euclidean', 'dotproduct')
    
    Configuración de infraestructura:
        - Pods: Unidades de cómputo paralelo que procesan las consultas
          • 1 pod = suficiente para desarrollo y proyectos pequeños
          • Más pods = mayor capacidad de consultas simultáneas pero mayor costo
        
        - Réplicas: Copias idénticas del índice distribuidas geográficamente
          • 1 réplica = configuración básica
          • Más réplicas = mayor disponibilidad y tolerancia a fallos
        
        - Tipos de pod disponibles:
          • p1.x1: 1 vCPU, ~5GB RAM (plan gratuito/starter)
          • p1.x2: 2 vCPU, ~10GB RAM
          • p1.x4: 4 vCPU, ~20GB RAM
          • p2.x1: Optimizado para performance
    
    Returns:
        bool: True si se creó exitosamente
    """
    
    # Verificar si el índice ya existe
    # indices_existentes = pinecone.list_indexes()
    indices_existentes = pinecone_client.list_indexes()
    
    if nombre_indice in indices_existentes:
        print(f"⚠️  El índice '{nombre_indice}' ya existe")
        return True
    
    podSpec = PodSpec(pods=1, replicas=1, pod_type="p1.x1", environment="us-west1-gcp")

    # Crear el índice
    pinecone_client.create_index(
        name=nombre_indice,
        dimension=dimension,
        metric=metrica,
        spec=podSpec
        # pods=1,  # Pods: Unidades de cómputo que procesan queries. Más pods = mayor throughput pero mayor costo
        # replicas=1,  # Réplicas: Copias del índice para alta disponibilidad. Más réplicas = mayor disponibilidad
        # pod_type="p1.x1"  # Tipo de pod: p1.x1 (gratuito, 1 vCPU), p1.x2 (2 vCPU), p2.x1 (optimizado), etc.
    )
    
    # Esperar a que el índice esté listo
    print(f"🔄 Creando índice '{nombre_indice}'...")
    while nombre_indice not in pinecone_client.list_indexes():
        time.sleep(1)
    
    print(f"✅ Índice '{nombre_indice}' creado exitosamente")
    return True


# ================================
# 2. GENERACIÓN DE EMBEDDINGS
# ================================

class GeneradorEmbeddings:
    """
    Clase para generar embeddings usando diferentes modelos.
    """
    
    def __init__(self, modelo: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Inicializa el generador de embeddings.
        
        Args:
            modelo (str): Nombre del modelo de Sentence Transformers
        """
        self.modelo_nombre = modelo
        self.modelo = SentenceTransformer(modelo)
        self.dimension = self.modelo.get_sentence_embedding_dimension()
        
        print(f"✅ Modelo '{modelo}' cargado (dimensión: {self.dimension})")
    
    def generar_embedding(self, texto: str) -> List[float]:
        """
        Genera embedding para un texto individual.
        
        Args:
            texto (str): Texto a convertir en embedding
            
        Returns:
            List[float]: Vector de embedding
        """
        embedding = self.modelo.encode(texto)
        return embedding.tolist()
    
    def generar_embeddings_lote(self, textos: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos de manera eficiente.
        
        Args:
            textos (List[str]): Lista de textos
            
        Returns:
            List[List[float]]: Lista de vectores de embedding
        """
        embeddings = self.modelo.encode(textos)
        return [emb.tolist() for emb in embeddings]


# ================================
# 3. POBLACIÓN DEL ÍNDICE
# ================================

def poblar_indice_ejemplo(pinecone_client: Pinecone, nombre_indice: str, generador: GeneradorEmbeddings):
    """
    Puebla el índice con datos de ejemplo.
    
    Args:
        nombre_indice (str): Nombre del índice de Pinecone
        generador (GeneradorEmbeddings): Instancia del generador de embeddings
    """
    
    # Conectar al índice
    indice = pinecone_client.Index(nombre_indice)
    
    # Datos de ejemplo: documentos sobre inteligencia artificial
    documentos_ejemplo = [
        {
            "id": "doc_001",
            "texto": "Nombre: Gaspar | Apellido: Acevedo Zain | Fecha Nacimiento: 19/01/1993 | Experiencia 1: Lugar de trabajo 1, realizando tareas de desarollo con C# | Experiencia 2: Lugar de trabajo 2, realizando tareas de desarollo con Java | Experiencia 3: Lugar de trabajo 3, realizando tareas de desarollo con Javascript, C#, SQL.",
            "categoria": "definicion",
            "fecha": "2024-01-15"
        },
        {
            "id": "doc_002", 
            "texto": "Nombre: Carlos | Apellido: Villalobos | Fecha Nacimiento: 19/01/1985 | Experiencia 1: Lugar de trabajo 4, realizando tareas de desarollo con SQL, Python, VectorDB. | Experiencia 2: Lugar de trabajo 2, realizando tareas de desarollo con LLMs, Pytorch. | Experiencia 3: Lugar de trabajo 3, realizando tareas de desarollo con Kubernetes, Docker, LLMs, Scikit-learn.",
            "categoria": "modelos",
            "fecha": "2024-01-16"
        },
    ]
    
    print(f"🔄 Poblando índice con {len(documentos_ejemplo)} documentos...")
    
    # Generar embeddings para todos los textos
    textos = [doc["texto"] for doc in documentos_ejemplo]
    embeddings = generador.generar_embeddings_lote(textos)
    
    # Preparar datos para inserción en lotes
    vectors_para_insertar = []
    
    for i, doc in enumerate(documentos_ejemplo):
        vector_data = {
            "id": doc["id"],
            "values": embeddings[i],
            "metadata": {
                "texto": doc["texto"],
                "categoria": doc["categoria"],
                "fecha": doc["fecha"],
                "longitud": len(doc["texto"])
            }
        }
        vectors_para_insertar.append(vector_data)
    
    # Insertar vectores en el índice
    indice.upsert(vectors=vectors_para_insertar)
    
    # Verificar estadísticas del índice
    estadisticas = indice.describe_index_stats()
    print(f"✅ Índice poblado exitosamente")
    print(f"   📊 Total de vectores: {estadisticas['total_vector_count']}")
    print(f"   📏 Dimensión: {estadisticas['dimension']}")
    
    return True


# ================================
# 4. BÚSQUEDAS EN EL ÍNDICE
# ================================

def buscar_documentos_similares(
        pinecone_client: Pinecone,
    nombre_indice: str, 
    consulta: str, 
    generador: GeneradorEmbeddings,
    top_k: int = 3,
    filtro_metadata: Dict = None
) -> List[Dict[str, Any]]:
    """
    Realiza una búsqueda por similitud en el índice.
    
    Args:
        nombre_indice (str): Nombre del índice de Pinecone
        consulta (str): Texto de consulta para buscar
        generador (GeneradorEmbeddings): Generador de embeddings
        top_k (int): Número de resultados más similares a devolver
        filtro_metadata (Dict): Filtros opcionales por metadata
        
    Returns:
        List[Dict]: Lista de documentos similares con scores
    """
    
    # Conectar al índice
    indice = pinecone_client.Index(nombre_indice)
    
    # Generar embedding para la consulta
    print(f"🔍 Buscando documentos similares a: '{consulta}'")
    embedding_consulta = generador.generar_embedding(consulta)
    
    # Realizar la búsqueda
    resultados = indice.query(
        vector=embedding_consulta,
        top_k=top_k,
        include_metadata=True,
        filter=filtro_metadata
    )
    
    # Procesar y formatear resultados
    documentos_encontrados = []
    
    print(f"\n📋 Resultados encontrados ({len(resultados['matches'])}):")
    print("=" * 80)
    
    for i, match in enumerate(resultados['matches'], 1):
        documento = {
            "posicion": i,
            "id": match["id"],
            "score": round(match["score"], 4),
            "texto": match["metadata"]["texto"],
            "categoria": match["metadata"]["categoria"],
            "fecha": match["metadata"]["fecha"]
        }
        
        documentos_encontrados.append(documento)
        
        # Mostrar resultado formateado
        print(f"{i}. ID: {documento['id']}")
        print(f"   📊 Score: {documento['score']}")
        print(f"   🏷️  Categoría: {documento['categoria']}")
        print(f"   📅 Fecha: {documento['fecha']}")
        print(f"   📝 Texto: {documento['texto'][:100]}...")
        print("-" * 80)
    
    return documentos_encontrados


def buscar_con_filtros_ejemplo(pinecone_client: Pinecone, nombre_indice: str, generador: GeneradorEmbeddings):
    """
    Demuestra búsquedas con filtros de metadata.
    
    Args:
        nombre_indice (str): Nombre del índice
        generador (GeneradorEmbeddings): Generador de embeddings
    """
    
    print("\n🔍 EJEMPLO DE BÚSQUEDAS CON FILTROS")
    print("=" * 50)
    
    # Búsqueda 1: Sin filtros
    print("\n1️⃣ Lugares de trabajo de Carlos':")
    buscar_documentos_similares(
        pinecone_client,
        nombre_indice, 
        "Donde trabajó Carlos?", 
        generador,
        top_k=3
    )
    
    # Búsqueda 2: Con filtro por categoría
    print("\n2️⃣ Experiencia laboral de Gaspar:")
    filtro_categoria = {"categoria": {"$eq": "modelos"}}
    buscar_documentos_similares(
        pinecone_client,
        nombre_indice,
        "En qué tiene experiencia Gaspar?",
        generador,
        top_k=2,
        filtro_metadata=filtro_categoria
    )
    
    # Búsqueda 3: Con filtro por fecha
    print("\n3️⃣ Coincidieron en algún lugar de trabajo Carlos y Gaspar?:")
    filtro_fecha = {"fecha": {"$gte": "2024-01-17"}}
    buscar_documentos_similares(
        pinecone_client,
        nombre_indice,
        "Coincidieron en algún lugar de trabajo Carlos y Gaspar?",
        generador,
        top_k=5,
        filtro_metadata=filtro_fecha
    )


# ================================
# 5. GESTIÓN DEL ÍNDICE
# ================================

def obtener_estadisticas_indice(pinecone_client: Pinecone, nombre_indice: str):
    """
    Muestra estadísticas detalladas del índice.
    
    Args:
        nombre_indice (str): Nombre del índice
    """
    
    indice = pinecone_client.Index(nombre_indice)
    estadisticas = indice.describe_index_stats()
    
    print(f"\n📊 ESTADÍSTICAS DEL ÍNDICE '{nombre_indice}'")
    print("=" * 50)
    print(f"📦 Total de vectores: {estadisticas.get('total_vector_count', 0)}")
    print(f"📏 Dimensión: {estadisticas.get('dimension', 0)}")
    
    # Mostrar estadísticas por namespace si existen
    if 'namespaces' in estadisticas:
        print(f"🏷️  Namespaces:")
        for namespace, stats in estadisticas['namespaces'].items():
            print(f"   - {namespace}: {stats.get('vector_count', 0)} vectores")


def eliminar_documentos(nombre_indice: str, ids_documentos: List[str]):
    """
    Elimina documentos específicos del índice.
    
    Args:
        nombre_indice (str): Nombre del índice
        ids_documentos (List[str]): Lista de IDs a eliminar
    """
    
    indice = pinecone.Index(nombre_indice)
    
    print(f"🗑️  Eliminando {len(ids_documentos)} documentos...")
    indice.delete(ids=ids_documentos)
    
    print(f"✅ Documentos eliminados: {', '.join(ids_documentos)}")


def limpiar_indice_completo(nombre_indice: str):
    """
    Elimina todos los vectores del índice.
    
    Args:
        nombre_indice (str): Nombre del índice a limpiar
    """
    
    indice = pinecone.Index(nombre_indice)
    
    print(f"🧹 Limpiando índice '{nombre_indice}' completamente...")
    indice.delete(delete_all=True)
    
    print("✅ Índice limpiado exitosamente")


def eliminar_indice(nombre_indice: str):
    """
    Elimina completamente un índice de Pinecone.
    
    Args:
        nombre_indice (str): Nombre del índice a eliminar
    """
    
    print(f"🗑️  Eliminando índice '{nombre_indice}'...")
    pinecone.delete_index(nombre_indice)
    
    print(f"✅ Índice '{nombre_indice}' eliminado exitosamente")


# ================================
# 6. FUNCIÓN PRINCIPAL DE EJEMPLO
# ================================

def ejecutar_ejemplo_completo():
    """
    Ejecuta un ejemplo completo de uso de Pinecone:
    1. Configuración
    2. Creación del índice
    3. Población con datos
    4. Búsquedas de ejemplo
    5. Limpieza (opcional)
    """
    
    # Configuración
    nombre_indice = "ejemplo-ceia-llmiag"
    
    try:
        print("🚀 INICIANDO EJEMPLO DE PINECONE")
        print("=" * 50)
        
        # 1. Configurar conexión
        pc = configurar_pinecone()
        
        # 2. Inicializar generador de embeddings
        generador = GeneradorEmbeddings()
        
        # 3. Crear índice
        crear_indice(pc, nombre_indice, dimension=generador.dimension)
        
        # 4. Poblar índice con datos de ejemplo
        poblar_indice_ejemplo(pc, nombre_indice, generador)
        
        # 5. Mostrar estadísticas
        obtener_estadisticas_indice(pc, nombre_indice)
        
        # 6. Realizar búsquedas de ejemplo
        buscar_con_filtros_ejemplo(pc, nombre_indice, generador)
        
        # 7. Búsqueda personalizada
        print("\n🎯 BÚSQUEDA PERSONALIZADA")
        print("=" * 30)
        consulta_personalizada = "¿Qué son las redes neuronales?"
        resultados = buscar_documentos_similares(
            pc,
            nombre_indice, 
            consulta_personalizada, 
            generador,
            top_k=2
        )
        
        print(f"\n✅ EJEMPLO COMPLETADO EXITOSAMENTE")
        print(f"📁 Índice '{nombre_indice}' está listo para usar")
        
        # Opcional: Comentar la siguiente línea si quieres mantener el índice
        # eliminar_indice(nombre_indice)
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        raise


# ================================
# 7. UTILIDADES ADICIONALES
# ================================

def crear_indice_desde_documentos(
    nombre_indice: str,
    documentos: List[Dict[str, Any]],
    campo_texto: str = "texto",
    modelo_embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Función helper para crear un índice directamente desde una lista de documentos.
    
    Args:
        nombre_indice (str): Nombre del índice a crear
        documentos (List[Dict]): Lista de documentos con texto y metadata
        campo_texto (str): Nombre del campo que contiene el texto
        modelo_embedding (str): Modelo para generar embeddings
    
    Returns:
        bool: True si se creó exitosamente
    """
    
    # Configurar y crear componentes
    configurar_pinecone()
    generador = GeneradorEmbeddings(modelo_embedding)
    crear_indice(nombre_indice, dimension=generador.dimension)
    
    # Conectar al índice
    indice = pinecone.Index(nombre_indice)
    
    # Procesar documentos en lotes
    lote_size = 100
    total_procesados = 0
    
    print(f"🔄 Procesando {len(documentos)} documentos en lotes de {lote_size}...")
    
    for i in range(0, len(documentos), lote_size):
        lote_docs = documentos[i:i + lote_size]
        
        # Generar embeddings para el lote
        textos_lote = [doc[campo_texto] for doc in lote_docs]
        embeddings_lote = generador.generar_embeddings_lote(textos_lote)
        
        # Preparar vectores para inserción
        vectores_lote = []
        for j, doc in enumerate(lote_docs):
            # Generar ID si no existe
            doc_id = doc.get("id", f"doc_{i+j:06d}")
            
            # Preparar metadata (excluir el texto para evitar duplicación)
            metadata = {k: v for k, v in doc.items() if k != campo_texto and k != "id"}
            metadata["texto"] = doc[campo_texto]  # Incluir texto en metadata
            
            vector_data = {
                "id": doc_id,
                "values": embeddings_lote[j],
                "metadata": metadata
            }
            vectores_lote.append(vector_data)
        
        # Insertar lote
        indice.upsert(vectors=vectores_lote)
        total_procesados += len(lote_docs)
        
        print(f"   ✅ Procesados {total_procesados}/{len(documentos)} documentos")
    
    print(f"🎉 Índice '{nombre_indice}' creado con {total_procesados} documentos")
    return True


if __name__ == "__main__":
    """
    Para ejecutar este ejemplo, asegúrate de:
    
    1. Instalar las dependencias:
       pip install pinecone-client sentence-transformers openai numpy
    
    2. Configurar variables de entorno:
       export PINECONE_API_KEY="tu-api-key-aqui"
       export PINECONE_ENVIRONMENT="us-west1-gcp"  # o tu región
    
    3. Ejecutar el script:
       python ejemplo_pinecone.py
    """
    
    # Verificar si las variables de entorno están configuradas
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ PINECONE_API_KEY no está configurada")
        print("💡 Configura tus variables de entorno antes de ejecutar:")
        print("   export PINECONE_API_KEY='tu-api-key'")
        print("   export PINECONE_ENVIRONMENT='tu-region'")
        exit(1)
    
    # Ejecutar ejemplo completo
    ejecutar_ejemplo_completo()