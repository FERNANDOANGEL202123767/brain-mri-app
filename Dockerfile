# Usar una imagen base de Python
FROM python:3.8-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivo de dependencias primero (optimización de caché)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto
EXPOSE 5000

# Comando para ejecutar la aplicación
CMD ["python", "app.py"]