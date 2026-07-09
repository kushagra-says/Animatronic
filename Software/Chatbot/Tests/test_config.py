from config import Config

print("Model:", Config.DEFAULT_MODEL)
print("Host:", Config.OLLAMA_HOST)
print("Temperature:", Config.TEMPERATURE)
print("Top P:", Config.TOP_P)
print("Stream:", Config.STREAM)
print("Data:", Config.DATA_DIR)