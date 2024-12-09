# Number of worker processes
workers = 2

# Number of threads per worker
threads = 4

# Maximum requests before worker restart
max_requests = 1000
max_requests_jitter = 50

# Timeout configuration
timeout = 120

# Worker class
worker_class = 'gthread'

# Maximum concurrent requests per worker
worker_connections = 1000

# Preload application code
preload_app = True

# Log configuration
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Initialize on worker start
def on_starting(server):
    print("Initializing server...") 