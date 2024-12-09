# Number of worker processes - reduced for memory constraints
workers = 1

# Number of threads per worker - reduced for stability
threads = 2

# Maximum requests before worker restart - reduced to prevent memory leaks
max_requests = 500
max_requests_jitter = 50

# Timeout configuration - increased for model loading
timeout = 300

# Worker class
worker_class = 'gthread'

# Maximum concurrent requests per worker - reduced for stability
worker_connections = 500

# Preload application code
preload_app = True

# Graceful timeout
graceful_timeout = 120

# Keep-alive timeout
keepalive = 5

# Log configuration
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Initialize on worker start
def on_starting(server):
    print("Initializing server...")

def child_exit(server, worker):
    print(f"Worker {worker.pid} exited")