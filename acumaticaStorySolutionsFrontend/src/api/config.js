// API Configuration
const API_CONFIG = {
  baseURL: (() => {
    // Get the current hostname
    const hostname = (typeof globalThis !== 'undefined' && globalThis.location?.hostname) 
      ? globalThis.location.hostname 
      : 'localhost';
    
    // For local development, try multiple possible backend URLs
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      // Story Solutions backend runs on port 8001
      return 'http://localhost:8001';
    }

    // For non-local environments
    return `http://${hostname}:8001`;
  })(),

  endpoints: {
    health: '/health',
    solutions: {
      process: '/solutions/process',
      health: '/solutions/health'
    }
  },
  defaultHeaders: {
    'Content-Type': 'application/json'
  },
  
  // Request timeout for story processing (can take longer)
  timeout: 600000, // 10 minutes for complex story processing (doubled from 5 minutes)
  
  // No timeout for story processing endpoint (allows backend to complete)
  processTimeout: 0, // 0 = no timeout (unlimited)
};

export default API_CONFIG;

