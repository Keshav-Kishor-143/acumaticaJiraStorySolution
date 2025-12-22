// API Configuration
const API_CONFIG = {
  baseURL: (() => {
    // Get the current hostname from the browser
    const hostname = (typeof globalThis !== 'undefined' && globalThis.location?.hostname) 
      ? globalThis.location.hostname 
      : 'localhost';
    
    // Determine backend URL based on frontend hostname
    // This ensures the frontend connects to the backend on the same machine
    if (hostname === 'localhost' || hostname === '127.0.0.1') {
      // Local development - backend on same machine
      return 'http://localhost:8001';
    } else if (hostname.startsWith('192.168.') || hostname.startsWith('10.') || hostname.startsWith('172.')) {
      // Network IP address (LAN) - backend on same machine IP
      return `http://${hostname}:8001`;
    } else {
      // Production or other environments - use same hostname
      return `http://${hostname}:8001`;
    }
  })(),

  endpoints: {
    health: '/health',
    solutions: {
      process: '/solutions/process',
      health: '/solutions/health',
      manuals: '/solutions/manuals'
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

