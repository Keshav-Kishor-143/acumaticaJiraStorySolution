import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // Bind to all network interfaces (0.0.0.0) to allow access from other machines
    host: '0.0.0.0',
    port: 5173,
    // Enable strict port checking
    strictPort: false,
    // Automatically open browser (optional - can be disabled for network access)
    open: false,
    // CORS is handled by the backend, but we can enable it here too for dev server
    cors: true,
  },
  preview: {
    // Same settings for preview mode (production build preview)
    host: '0.0.0.0',
    port: 4173,
    strictPort: false,
    cors: true,
  },
})
