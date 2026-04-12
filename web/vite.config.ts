import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    watch: {
      ignored: ['**/mano_v1_2/**', '**/for_elise/**', '**/node_modules/**'],
    },
    proxy: {
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/audio': 'http://localhost:8080',
      '/metadata': 'http://localhost:8080',
      '/mano_faces_data': 'http://localhost:8080',
      '/mano_vertices_data': 'http://localhost:8080',
      '/fingering_data': 'http://localhost:8080',
      '/piano_mesh': 'http://localhost:8080',
      '/hitting_points': 'http://localhost:8080',
      '/pieces_metadata': 'http://localhost:8080',
      '/annotation_status': 'http://localhost:8080',
      '/annotation_sources': 'http://localhost:8080',
      '/annotation_progress': 'http://localhost:8080',
      '/motion_issues': 'http://localhost:8080',
      '/post_playing': 'http://localhost:8080',
      '/test_segments': 'http://localhost:8080',
      '/resources': 'http://localhost:8080',
      '/js': 'http://localhost:8080',
      '/css': 'http://localhost:8080',
    },
  },
})
