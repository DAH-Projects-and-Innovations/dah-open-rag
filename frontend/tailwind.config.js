/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Couleurs pro pour un outil de gestion de données
        brand: {
          primary: '#2563eb', // Bleu moderne
          secondary: '#64748b', // Gris ardoise
        }
      }
    },
  },
  plugins: [],
}