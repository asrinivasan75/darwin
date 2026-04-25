/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bark: {
          950: "#0f1311",
          900: "#131815",
          850: "#161b18",
          800: "#1b211d",
          750: "#222923",
          700: "#2a322c",
          600: "#384038",
          500: "#4a5249",
          400: "#6b7166",
          300: "#9aa092",
          200: "#c8c6b6",
          100: "#e8e2d3",
        },
        moss: {
          900: "#1f2a1e",
          800: "#2d3f2a",
          700: "#3f5739",
          600: "#54704a",
          500: "#6b8a5c",
          400: "#88a474",
          300: "#aabd95",
        },
        bronze: {
          800: "#5a4128",
          700: "#7a5a37",
          600: "#9c7647",
          500: "#b58957",
          400: "#c9a876",
          300: "#dcc294",
          200: "#ecdab8",
        },
        ember: {
          700: "#7a3a2a",
          600: "#a85040",
          500: "#c5705f",
        },
      },
      fontFamily: {
        display: ['"Fraunces"', "ui-serif", "Georgia", "serif"],
        sans: ['"Instrument Sans"', "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "Menlo", "monospace"],
      },
      letterSpacing: {
        woodland: "0.18em",
      },
      boxShadow: {
        leaf: "0 1px 0 0 rgba(232,226,211,0.04) inset, 0 18px 40px -28px rgba(0,0,0,0.7), 0 1px 0 0 rgba(0,0,0,0.5)",
      },
    },
  },
  plugins: [],
};
