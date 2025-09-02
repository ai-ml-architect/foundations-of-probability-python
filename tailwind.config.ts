import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}"],
  darkMode: "class",
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Mozilla Text"', "ui-sans-serif", "system-ui", "Inter", "Arial", "sans-serif"],
        display: ['"Mozilla Headline"', '"Mozilla Text"', "ui-sans-serif", "system-ui", "Inter", "Arial", "sans-serif"]
      },
      typography: {
        DEFAULT: {
          css: {
            "--tw-prose-body": "inherit",
            fontFamily: '"Mozilla Text", ui-sans-serif, system-ui, Inter, Arial, sans-serif',
            "h1,h2,h3,h4,h5,h6": { 
              fontFamily: '"Mozilla Headline", "Mozilla Text", ui-sans-serif, system-ui, Inter, Arial, sans-serif',
              fontWeight: '600'
            },
            "code": {
              backgroundColor: "rgb(243 244 246)",
              padding: "0.125rem 0.25rem",
              borderRadius: "0.25rem",
              fontWeight: "400"
            },
            "code::before": {
              content: '""'
            },
            "code::after": {
              content: '""'
            },
            "pre": {
              backgroundColor: "rgb(15 23 42)",
              color: "rgb(226 232 240)"
            },
            "pre code": {
              backgroundColor: "transparent",
              padding: "0"
            }
          }
        },
        invert: {
          css: {
            "code": {
              backgroundColor: "rgb(55 65 81)",
              color: "rgb(229 231 235)"
            }
          }
        }
      },
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;
