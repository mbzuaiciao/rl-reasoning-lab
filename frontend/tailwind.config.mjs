/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          '-apple-system',
          'BlinkMacSystemFont',
          '"SF Pro Text"',
          '"SF Pro Display"',
          'system-ui',
          'sans-serif'
        ],
        mono: [
          '"SF Mono"',
          'ui-monospace',
          'Menlo',
          'Monaco',
          'Consolas',
          'monospace'
        ]
      },
      colors: {
        apple: {
          blue: {
            DEFAULT: '#0071e3',
            dark: '#2997ff',
            subtle: 'rgba(0, 113, 227, 0.08)',
            'subtle-dark': 'rgba(41, 151, 255, 0.12)'
          },
          green: {
            DEFAULT: '#34c759',
            dark: '#30d158',
            subtle: 'rgba(52, 199, 89, 0.08)',
            'subtle-dark': 'rgba(48, 209, 88, 0.12)'
          },
          orange: {
            DEFAULT: '#ff9500',
            dark: '#ff9f0a',
            subtle: 'rgba(255, 149, 0, 0.08)',
            'subtle-dark': 'rgba(255, 159, 10, 0.12)'
          },
          red: {
            DEFAULT: '#ff3b30',
            dark: '#ff453a',
            subtle: 'rgba(255, 59, 48, 0.08)',
            'subtle-dark': 'rgba(255, 69, 58, 0.12)'
          },
          purple: {
            DEFAULT: '#af52de',
            dark: '#bf5af2',
            subtle: 'rgba(175, 82, 222, 0.08)',
            'subtle-dark': 'rgba(191, 90, 242, 0.12)'
          },
          gray: {
            50: '#fbfbfd',
            100: '#f5f5f7',
            200: '#e8e8ed',
            300: '#d2d2d7',
            400: '#86868b',
            500: '#6e6e73',
            600: '#424245',
            700: '#333336',
            800: '#1d1d1f',
            900: '#161618',
            950: '#0a0a0c'
          }
        }
      },
      borderRadius: {
        'xl': '12px',
        '2xl': '16px',
        '3xl': '22px',
        '4xl': '28px'
      },
      boxShadow: {
        'subtle': '0 1px 2px rgba(0, 0, 0, 0.04), 0 2px 6px rgba(0, 0, 0, 0.02)',
        'card': '0 2px 8px rgba(0, 0, 0, 0.03), 0 10px 24px rgba(0, 0, 0, 0.03)',
        'elevated': '0 4px 14px rgba(0, 0, 0, 0.06), 0 14px 40px rgba(0, 0, 0, 0.06)',
        'dark-card': '0 2px 8px rgba(0, 0, 0, 0.4), 0 10px 24px rgba(0, 0, 0, 0.3)'
      }
    }
  },
  plugins: []
};
