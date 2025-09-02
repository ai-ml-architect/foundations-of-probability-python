import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import tailwind from '@astrojs/tailwind';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import rehypePrettyCode from 'rehype-pretty-code';

// https://astro.build/config
export default defineConfig({
  integrations: [
    mdx({
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeKatex,
        [
          rehypePrettyCode,
          {
            theme: {
              dark: 'github-dark',
              light: 'github-light',
            },
            keepBackground: false,
          },
        ],
      ],
    }),
    tailwind({
      applyBaseStyles: false,
    }),
  ],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  server: {
    port: 8000,
    host: true,
  },
  vite: {
    server: {
      host: true,
      allowedHosts: ['j7z6hl-8000.csb.app', '.csb.app'],
      hmr: {
        clientPort: 443,
      },
    },
  },
  build: {
    assets: 'assets',
  },
  site: 'https://foundations-probability-python.dev',
});
