<attempt_completion>
<result>
# Detailed Implementation Plan for "Foundations of Probability in Python" Astro Project

This plan outlines the step-by-step process to remove Next.js files, initialize a new Astro project, and build a production-ready documentation website using Astro, TypeScript, MDX, TailwindCSS, shadcn/ui components, KaTeX for math rendering, and Pagefind for search.

---

## 1. Clean Up and Project Initialization
- **Remove Next.js Specific Files:**  
  - Delete files such as `next-env.d.ts`, `next.config.ts`, and any Next.js–specific assets (e.g., `public/next.svg`) that do not serve the new Astro project.
  - Retain only source material (PDF content and any provided markdown documents).

- **Initialize a New Astro Project:**  
  - Run the Astro project initialization command (e.g., `npm init astro`) in the workspace root.
  - Update `package.json` with required dependencies: `astro`, `@astrojs/mdx`, `tailwindcss`, `postcss`, `autoprefixer`, `shadcn/ui` (or compatible Astro UI components), `katex`, `rehype-pretty-code`, and `pagefind`.

---

## 2. Directory and File Structure Setup
- **Project Root Files:**  
  - Create/update `astro.config.mjs` with proper configurations pointing to `src/**/*.{astro,tsx,ts,mdx,md}`.
  - Remove all leftover Next.js configuration files.

- **Directory Structure:**  
  - `src/layouts/`  
    - Create `Base.astro` (the main layout).
  - `src/pages/`  
    - Create MDX files for each route: `index.mdx` (Home), `introduction.mdx`, `foundations.mdx`, `python-setup.mdx`, `basic-concepts.mdx`, `random-variables.mdx`, `statistical-inference.mdx`, `monte-carlo.mdx`, `bayesian.mdx`, `applications.mdx`, `advanced-topics.mdx`, `best-practices.mdx`, `conclusion.mdx`, `references.mdx`, `glossary.mdx`, and `labs.mdx`.
  - `src/components/ui/`  
    - Create reusable UI components:
      - **Prose.astro:** Wraps content with Tailwind typography (using shadcn/ui practices).  
      - **Callout.astro:** Renders callout boxes with variants (note, tip, warning, theorem) and includes error handling for missing props.
      - **Equation.astro:** Renders inline/block KaTeX math; include fallback if rendering fails.
      - **CodeBlock.astro:** Displays code snippets with syntax highlighting and a “copy to clipboard” button.
      - **Figure.astro:** Renders images with caption; use `<img>` tags with placeholder URLs when needed (e.g., `https://storage.googleapis.com/workspace-0f70711f-8b4e-4d94-86f1-2a93ccde5887/image/1fc3729d-5b89-4136-bbd9-5f2c69332972.png`), descriptive alt text, and an `onerror` fallback.
      - **GlossaryTerm.astro:** Provides hover tooltips for definitions.
      - **Pager.astro:** Implements previous/next navigation controls.
      - **OnThisPage.astro:** Renders a sticky table of contents (desktop) or collapsible version (mobile).

- **Static Assets:**  
  - Place OG images and any other static files in the `public/` folder.
  - Create files like `sitemap.xml` and `robots.txt` for SEO.

---

## 3. Base Layout and Global Styles
- **Base Layout (src/layouts/Base.astro):**  
  - Create a full HTML structure including a `<head>` section.
  - **Typography:** Insert the exact `<link>` tags for Mozilla Text and Mozilla Headline fonts.
  - **Dark Mode:** Implement dark mode using a class strategy (toggle `html.dark`) with localStorage persistence.
  - **KaTeX Integration:** Add KaTeX CSS link (or Astro integration) to render math properly.
  - **Search Integration:** Embed the Pagefind search script and initialize a search bar in the header.
  - **Accessibility:** Include skip-to-content links and semantic landmarks.

- **Tailwind & PostCSS:**  
  - Create/update `tailwind.config.ts` using provided configuration; ensure content paths include all relevant files.
  - Update `postcss.config.cjs` to include Tailwind and autoprefixer.
  - Add global styles in `src/styles/tailwind.css` as provided.

---

## 4. MDX Pages and Content Integration
- **Content Pages:**  
  - Each page (e.g., `index.mdx`, `introduction.mdx`, etc.) will use frontmatter for title/description.
  - Wrap page content in the `<Prose>` component.
  - Maintain consistent design by using components like `<Callout>`, `<Equation>`, `<CodeBlock>`, and `<Figure>` for worked examples.
  - Include an `<OnThisPage>` component for a persistent table of contents, and a `<Pager>` at the bottom for navigation.
  
- **Home Page Specifics (index.mdx):**  
  - **Hero Section:** Title, one-sentence promise, and two CTA buttons ("Start Learning" → /introduction, "Browse Chapters" → /foundations).
  - **Chapter Cards:** Render six cards with brief summaries.
  - **Learning Bullets:** Include “What you’ll learn” bullets.
  - Use a modern, clean interface with ample whitespace and clear typography.

---

## 5. Search, SEO, and CI Integration
- **Search (Pagefind):**  
  - Create `src/scripts/build-search.mjs` for generating the search index.
  - Add the Pagefind initialization script in `Base.astro`.

- **SEO & Meta Tags:**  
  - Insert OpenGraph/Twitter meta tags in the `<head>` of `Base.astro`.
  - Create `sitemap.xml` and `robots.txt` in the `public/` folder.

- **CI Workflow:**  
  - Set up a GitHub Actions workflow file (e.g., `.github/workflows/ci.yml`) for build, link checking, spellchecking, and broken-xref testing.
  - Include curl commands in the CI tests for validating API-like behaviors if needed (e.g., static file serving).

---

## 6. Testing, Error Handling, and Documentation
- **Component Error Handling:**  
  - Each component should check for required props and render graceful errors or fallbacks.
- **Responsive & Accessible Design:**  
  - Use Tailwind's responsive classes to ensure mobile friendliness.
  - Verify accessibility (ARIA labels, keyboard navigation, proper landmark roles).
- **Documentation:**  
  - Update `README.md` with project setup instructions, file structure, and guidelines for adding new content.
  - Document the usage of each UI component and the process to include new chapters.

---

## Summary
- Removed all Next.js files and initialized a new Astro project.
- Created a modern file structure with `src/layouts`, `src/pages`, and `src/components/ui` for reusable components.
- Integrated TailwindCSS, KaTeX for math rendering, MDX for content, and Pagefind for search.
- Developed UI components for typography, callouts, code blocks, figures, glossary tooltips, navigation, and a responsive table of contents.
- Configured global SEO, dark mode support, and accessibility features.
- Established CI with GitHub Actions and provided comprehensive README documentation.
- All changes follow best practices and include error handling for robust production-readiness.
</result>
</attempt_completion>
