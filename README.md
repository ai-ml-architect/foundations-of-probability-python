# Foundations of Probability in Python

A comprehensive, production-ready documentation website that teaches probability theory through Python implementation. Built with Astro, TypeScript, MDX, and TailwindCSS.

## 🎯 Overview

This project provides an elegant, fast, and accessible site that teaches probability theory through Python, featuring clear structure, beautiful typography, embedded math, runnable code snippets, and real-world examples across finance, healthcare, and engineering domains.

## ✨ Features

- **Modern Tech Stack**: Astro 4.0 + TypeScript + MDX + TailwindCSS
- **Beautiful Typography**: Mozilla Text & Mozilla Headline fonts via Google Fonts
- **Mathematical Content**: KaTeX integration for rendering mathematical equations
- **Interactive Code**: Comprehensive Python examples with syntax highlighting
- **Responsive Design**: Mobile-first design that works across all devices
- **Dark Mode**: Toggle between light and dark themes with localStorage persistence
- **Accessibility**: WCAG-compliant with semantic HTML and proper navigation
- **Fast Performance**: Optimized for speed with static site generation

## 📚 Content Structure

The site contains 15 comprehensive chapters covering:

1. **Introduction** - Motivation, scope, and learning objectives
2. **Mathematical Foundations** - Kolmogorov axioms, probability spaces, Bayes' theorem
3. **Python Environment Setup** - Libraries, tools, and best practices
4. **Basic Probability Concepts** - Sample spaces, events, conditional probability
5. **Random Variables** - Discrete/continuous distributions, moments, transformations
6. **Statistical Inference** - Hypothesis testing, confidence intervals, CLT
7. **Monte Carlo Methods** - Simulation techniques, variance reduction, convergence
8. **Bayesian Probability** - Bayesian inference, MCMC, PyMC implementation
9. **Real-World Applications** - Finance, healthcare, engineering, data science
10. **Advanced Topics** - Machine learning, stochastic processes, modern methods
11. **Best Practices** - Reproducibility, validation, common pitfalls
12. **Conclusion** - Summary and future directions
13. **References** - Comprehensive bibliography and resources
14. **Glossary** - Definitions of key terms and notation
15. **Labs** - Hands-on exercises and practical implementations

## 🚀 Quick Start

### Prerequisites

- Node.js 18+ 
- npm, yarn, pnpm, or bun

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd foundations-of-probability-python

# Install dependencies
npm install

# Start development server
npm run dev
```

The site will be available at `http://localhost:8000`

### Build for Production

```bash
# Build the site
npm run build

# Preview the build
npm run preview
```

## 🛠 Development

### Project Structure

```
├── src/
│   ├── components/ui/          # Reusable UI components
│   │   ├── Prose.astro        # Typography wrapper
│   │   ├── Callout.astro      # Note/tip/warning boxes
│   │   ├── Equation.astro     # Math equation rendering
│   │   ├── Figure.astro       # Image with caption
│   │   ├── Pager.astro        # Previous/next navigation
│   │   └── OnThisPage.astro   # Table of contents
│   ├── layouts/
│   │   └── Base.astro         # Main layout template
│   ├── pages/                 # MDX content pages
│   │   ├── index.mdx          # Homepage
│   │   ├── introduction.mdx   # Chapter pages
│   │   └── ...
│   └── styles/
│       └── tailwind.css       # Global styles
├── public/                    # Static assets
├── astro.config.mjs          # Astro configuration
├── tailwind.config.ts        # Tailwind configuration
└── package.json
```

### Key Components

#### Prose Component
Wraps content with proper typography and spacing:
```astro
<Prose>
  <!-- Your content here -->
</Prose>
```

#### Callout Component
Creates styled callout boxes:
```astro
<Callout variant="note|tip|warning|theorem">
  Your message here
</Callout>
```

#### Equation Component
Renders mathematical equations with KaTeX:
```astro
<Equation>
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
</Equation>
```

#### Figure Component
Displays images with captions:
```astro
<Figure 
  src="https://storage.googleapis.com/workspace-0f70711f-8b4e-4d94-86f1-2a93ccde5887/image/8900a361-5552-4975-aaab-7df2af76991e.png"
  alt="Descriptive alt text"
  caption="Figure caption"
/>
```

#### Pager Component
Provides navigation between pages:
```astro
<Pager 
  prev={{ title: "Previous Page", href: "/previous" }}
  next={{ title: "Next Page", href: "/next" }}
/>
```

### Adding New Content

1. **Create a new MDX file** in `src/pages/`
2. **Add frontmatter** with title and description
3. **Import required components** at the top
4. **Wrap content** in `<Prose>` component
5. **Add navigation** with `<Pager>` component
6. **Include table of contents** with `<OnThisPage />`

Example page structure:
```mdx
---
title: "Your Page Title"
description: "Page description for SEO"
---

import Prose from '../components/ui/Prose.astro';
import Callout from '../components/ui/Callout.astro';
import Pager from '../components/ui/Pager.astro';
import OnThisPage from '../components/ui/OnThisPage.astro';

<Prose>

# Your Page Title

Your content here...

<Callout variant="note">
Important information
</Callout>

<Pager 
  prev={{ title: "Previous", href: "/previous" }}
  next={{ title: "Next", href: "/next" }}
/>

</Prose>

<OnThisPage />
```

## 🎨 Styling and Theming

### Typography

The site uses Mozilla fonts loaded via Google Fonts:
- **Body text**: Mozilla Text (17px, leading-7)
- **Headings**: Mozilla Headline (variable weights 200-700)

### Color Scheme

The site supports both light and dark modes with a clean, academic aesthetic:
- **Light mode**: Clean whites and grays
- **Dark mode**: Dark backgrounds with high contrast text
- **Accent colors**: Subtle blues and greens for interactive elements

### Responsive Design

- **Mobile-first approach** with Tailwind's responsive utilities
- **Breakpoints**: sm (640px), md (768px), lg (1024px), xl (1280px)
- **Content width**: Limited to 70-75ch for optimal readability

## 🧪 Testing

### Manual Testing Checklist

- [ ] All pages load successfully (200 status)
- [ ] Navigation works between all pages
- [ ] Dark mode toggle functions properly
- [ ] Responsive design works on mobile/tablet/desktop
- [ ] Mathematical equations render correctly
- [ ] Code blocks display with proper syntax highlighting
- [ ] Images load with proper fallbacks

### Performance Testing

The site is optimized for:
- **Lighthouse scores ≥95** (performance, accessibility, SEO, best practices)
- **Fast loading times** with static site generation
- **Minimal JavaScript** for better performance

## 📝 Content Guidelines

### Writing Style

Follow the Anthropic Style Guide:
- **Clarity first**: Plain language, short sentences, active voice
- **Trustworthy tone**: Helpful and neutral, avoid hype
- **Structure**: Informative headings, front-load key facts
- **Examples**: Use examples before abstraction
- **Safety**: Note assumptions and limitations

### Code Examples

- **Use Python 3.8+** syntax and features
- **Include imports** at the top of code blocks
- **Add comments** to explain complex logic
- **Provide complete examples** that can be run independently
- **Use realistic data** and meaningful variable names

### Mathematical Content

- **Use KaTeX** for all mathematical notation
- **Define symbols** when first introduced
- **Provide intuitive explanations** alongside formal definitions
- **Include worked examples** for complex concepts

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-content`
3. **Make your changes** following the style guidelines
4. **Test thoroughly** on different devices and browsers
5. **Submit a pull request** with a clear description

### Content Contributions

We welcome contributions of:
- **New examples** and case studies
- **Additional exercises** for the labs section
- **Corrections** and improvements to existing content
- **Translations** to other languages
- **Performance optimizations**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Source Material**: Based on "Foundations of Probability in Python: A Comprehensive Guide" by Manus AI
- **Typography**: Mozilla Text and Mozilla Headline fonts
- **Framework**: Built with Astro and the amazing open-source ecosystem
- **Inspiration**: Academic documentation sites and modern web design principles

## 📞 Support

For questions, issues, or contributions:
- **Open an issue** on GitHub for bugs or feature requests
- **Check the documentation** in the `/references` section
- **Review the glossary** for term definitions
- **Try the hands-on labs** for practical learning

---

**Built with ❤️ for the probability and Python communities**
