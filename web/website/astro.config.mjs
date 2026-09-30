import { defineConfig } from 'astro/config';

// static output, no adapter, no integrations. the site is files on a disk: it
// cannot cold-start or fall over during a viva demo, and it works offline for
// the screen recording (R17).
//
// no framework integration on purpose - there is no client state here beyond
// filter, sort and theme, and plain <script> islands cover all three. adding
// React would be a dependency carrying no weight.
export default defineConfig({
  output: 'static',
  build: { format: 'directory' },
  // relative links so the build works from a subpath (project pages) or a root
  // domain without a rebuild
  base: process.env.SITE_BASE || '/',
  vite: {
    build: {
      // bundle everything. no CDN at runtime is a hard requirement (brief 3)
      assetsInlineLimit: 4096,
    },
  },
});
