---
name: Verdant Expedition
colors:
  surface: '#1b1202'
  surface-dim: '#1b1202'
  surface-bright: '#433821'
  surface-container-lowest: '#150d00'
  surface-container-low: '#241a06'
  surface-container: '#281e0a'
  surface-container-high: '#332813'
  surface-container-highest: '#3f331d'
  on-surface: '#f4e0c0'
  on-surface-variant: '#c3c8bd'
  inverse-surface: '#f4e0c0'
  inverse-on-surface: '#3a2f19'
  outline: '#8d9288'
  outline-variant: '#434840'
  surface-tint: '#b1cfa5'
  primary: '#b1cfa5'
  on-primary: '#1e3619'
  primary-container: '#1a3215'
  on-primary-container: '#7f9c75'
  inverse-primary: '#4b6543'
  secondary: '#e9c349'
  on-secondary: '#3c2f00'
  secondary-container: '#af8d11'
  on-secondary-container: '#342800'
  tertiary: '#f3bb91'
  on-tertiary: '#4a2809'
  tertiary-container: '#452406'
  on-tertiary-container: '#bb8963'
  error: '#ffb4ab'
  on-error: '#690005'
  error-container: '#93000a'
  on-error-container: '#ffdad6'
  primary-fixed: '#cdebc0'
  primary-fixed-dim: '#b1cfa5'
  on-primary-fixed: '#092006'
  on-primary-fixed-variant: '#344d2d'
  secondary-fixed: '#ffe088'
  secondary-fixed-dim: '#e9c349'
  on-secondary-fixed: '#241a00'
  on-secondary-fixed-variant: '#574500'
  tertiary-fixed: '#ffdcc3'
  tertiary-fixed-dim: '#f3bb91'
  on-tertiary-fixed: '#2f1500'
  on-tertiary-fixed-variant: '#643e1e'
  background: '#1b1202'
  on-background: '#f4e0c0'
  surface-variant: '#3f331d'
typography:
  display-lg:
    fontFamily: Literata
    fontSize: 48px
    fontWeight: '900'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Literata
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 40px
  headline-md:
    fontFamily: Literata
    fontSize: 24px
    fontWeight: '700'
    lineHeight: 32px
  body-lg:
    fontFamily: Literata
    fontSize: 18px
    fontWeight: '500'
    lineHeight: 28px
  body-md:
    fontFamily: Literata
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-sm:
    fontFamily: Literata
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  unit: 8px
  container-padding: 24px
  element-gap: 16px
  inner-border-offset: 4px
---

## Brand & Style

This design system is rooted in the **Tactile / Skeuomorphic** high-fantasy adventurer aesthetic. It evokes the feeling of a physical field journal kept by a naturalist in a mystical forest. The UI is designed to feel heavy, weathered, and grounded in nature, using physical metaphors like carved wood, aged parchment, and embossed metals. 

The target audience is users seeking an immersive, "gamified" experience where every interaction feels like a physical discovery. The atmosphere is quiet yet adventurous—combining the academic stillness of a botanist’s study with the rugged life of a forest explorer.

## Colors

The palette is derived from a deep-forest ecosystem, prioritizing organic tones over synthetic ones.

*   **Primary (Forest Deep):** A dense, dark green used for base containers and primary navigation states.
*   **Secondary (Gilded Gold):** A luminous, metallic gold used for highlights, borders, and critical iconography.
*   **Tertiary (Burnt Earth):** A rich chocolate brown used for wood-textured frames and structural dividers.
*   **Parchment Neutral:** A warm, off-white with yellow undertones, serving as the primary background for readable content areas.
*   **Accent Green:** A vibrant, glowing sap green used exclusively for progress bars and active state indicators.

## Typography

The typography system utilizes **Literata** to capture the authoritative yet romantic feel of classical publishing. 

*   **Headlines:** Use high-weight variants with tight letter-spacing. On dark backgrounds, use Gilded Gold; on Parchment, use Burnt Earth.
*   **Body Text:** Prioritizes legibility with increased line-height to mimic book typesetting.
*   **Labels:** Small-caps or all-caps styling is used for secondary metadata and "button-within-button" labels to maintain a structured, scholarly look.

## Layout & Spacing

This design system uses a **Fixed Grid** philosophy. Content is housed within "Panels" that resemble physical objects (journals, wooden crates, stone plaques).

*   **The Journal Model:** The primary content area should follow a two-page spread logic on desktop, reflowing into a single stacked column on mobile.
*   **Margins:** Generous outer margins (32px+) create a sense of the UI "floating" within the forest environment.
*   **Z-Axis Spacing:** Spacing is not just horizontal/vertical but visual. Elements that are "closer" to the user should have larger, softer shadows and brighter highlights.

## Elevation & Depth

Hierarchy is established through **Physical Layering** and **Ambient Shadows**.

*   **Layer 0 (Background):** A blurred forest environment or dark wood grain.
*   **Layer 1 (The Table):** Dark, recessed wooden containers with inner-glow shadows to simulate depth.
*   **Layer 2 (The Journal):** Parchment-textured surfaces that sit atop Layer 1. These use high-offset, low-blur "drop shadows" (Color: `#000000`, Opacity: 40%).
*   **Layer 3 (The Accents):** Embossed metal medallions and glowing status indicators that appear to be physically attached or "inlaid" into the layers below.

## Shapes

Shapes in this design system are organic and "hand-carved."

*   **Outer Containers:** Use a soft (0.25rem) radius but are often overlaid with decorative "corner-pieces" or ornate filigree to break the clean digital line.
*   **Interactive Elements:** Buttons utilize a "pill-serif" hybrid shape—rectangular with heavily rounded ends, often framed by a secondary metallic border.
*   **Borders:** Never use simple 1px lines. Use "carved" borders with a 2-3px width featuring a gradient to simulate a 3D bevel.

## Components

### Buttons
Buttons are tactile "slabs." The default state features a top-down gradient (lighter at top) and a 4px bottom shadow. The "Pressed" state removes the bottom shadow and shifts the element down by 2px to simulate physical depression.

### Cards & Panels
Cards should always feature a parchment texture (`.jpg` or `.png` overlay at low opacity). Borders should be "distressed" or feature ornate corner flourishing in gold or dark wood.

### Input Fields
Inputs are recessed into the parchment, using an `inset` shadow to create a "well" effect. The focus state is indicated by a glowing green inner-shadow.

### Progress Bars
Bars are "hollowed out" wooden channels. The "fill" should be a vibrant, glowing green gradient that looks like liquid or bioluminescent sap.

### Icons
Icons must be enclosed within circular or hexagonal metallic "coins" (medallions). They should never be standalone thin-line glyphs.