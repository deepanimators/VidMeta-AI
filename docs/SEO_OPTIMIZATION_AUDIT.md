# SEO Optimization Audit

## Current behavior

The app asks the LLM to generate titles, descriptions, hashtags, keywords, CTAs, and posting tips for YouTube, Instagram, Facebook, TikTok, and LinkedIn.

## Strengths

- One metadata run covers multiple platforms.
- Prompt includes brand, niche, audience, tone, visual analysis, and transcript.
- Export formats are useful for manual publishing.

## Gaps

- No validation that YouTube titles stay under 60 characters.
- No duplicate hashtag/keyword cleanup.
- No brand-safe terms blocklist.
- No locale selector beyond India-focused prompt text.
- No platform rules engine for length and formatting constraints.

## Recommended improvements

- Add post-processing validators for title length, hashtag count, keyword count, and CTA presence.
- Add a "target country/language" setting.
- Add optional competitor/seed keyword input.
- Add reusable platform constraints in code instead of prompt-only enforcement.
- Add SEO score output per platform with actionable reasons.
