# Deployment

These steps put WildLens online on free tiers. Each one needs a sign-in that only you can do. Hitting
a free limit pauses the service. None of these steps need a payment card.

## 1. Models on Hugging Face

1. Create a free account at huggingface.co and a token with write access.
2. Run this once the training sequence has finished:

   ```bash
   cd ml
   HF_TOKEN=your-token uv run --with huggingface_hub python export/upload_models.py zac-knight/wildlens-models
   ```

## 2. Database on Neon (optional)

Feedback and cached species cards need this. Without it the app still works, and feedback reports
that it is switched off.

1. Create a free project at neon.com and copy its connection string.
2. Create the tables:

   ```bash
   cd web
   DATABASE_URL="your-connection-string" npx drizzle-kit migrate
   ```

## 3. Website on Vercel

1. Import the GitHub repository at vercel.com and set the **Root Directory** to `web`.
2. Add these environment variables:

| Name | Value | Needed? |
| --- | --- | --- |
| `NEXT_PUBLIC_MODEL_BASE_URL` | `https://huggingface.co/zac-knight/wildlens-models/resolve/main` | Yes |
| `DATABASE_URL` | Neon connection string | For feedback and cached cards |
| `SESSION_HASH_SECRET` | A long random string | For feedback |
| `REVIEW_TOKEN` | A long random string | For the review queue |
| `GOOGLE_GENERATIVE_AI_API_KEY` | Gemini API key | For AI-written profiles |

3. Deploy, then open `/identify` and identify a test photo.

## Free tier limits to watch

- **Vercel Hobby:** Non-commercial use only. 100 GB of transfer, 4 hours of active CPU and 1,000,000
  function calls. Going over can pause the feature for 30 days. Models are not served from Vercel,
  and images skip Vercel's optimiser, to stay well clear of these limits.
- **Neon Free:** 1 GB of storage per project and 100 compute hours a month.
- **OpenStreetMap tiles:** Fair use only, with no offline saving. Move to another tile provider if
  traffic grows.
- **Gemini free tier:** Limits vary by model. Inputs may be used to improve Google's products, so
  WildLens sends only public source text, and it stores cards so that each is generated rarely.

## Reviewing feedback

```bash
curl -H "Authorization: Bearer $REVIEW_TOKEN" https://your-site/api/review
```

Items arrive with the most uncertain first. Send a decision with a POST to the same address, with
the body `{"id": "...", "approved": true, "species": "Dacelo novaeguineae"}`.
