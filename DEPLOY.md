# Deploy the pricing API (public HTTPS)

This service is containerized (`Dockerfile`) and ships a Render Blueprint (`render.yaml`)
for a **free** public HTTPS URL suitable for the GitHub Pages demo origin
`https://arthursetiawan.github.io`.

## One-shot: Render Blueprint (recommended)

1. Sign in at [https://dashboard.render.com](https://dashboard.render.com) (GitHub OAuth is fine).
2. **New → Blueprint** → select `arthursetiawan/automotive-dynamic-pricing-engine`.
3. Confirm `render.yaml` and create the Blueprint instance (free web service).
4. Wait for the first Docker build + deploy (several minutes).
5. Copy the service URL from the Render dashboard, **without a trailing slash**, e.g.
   `https://automotive-dynamic-pricing-engine.onrender.com`.
6. Smoke-test:

```bash
curl -sS "$API_BASE/health" | jq
curl -sS -X OPTIONS "$API_BASE/optimize-price" \
  -H "Origin: https://arthursetiawan.github.io" \
  -H "Access-Control-Request-Method: POST" -D - -o /dev/null
```

7. Point the Pages demo at that base URL and keep CORS origins as configured
   (`ADPE_CORS_ORIGINS` already includes `https://arthursetiawan.github.io` + localhost).

### Notes

- Free Render web services **sleep after ~15 minutes idle** and take ~1 minute to wake.
- Local filesystem is ephemeral; the committed `artifacts/model.joblib` is baked into the image.
- No API keys are required beyond Render ↔ GitHub OAuth.

## Alternative: manual Render Docker web service

1. **New → Web Service** → connect this repo → Language **Docker**.
2. Plan: **Free**. Health check path: `/health`.
3. Set env vars from `render.yaml` / `.env.example` as needed.
4. Deploy and use `https://<name>.onrender.com` as above.

## Alternative: local Docker

```bash
docker build -t automotive-dynamic-pricing-engine .
docker run --rm -p 8000:8000 automotive-dynamic-pricing-engine
```

## CORS

Default allowed origins (override with comma-separated `ADPE_CORS_ORIGINS`):

- `https://arthursetiawan.github.io`
- `http://localhost:3000`, `http://localhost:5173`, `http://localhost:8000`
- `http://127.0.0.1:3000`, `http://127.0.0.1:5173`, `http://127.0.0.1:8000`
