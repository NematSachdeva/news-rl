// Single place the frontend learns where the backend API lives.
//
// 1. Same-origin deployment (default local dev and combined deployments):
//    When the frontend is served by the FastAPI backend (e.g. http://localhost:8000/app/,
//    http://127.0.0.1:8000/app/, or a LAN IP), the API base is window.location.origin.
//    This works seamlessly with the CSP header `connect-src 'self'`.
//
// 2. Split-origin deployment (e.g. frontend on Netlify, backend on Render):
//    Override window.NEUROFEED_API_BASE by setting OVERRIDE_API_BASE below or defining
//    window.NEUROFEED_API_BASE before config.js loads.
//    NOTE: In a split-origin deployment, the backend's CSP `connect-src` must allow the
//    backend's origin if connecting across different origins.
(function () {
  // If an external override was defined before this script ran, honor it:
  if (typeof window !== "undefined" && window.NEUROFEED_API_BASE) {
    return;
  }

  // Configurable override for split-deployment setups (e.g. "https://neurofeed-api.onrender.com")
  // Leave empty ("") for same-origin (local dev or combined hosting).
  const OVERRIDE_API_BASE = "";

  if (OVERRIDE_API_BASE) {
    window.NEUROFEED_API_BASE = OVERRIDE_API_BASE.replace(/\/+$/, "");
    return;
  }

  // When served over http/https, use window.location.origin so localhost, 127.0.0.1,
  // LAN IPs, and custom domains all match connect-src 'self' without hardcoding.
  // For file:// or non-HTTP protocols, fallback to empty string.
  if (typeof window !== "undefined" && window.location && window.location.origin && window.location.origin.startsWith("http")) {
    window.NEUROFEED_API_BASE = window.location.origin;
  } else {
    window.NEUROFEED_API_BASE = "";
  }
})();
