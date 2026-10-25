# AgriSmart MVP

AgriSmart combines the farming dashboard with the FarmScan crop-disease tool. The Python API keeps AI requests on the server and stores scan reports and uploaded images in a local SQLite database.

## Run on Windows

1. Open PowerShell in this folder.
2. Create and activate a virtual environment:

   ```powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and enter your Gemini API key in `GEMINI_API_KEY`. Get a key from Google AI Studio. Keep the key in `.env`; do not commit or share it. Anthropic remains available as a fallback when `GEMINI_API_KEY` is empty.
5. Start the app:

   ```powershell
   python -m uvicorn server:app --reload --host 127.0.0.1 --port 8001
   ```

6. Open <http://127.0.0.1:8001/>.

Do not open the HTML files directly from Explorer. The database-backed features need the FastAPI server, so direct `file://` pages cannot load or save data.

The app also works without an API key, but AI-powered scanning, chat, and weather estimates return a setup message until a key is configured. Gemini is selected when `GEMINI_API_KEY` is set; otherwise the server uses `ANTHROPIC_API_KEY`. Requests go through the local backend; keys are never sent to the browser.

## Deploy a Public Demo on Railway

The frontend and API are served by the same FastAPI service, so Railway provides one public URL; separate frontend hosting and CORS configuration are not needed. `railway.json` configures the Python build, app start command, and `/api/health` health check. SQLite and uploaded scan images must live on a Railway persistent volume because the service filesystem is not durable across deployments.

1. In Railway, create a project and choose **Deploy from GitHub repo**. Connect GitHub if prompted, then select `Vaishu219/New-farm`.
2. In the service's **Volumes** settings, add a persistent volume mounted at `/data`.
3. In the service's **Variables** settings, add:
   - `DATABASE_PATH=/data/farm_scans.sqlite3`
   - `COOKIE_SECURE=true`
   - `ADMIN_EMAIL` and `ADMIN_PASSWORD` with your desired administrator credentials
   - `GEMINI_API_KEY` with your private Gemini API key if you want AI scanning/chat
   - Optionally set `GEMINI_MODEL=gemini-2.5-flash`
4. Deploy the service. `railway.json` supplies the start and health-check settings. In **Networking** / **Settings**, generate a public domain, then open that domain and create a farmer/customer account.
5. Keep the admin credentials available until the app has started successfully and provisioned the admin account. On subsequent starts, an existing account at `ADMIN_EMAIL` is promoted to admin; its password is not reset.

Enter secrets only in Railway's private Variables interface—never commit them to GitHub. A public URL means anyone can access the demo; do not enter sensitive or real farm/customer data. A volume preserves data, but is not a backup. Configure email delivery and review backups, request limits, and deployment security before production use.

## Alternate: Deploy a Public Demo on Render

The frontend and API are served by the same FastAPI service, so the app uses one public URL and does not need separate frontend hosting or CORS configuration. The checked-in `render.yaml` configures a Render web service, HTTPS-only session cookies, and a persistent disk for SQLite and uploaded scan images. The `starter` instance and persistent disk are paid Render resources; do not remove the disk if you need the database and uploads to persist.

1. Push this project to a private or public GitHub repository. Do not include `.env`, API keys, or the local `farm_scans.sqlite3` database.
2. In Render, choose **New > Blueprint**, connect the repository, and deploy using `render.yaml`.
3. When prompted for `GEMINI_API_KEY`, `ADMIN_EMAIL`, and `ADMIN_PASSWORD`, enter values in Render's private environment-variable form. Use a unique, strong admin password and never put these secrets in source control.
4. Wait for the `/api/health` check to pass, then open the `onrender.com` URL shown in the Render dashboard. Create a farmer/customer account and sign in to test the app.
5. Keep the initial admin email/password environment values available until the administrator account is provisioned on first startup. The app creates or promotes that account when the service starts.

Render's public URL makes the demo accessible to anyone. This is a demo deployment, not a production-readiness review: avoid entering real or sensitive farm/customer data, configure email before relying on password resets, and review backups, request limits, and operational/security requirements before production use. A persistent disk preserves SQLite across deploys, but it is not a backup.

## Accounts and Data Freshness

- Public signup supports farmer and customer accounts. Existing accounts are migrated as farmers.
- To provision an administrator, set `ADMIN_EMAIL` and `ADMIN_PASSWORD` in the server environment before startup. The admin account is created on first startup; an existing account at that email is promoted to admin without resetting its password.
- Farmer tools include a multilingual top-right profile panel for farm location and size, crops grown, requirements, preferred follow-up, and need status. Active requests appear on the signed-in farmer needs board; other users can reply privately in the app, farmers can pause or mark requests fulfilled, and replies generate in-app alerts. The marketplace suggests available products based on the farmer's crop and requirements. Email and phone preferences are displayed to responders but contact details are never exposed; email/SMS delivery is not connected. Other tools include field/task planning, season selection for each field and crop cycle (Kharif, Rabi, Zaid, Perennial, or Other), crop suggestions with custom crop names, a crop-cycle ledger with variety, sowing and expected harvest dates, quantity, expenses, sales income, health, and treatment notes, plus soil-status estimates.
- Market alerts can be refreshed manually or automatically every 20 seconds, 1 minute, or 5 minutes. They are calculated from locally entered observations, not a live exchange feed; the dashboard shows the report date and source and never invents missing prices.
- The dashboard includes a real 7-day village/town forecast from Open-Meteo, rule-based heavy-rain, thunderstorm, cold, and low-rain notices, and optional browser speech input for a location. Hail is not specifically forecast, notices are not persistent, and SMS/WhatsApp delivery is not connected. Irrigation status is estimated from weather and crop context, not a physical soil sensor.
- The scheme guide links to official PM-KISAN, PMFBY, KCC, and Soil Health Card resources. Farmers can save a user-entered, officially verified scheme deadline as a farm-planner task; the app does not determine eligibility, submit applications, or maintain live scheme deadlines.
- Quick calculators estimate crop net income, seed/fertilizer quantities from user-entered rates, irrigation water volume, and loan EMI. The crop-cycle ledger remains the farm diary for expenses, sales, harvest quantities, and input notes.
- Marketplace listings, stock quantities, orders, order-item price snapshots, farmer needs, and private need responses are stored in SQLite. Farmers and admins can create listings; signed-in users can place orders and view their own order history. Admins can change account roles, hide or restore listings, and move orders through placed, confirmed, completed, or cancelled states. Cancelling an order restores its reserved stock. Payment, delivery, product images, tracking, and equipment rental are not connected.
- Admin access includes protected platform metrics and tools for managing user roles, marketplace listings, and order status. The last admin cannot be demoted, and admins cannot change their own role.
- Farm task and harvest reminders are generated from saved task due dates and crop harvest dates. The dashboard displays overdue items and those due within 14 days; reminders are not sent by email or push notification. Need-response alerts are in-app only and refresh while the dashboard is open. Tasks can be marked complete from the reminder or task list.

## Current Scope

- Crop scans are analyzed by the configured AI service and saved automatically with the image and report in `farm_scans.sqlite3`.
- Scan history can be viewed and deleted in the detector.
- Market prices can be entered with a mandi, reported date, and source. Records are saved in SQLite; the dashboard calculates a simple 30-day trend only when it has two observations from the same market.
- Marketplace catalog data is seeded once into SQLite; farmer listings and buyer orders persist across sessions. Checkout records intent and updates stock but does not process payment or delivery.
- Admin order cancellation restores listing stock; completed and cancelled orders are final.
- The 7-day forecast is provider data, while the separate disease/weather-risk analysis may fall back to an AI-generated estimate; neither is a field sensor reading or an official warning.
- Irrigation controls and marketplace behavior remain prototype features. They are not connected to physical sensors, payments, or orders.
- The local development server binds to `127.0.0.1`. Add authentication, request limits, and deployment hardening before exposing it to a network.
- Users sign in through `login.html`. Passwords are PBKDF2-hashed, sessions use HttpOnly cookies, and password reset links are one-time and expire after 30 minutes. In local development, the reset link is shown on screen; configure email delivery before production use.

AI crop diagnoses and treatment suggestions can be wrong. Verify important decisions with a qualified local agricultural specialist and follow pesticide labels and local regulations.
