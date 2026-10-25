import asyncio
import base64
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

import server


PNG_BYTES = b"\x89PNG\r\n\x1a\nunit-test-image"
DIAGNOSIS = {
    "disease": "Leaf spot",
    "cropDetected": "Tomato",
    "confidence": 91,
    "severity": "Medium",
    "urgency": "Monitor and consult a local specialist.",
    "description": "Test diagnosis.",
    "symptoms": ["Spots"],
    "medicines": ["Follow local guidance."],
    "prevention": ["Inspect plants."],
    "spreadRisk": 40,
    "yieldLoss": 10,
    "soilImpact": 5,
    "treatmentCostPerAcre": 100,
    "lossWithoutTreatment": 200,
    "savingsWithTreatment": 100,
}


class AgriSmartAPITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.original_database_path = server.DATABASE_PATH
        server.DATABASE_PATH = Path(self.temporary_directory.name) / "test-scans.sqlite3"
        self.client = TestClient(server.app)
        self.client.__enter__()
        signup = self.client.post(
            "/api/auth/signup",
            json={"name": "Test Farmer", "email": "farmer@example.com", "password": "strong-pass-123"},
        )
        self.assertEqual(signup.status_code, 200)

    def tearDown(self) -> None:
        self.client.__exit__(None, None, None)
        server.DATABASE_PATH = self.original_database_path
        self.temporary_directory.cleanup()

    def test_analysis_persists_scan_and_image_then_deletes(self) -> None:
        with (
            patch.object(server, "get_ai_credentials", return_value=("gemini", "test-key")),
            patch.object(server, "ask_ai", new=AsyncMock(return_value=json.dumps(DIAGNOSIS))),
        ):
            response = self.client.post(
                "/api/analyze",
                files={"image": ("leaf.png", PNG_BYTES, "image/png")},
                data={"crop_type": "Tomato", "farm_size": "2", "language": "English"},
            )

        self.assertEqual(response.status_code, 200)
        scan_id = response.json()["id"]
        self.assertEqual(response.json()["result"]["_farmSize"], 2)

        history = self.client.get("/api/scans")
        self.assertEqual(history.status_code, 200)
        self.assertEqual(history.json()[0]["id"], scan_id)
        self.assertEqual(history.json()[0]["result"]["disease"], "Leaf spot")
        self.assertEqual(self.client.get(f"/api/scans/{scan_id}/image").content, PNG_BYTES)

        deleted = self.client.delete(f"/api/scans/{scan_id}")
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(self.client.get("/api/scans").json(), [])

    def test_analysis_requires_server_api_key(self) -> None:
        with patch.dict("os.environ", {"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": ""}):
            response = self.client.post(
                "/api/analyze",
                files={"image": ("leaf.png", PNG_BYTES, "image/png")},
                data={"crop_type": "Tomato", "farm_size": "1", "language": "English"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertIn("GEMINI_API_KEY", response.json()["detail"])

    def test_gemini_provider_converts_text_and_images(self) -> None:
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=SimpleNamespace(text="Gemini response"))
        mock_client.aio.aclose = AsyncMock()
        image_data = base64.b64encode(PNG_BYTES).decode("ascii")

        with (
            patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key", "ANTHROPIC_API_KEY": ""}),
            patch.object(server.genai, "Client", return_value=mock_client),
        ):
            response = asyncio.run(
                server.ask_ai(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Inspect this crop."},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_data,
                                    },
                                },
                            ],
                        }
                    ],
                    max_tokens=120,
                    system="Be concise.",
                )
            )

        self.assertEqual(response, "Gemini response")
        call = mock_client.aio.models.generate_content.await_args.kwargs
        self.assertEqual(call["model"], "gemini-2.5-flash")
        self.assertEqual(call["contents"][0].parts[0].text, "Inspect this crop.")
        self.assertEqual(call["contents"][0].parts[1].inline_data.mime_type, "image/png")
        self.assertEqual(call["contents"][0].parts[1].inline_data.data, PNG_BYTES)
        self.assertEqual(call["config"].system_instruction, "Be concise.")
        mock_client.aio.aclose.assert_awaited_once()

    def test_ai_provider_prefers_gemini_and_keeps_anthropic_fallback(self) -> None:
        with patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-key", "ANTHROPIC_API_KEY": "anthropic-key"}):
            self.assertEqual(server.get_ai_credentials(), ("gemini", "gemini-key"))
        with patch.dict("os.environ", {"GEMINI_API_KEY": "", "ANTHROPIC_API_KEY": "anthropic-key"}):
            self.assertEqual(server.get_ai_credentials(), ("anthropic", "anthropic-key"))

    def test_detector_opens_inside_the_dashboard(self) -> None:
        dashboard = self.client.get("/")
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn('id="farm-scan-app"', dashboard.text)
        self.assertNotIn('id="detector-frame"', dashboard.text)

        response = self.client.get("/crop-disease-detector.html", follow_redirects=False)
        self.assertEqual(response.status_code, 307)
        self.assertEqual(response.headers["location"], "/#disease-detection")

        embedded = self.client.get("/detector-embed")
        self.assertEqual(embedded.status_code, 200)
        self.assertIn("FarmScan AI", embedded.text)

    def test_market_prices_are_saved_and_projected_from_observations(self) -> None:
        first = self.client.post(
            "/api/market-prices",
            json={
                "crop": "cotton",
                "market": "Nanded APMC",
                "price_per_quintal": 6000,
                "reported_on": "2026-01-01",
                "source": "APMC notice",
            },
        )
        second = self.client.post(
            "/api/market-prices",
            json={
                "crop": "cotton",
                "market": "Nanded APMC",
                "price_per_quintal": 6300,
                "reported_on": "2026-01-11",
                "source": "APMC notice",
            },
        )

        self.assertEqual(first.status_code, 200)
        summary = second.json()
        self.assertEqual(summary["latest_price"], 6300)
        self.assertEqual(summary["previous_price"], 6000)
        self.assertEqual(summary["forecast_30d"], 7200)
        self.assertEqual(summary["trend"], "up")
        self.assertEqual(len(self.client.get("/api/market-prices?crop=cotton").json()[0]["observations"]), 2)
        with server.database_connection() as connection:
            indexes = {row[1] for row in connection.execute("PRAGMA index_list('market_prices')")}
        self.assertIn("idx_market_prices_crop_reported_on", indexes)

        invalid = self.client.post(
            "/api/market-prices",
            json={
                "crop": "cotton",
                "market": "Nanded APMC",
                "price_per_quintal": 0,
                "reported_on": "2026-01-12",
                "source": "APMC notice",
            },
        )
        self.assertEqual(invalid.status_code, 422)

    def test_marketplace_listings_orders_and_stock_are_persisted(self) -> None:
        products_response = self.client.get("/api/marketplace/products")
        self.assertEqual(products_response.status_code, 200)
        products = products_response.json()
        self.assertEqual(len(products), 8)
        seed_product = products[0]
        stock_before = seed_product["quantity"]

        listing = self.client.post(
            "/api/marketplace/products",
            json={
                "name": "Test Lentils",
                "category": "pulses",
                "price": 88.5,
                "unit": "per kg",
                "quantity": 25,
                "emoji": "🫘",
            },
        )
        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["farmer"], "Test Farmer")

        placed = self.client.post(
            "/api/marketplace/orders",
            json={"items": [{"product_id": seed_product["id"], "quantity": 2}]},
        )
        self.assertEqual(placed.status_code, 200)
        self.assertEqual(placed.json()["total"], round(seed_product["price"] * 2, 2))

        updated_product = next(
            item for item in self.client.get("/api/marketplace/products").json()
            if item["id"] == seed_product["id"]
        )
        self.assertEqual(updated_product["quantity"], stock_before - 2)
        order = self.client.get("/api/marketplace/orders").json()[0]
        self.assertEqual(order["id"], placed.json()["id"])
        self.assertEqual(order["items"][0]["name"], seed_product["name"])
        self.assertEqual(order["items"][0]["unit_price"], seed_product["price"])

        oversell = self.client.post(
            "/api/marketplace/orders",
            json={"items": [{"product_id": seed_product["id"], "quantity": stock_before}]},
        )
        self.assertEqual(oversell.status_code, 409)
        self.assertEqual(self.client.get("/api/marketplace/orders").json()[0]["id"], placed.json()["id"])

    def test_live_market_data_and_notifications_are_available(self) -> None:
        empty_live = self.client.get("/api/market-live").json()
        empty_cotton = next(item for item in empty_live if item["crop"] == "cotton")
        self.assertIsNone(empty_cotton["price_per_quintal"])
        self.assertIsNone(empty_cotton["trend"])
        self.assertIn("No cotton observations", empty_cotton["notification"])

        self.client.post(
            "/api/market-prices",
            json={
                "crop": "cotton",
                "market": "Nanded APMC",
                "price_per_quintal": 6100,
                "reported_on": "2026-01-01",
                "source": "APMC notice",
            },
        )
        single_report = next(
            item for item in self.client.get("/api/market-live").json() if item["crop"] == "cotton"
        )
        self.assertIsNone(single_report["trend"])
        self.assertIn("Add another dated observation", single_report["notification"])
        self.client.post(
            "/api/market-prices",
            json={
                "crop": "cotton",
                "market": "Nanded APMC",
                "price_per_quintal": 6600,
                "reported_on": "2026-01-12",
                "source": "APMC notice",
            },
        )

        live = self.client.get("/api/market-live")
        self.assertEqual(live.status_code, 200)
        self.assertTrue(live.json())
        self.assertIn("notification", live.json()[0])
        self.assertIn("change_percent", live.json()[0])
        cotton = next(item for item in live.json() if item["crop"] == "cotton")
        self.assertEqual(cotton["price_per_quintal"], 6600)
        self.assertEqual(cotton["reported_on"], "2026-01-12")
        self.assertEqual(cotton["source"], "APMC notice")

    def test_auth_logout_and_password_reset_are_one_time(self) -> None:
        duplicate = self.client.post(
            "/api/auth/signup",
            json={"name": "Another Farmer", "email": "farmer@example.com", "password": "strong-pass-123"},
        )
        self.assertEqual(duplicate.status_code, 409)

        reset = self.client.post("/api/auth/forgot-password", json={"email": "farmer@example.com"})
        self.assertEqual(reset.status_code, 200)
        reset_url = reset.json()["reset_url"]
        self.assertIsNotNone(reset_url)
        token = reset_url.split("reset_token=", 1)[1]

        self.assertEqual(self.client.post("/api/auth/logout").status_code, 200)
        self.assertEqual(self.client.get("/api/auth/me").status_code, 401)
        signin = self.client.post(
            "/api/auth/login",
            json={"email": "farmer@example.com", "password": "strong-pass-123"},
        )
        self.assertEqual(signin.status_code, 200)

        changed = self.client.post(
            "/api/auth/reset-password", json={"token": token, "password": "new-strong-pass-456"}
        )
        self.assertEqual(changed.status_code, 200)
        used_again = self.client.post(
            "/api/auth/reset-password", json={"token": token, "password": "another-pass-789"}
        )
        self.assertEqual(used_again.status_code, 400)
        self.client.post("/api/auth/logout")
        signin_new = self.client.post(
            "/api/auth/login",
            json={"email": "farmer@example.com", "password": "new-strong-pass-456"},
        )
        self.assertEqual(signin_new.status_code, 200)

    def test_secure_cookie_can_be_enabled_for_hosted_https(self) -> None:
        with patch.dict("os.environ", {"COOKIE_SECURE": "true"}):
            response = self.client.post(
                "/api/auth/login",
                json={"email": "farmer@example.com", "password": "strong-pass-123"},
            )
        self.assertIn("secure", response.headers["set-cookie"].lower())

    def test_customer_role_cannot_access_farmer_management(self) -> None:
        farmer = self.client.get("/api/auth/me").json()["user"]
        self.assertEqual(farmer["role"], "farmer")

        self.client.post("/api/auth/logout")
        signup = self.client.post(
            "/api/auth/signup",
            json={
                "name": "Market Customer",
                "email": "customer@example.com",
                "password": "strong-pass-123",
                "role": "customer",
            },
        )
        self.assertEqual(signup.status_code, 200)
        self.assertEqual(signup.json()["user"]["role"], "customer")
        self.assertEqual(self.client.get("/api/auth/me").json()["user"]["role"], "customer")
        self.assertEqual(self.client.get("/api/market-live").status_code, 200)
        self.assertEqual(self.client.get("/api/farm/summary").status_code, 403)
        self.assertEqual(self.client.get("/api/farm/crops").status_code, 403)
        self.assertEqual(self.client.get("/api/farm/reminders").status_code, 403)
        self.assertEqual(self.client.get("/api/admin/overview").status_code, 403)
        self.assertEqual(self.client.get("/api/admin/users").status_code, 403)
        self.assertEqual(self.client.get("/api/admin/products").status_code, 403)
        self.assertEqual(self.client.get("/api/admin/orders").status_code, 403)

        field = self.client.post(
            "/api/farm/fields",
            json={
                "name": "Blocked Plot",
                "crop": "Cotton",
                "area_acres": 1,
                "location": "Nanded",
                "status": "Healthy",
            },
        )
        self.assertEqual(field.status_code, 403)

        with server.database_connection() as connection:
            connection.execute("UPDATE users SET role = 'admin' WHERE email = ?", ("customer@example.com",))
        overview = self.client.get("/api/admin/overview")
        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.json()["users_by_role"]["farmer"], 1)

    def test_farm_crop_ledger_saves_lifecycle_and_financial_data(self) -> None:
        response = self.client.post(
            "/api/farm/crops",
            json={
                "crop_name": "Cotton",
                "variety": "Bt Hybrid",
                "sowing_date": "2026-06-15",
                "expected_harvest_date": "2026-11-20",
                "quantity_kg": 1800,
                "expenses": 22500,
                "sales_income": 0,
                "health": "Watch",
                "fertilizer_notes": "NPK applied in split doses",
                "pesticide_notes": "Neem-based spray recorded",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["variety"], "Bt Hybrid")
        self.assertEqual(response.json()["expenses"], 22500)

        records = self.client.get("/api/farm/crops")
        self.assertEqual(records.status_code, 200)
        self.assertEqual(records.json()[0]["crop_name"], "Cotton")
        self.assertEqual(records.json()[0]["expected_harvest_date"], "2026-11-20")
        self.assertEqual(records.json()[0]["fertilizer_notes"], "NPK applied in split doses")
    def test_signup_tracks_language_and_otp_reset_flow(self) -> None:
        signup = self.client.post(
            "/api/auth/signup",
            json={
                "name": "Hindi Farmer",
                "email": "hindi@example.com",
                "password": "strong-pass-123",
                "language": "Hindi",
            },
        )
        self.assertEqual(signup.status_code, 200)
        self.assertEqual(signup.json()["user"]["preferred_language"], "Hindi")

        me = self.client.get("/api/auth/me")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json()["user"]["preferred_language"], "Hindi")

        forgot = self.client.post("/api/auth/forgot-password", json={"email": "hindi@example.com"})
        self.assertEqual(forgot.status_code, 200)
        self.assertIn("otp_code", forgot.json())
        self.assertEqual(len(forgot.json()["otp_code"]), 6)
        self.assertTrue(forgot.json()["otp_code"].isdigit())

    def test_weather_risk_uses_live_data_when_available(self) -> None:
        class FakeResponse:
            def __init__(self, payload):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(self.payload).encode()

        live_payload = {
            "results": [{"name": "Nanded", "latitude": 19.15, "longitude": 77.32, "country": "India"}],
            "current": {
                "temperature_2m": 32,
                "relative_humidity_2m": 76,
                "wind_speed_10m": 18,
                "precipitation": 2.1,
                "weather_code": 2,
            },
        }
        with patch("server.urllib.request.urlopen", return_value=FakeResponse(live_payload)):
            response = self.client.post(
                "/api/weather-risk",
                json={"city": "Nanded"},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["city"], "Nanded")
        self.assertTrue(response.json()["riskLevel"] in {"low", "medium", "high"})

    def test_farm_fields_and_tasks_are_user_specific(self) -> None:
        field = self.client.post(
            "/api/farm/fields",
            json={
                "name": "North Plot",
                "crop": "Cotton",
                "area_acres": 2.5,
                "location": "Nanded",
                "status": "Healthy",
            },
        )
        self.assertEqual(field.status_code, 200)
        field_id = field.json()["id"]

        task = self.client.post(
            "/api/farm/tasks",
            json={
                "field_id": field_id,
                "title": "Irrigate plot",
                "due_date": "2026-10-03",
                "priority": "High",
                "status": "Pending",
                "notes": "Water before 8am.",
            },
        )
        self.assertEqual(task.status_code, 200)

        summary = self.client.get("/api/farm/summary")
        self.assertEqual(summary.status_code, 200)
        self.assertEqual(summary.json()["fields"][0]["name"], "North Plot")
        self.assertEqual(summary.json()["tasks"][0]["title"], "Irrigate plot")

        calendar = self.client.get("/api/farm/calendar")
        self.assertEqual(calendar.status_code, 200)
        self.assertIn("season", calendar.json())
        self.assertTrue(calendar.json()["recommendations"])

    def test_farm_reminders_include_due_tasks_and_harvest_dates(self) -> None:
        today = server.date.today()
        overdue = self.client.post(
            "/api/farm/tasks",
            json={
                "title": "Check north field",
                "due_date": (today - server.timedelta(days=1)).isoformat(),
                "priority": "High",
                "status": "Pending",
            },
        )
        self.assertEqual(overdue.status_code, 200)
        self.client.post(
            "/api/farm/tasks",
            json={
                "title": "Far future task",
                "due_date": (today + server.timedelta(days=15)).isoformat(),
                "priority": "Low",
                "status": "Pending",
            },
        )
        self.client.post(
            "/api/farm/crops",
            json={
                "crop_name": "Cotton",
                "sowing_date": today.isoformat(),
                "expected_harvest_date": (today + server.timedelta(days=5)).isoformat(),
                "quantity_kg": 100,
                "expenses": 0,
                "sales_income": 0,
            },
        )

        reminders = self.client.get("/api/farm/reminders")
        self.assertEqual(reminders.status_code, 200)
        self.assertEqual([item["kind"] for item in reminders.json()], ["task", "harvest"])
        self.assertEqual(reminders.json()[0]["message"], "Overdue")
        self.assertEqual(reminders.json()[1]["message"], "Expected in 5 days")

        completed = self.client.patch(
            f"/api/farm/tasks/{overdue.json()['id']}",
            json={"status": "Done"},
        )
        self.assertEqual(completed.status_code, 200)
        self.assertEqual([item["kind"] for item in self.client.get("/api/farm/reminders").json()], ["harvest"])

    def test_admin_can_manage_users_listings_and_order_lifecycle(self) -> None:
        listing = self.client.post(
            "/api/marketplace/products",
            json={
                "name": "Fresh Onions",
                "category": "vegetables",
                "price": 30,
                "unit": "per kg",
                "quantity": 10,
            },
        )
        self.assertEqual(listing.status_code, 200)

        self.client.post("/api/auth/logout")
        customer = self.client.post(
            "/api/auth/signup",
            json={
                "name": "Order Customer",
                "email": "order-customer@example.com",
                "password": "strong-pass-123",
                "role": "customer",
            },
        )
        self.assertEqual(customer.status_code, 200)
        placed = self.client.post(
            "/api/marketplace/orders",
            json={"items": [{"product_id": listing.json()["id"], "quantity": 2}]},
        )
        self.assertEqual(placed.status_code, 200)
        self.client.post("/api/auth/logout")

        with server.database_connection() as connection:
            connection.execute(
                "UPDATE users SET role = 'admin' WHERE email = ?",
                ("farmer@example.com",),
            )
        admin_login = self.client.post(
            "/api/auth/login",
            json={"email": "farmer@example.com", "password": "strong-pass-123"},
        )
        self.assertEqual(admin_login.status_code, 200)

        users = self.client.get("/api/admin/users")
        self.assertEqual(users.status_code, 200)
        admin_user = next(user for user in users.json() if user["email"] == "farmer@example.com")
        self.assertEqual(
            self.client.patch(
                f"/api/admin/users/{admin_user['id']}/role",
                json={"role": "farmer"},
            ).status_code,
            400,
        )
        customer_user = next(user for user in users.json() if user["email"] == "order-customer@example.com")
        self.assertEqual(
            self.client.patch(
                f"/api/admin/users/{customer_user['id']}/role",
                json={"role": "farmer"},
            ).status_code,
            200,
        )
        self.assertEqual(
            self.client.patch(
                f"/api/admin/users/{customer_user['id']}/role",
                json={"role": "invalid"},
            ).status_code,
            422,
        )

        product_id = listing.json()["id"]
        hidden = self.client.patch(
            f"/api/admin/products/{product_id}",
            json={"active": False},
        )
        self.assertEqual(hidden.status_code, 200)
        self.assertNotIn(product_id, [item["id"] for item in self.client.get("/api/marketplace/products").json()])
        self.assertEqual(
            self.client.post(
                "/api/marketplace/orders",
                json={"items": [{"product_id": product_id, "quantity": 1}]},
            ).status_code,
            404,
        )
        restored = self.client.patch(
            f"/api/admin/products/{product_id}",
            json={"active": True},
        )
        self.assertEqual(restored.status_code, 200)

        order_id = placed.json()["id"]
        confirmed = self.client.patch(
            f"/api/admin/orders/{order_id}",
            json={"status": "confirmed"},
        )
        self.assertEqual(confirmed.status_code, 200)
        cancelled = self.client.patch(
            f"/api/admin/orders/{order_id}",
            json={"status": "cancelled"},
        )
        self.assertEqual(cancelled.status_code, 200)
        self.assertEqual(
            self.client.patch(
                f"/api/admin/orders/{order_id}",
                json={"status": "completed"},
            ).status_code,
            409,
        )
        restored_listing = next(
            item for item in self.client.get("/api/admin/products").json() if item["id"] == product_id
        )
        self.assertEqual(restored_listing["quantity"], 10)

    def test_soil_status_and_irrigation_alerts_are_available(self) -> None:
        field = self.client.post(
            "/api/farm/fields",
            json={
                "name": "South Plot",
                "crop": "Soybean",
                "area_acres": 3,
                "location": "Nanded",
                "status": "Under stress",
            },
        )
        self.assertEqual(field.status_code, 200)

        with patch(
            "server.get_live_weather",
            return_value={
                "temp": 36,
                "humidity": 82,
                "rainfall": 0.4,
                "condition": "Hot",
                "city": "Nanded",
                "riskLevel": "high",
            },
        ):
            soil = self.client.get("/api/farm/soil-status")
        self.assertEqual(soil.status_code, 200)
        self.assertTrue(soil.json())
        self.assertIn("moisture_percent", soil.json()[0])
        self.assertIn("alert_level", soil.json()[0])
        self.assertIn("recommendation", soil.json()[0])


    def test_dashboard_sections_are_composed_once(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        for section_id in (
            "admin-overview",
            "field-form",
            "crop-record-form",
            "price-prediction",
            "irrigation",
            "marketplace",
            "farm-scan-app",
        ):
            self.assertEqual(response.text.count(f'id="{section_id}"'), 1)
        self.assertNotIn("INCLUDE:", response.text)

if __name__ == "__main__":
    unittest.main()
