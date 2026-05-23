import sys
import os
import requests

# Default to local development server, override this with an env var
# poitning to production server. (e.g. set API_URL=https://your-render-url.com)

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
ADMIN_TOKEN = "hackathon_admin_secret"


def get_status():
    try:
        resp = requests.get(
            f"{API_URL}/admin/scenarios/status", params={"admin_token": ADMIN_TOKEN}
        )
        if resp.status_code == 200:
            data = resp.json()
            print("\n" + "=" * 40)
            print("         SCENARIO STATUS")
            print("=" * 40)

            print(f"\n🟢 UNLOCKED ({len(data.get('unlocked', []))}):")
            for s in data.get("unlocked", []):
                print(f"   - {s}")

            print(f"\n🔴 LOCKED ({len(data.get('locked', []))}):")
            for s in data.get("locked", []):
                print(f"   - {s}")
            print("=" * 40 + "\n")
        else:
            print("Failed to get status:", resp.text)
    except Exception as e:
        print(f"Error connecting to {API_URL}: {e}")


def unlock(scenario_id):
    resp = requests.post(
        f"{API_URL}/admin/scenarios/unlock",
        json={"scenario_id": scenario_id, "admin_token": ADMIN_TOKEN},
    )
    if resp.status_code == 200:
        print(f"✅ Successfully UNLOCKED {scenario_id}!")
    else:
        print(f"❌ Failed to unlock: {resp.text}")


def lock(scenario_id):
    resp = requests.post(
        f"{API_URL}/admin/scenarios/lock",
        json={"scenario_id": scenario_id, "admin_token": ADMIN_TOKEN},
    )
    if resp.status_code == 200:
        print(f"🔒 Successfully LOCKED {scenario_id}!")
    else:
        print(f"❌ Failed to lock: {resp.text}")


def main():
    print(f"Starting Admin Controller for {API_URL}")
    while True:
        get_status()
        print("Commands:")
        print("  unlock <scenario_id>   (e.g., unlock t1_welcome)")
        print("  lock <scenario_id>     (e.g., lock t1_welcome)")
        print("  refresh                (refresh status)")
        print("  exit                   (quit)")

        try:
            cmd = input("admin> ").strip().split()
        except KeyboardInterrupt:
            break

        if not cmd:
            continue

        action = cmd[0].lower()
        if action == "exit":
            break
        elif action == "refresh":
            continue
        elif action == "unlock" and len(cmd) > 1:
            unlock(cmd[1])
        elif action == "lock" and len(cmd) > 1:
            lock(cmd[1])
        else:
            print("Invalid command.")


if __name__ == "__main__":
    main()
