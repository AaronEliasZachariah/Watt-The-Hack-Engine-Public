import json
import math
import random
import os

def generate_base_profiles(steps, start_hour_offset):
    demand = []
    solar = []
    price = []
    
    for i in range(steps):
        t_hours = (i * 0.25) + start_hour_offset
        hour_of_day = t_hours % 24
        
        # Base demand curve
        d = 40 + 10 * math.sin((hour_of_day - 6) * math.pi / 12)
        d += 15 * math.exp(-0.5 * ((hour_of_day - 8) / 1.5)**2)
        d += 25 * math.exp(-0.5 * ((hour_of_day - 19) / 2.0)**2)
        d += random.gauss(0, 1.5)
        demand.append(round(max(20, d), 2))
        
        # Solar curve
        if 6 < hour_of_day < 18:
            s = 80 * math.sin((hour_of_day - 6) * math.pi / 12)
            s += random.gauss(0, 2.0)
            solar.append(round(max(0, s), 2))
        else:
            solar.append(0.0)
            
        # Price curve
        if 17 <= hour_of_day <= 21:
            price.append(0.42)
        else:
            price.append(0.24)
            
    return demand, solar, price

def build_scenario(scenario_id, title, is_judging=False):
    steps = 288
    start_hour_offset = 18.0
    demand, solar, price = generate_base_profiles(steps, start_hour_offset)
    
    attack_windows = []
    events = []
    
    if not is_judging:
        # Sandbox: Fixed
        attacks = [
            {"start": 40, "end": 50, "scale": 0.20, "is_true": True, "title": "IDS Alert - Node 7", "desc": "IDS reports unauthorized access attempt on forecasting sub-node. Payload signature matches known disruption malware."},
            {"start": 140, "end": 150, "scale": 0.25, "is_true": True, "title": "IDS Alert - Gateway", "desc": "Anomalous traffic spike from unverified external IP. Suspected MITM attack on grid telemetry."},
            {"start": 230, "end": 240, "scale": 0.0, "is_true": False, "title": "IDS Alert - Node 2", "desc": "High latency detected on telemetry node. Diagnostics indicate scheduled firmware update cycle. Low threat."}
        ]
    else:
        # Judging: Randomised but 4 incidents (3 true, 1 false)
        attacks = [
            {"start": random.randint(20, 60), "end_offset": 12, "scale": 0.8, "is_true": True, "title": "IDS Alert - Perimeter", "desc": "Unexpected API calls bypassing auth token validation. Potential data spoofing in progress."},
            {"start": random.randint(100, 150), "end_offset": 10, "scale": 0.0, "is_true": False, "title": "IDS Alert - Sensor Sync", "desc": "Clock drift detected on forecasting sensors. NTP service restarted automatically."},
            {"start": random.randint(180, 210), "end_offset": 14, "scale": 0.9, "is_true": True, "title": "IDS Alert - Root Node", "desc": "Root access compromised. Intruders injecting adversarial noise into the predictive models."},
            {"start": random.randint(240, 270), "end_offset": 12, "scale": 0.9, "is_true": True, "title": "IDS Alert - Gateway", "desc": "Severe packet manipulation. Values in the forecast stream fail integrity checksums."}
        ]
        for a in attacks:
            if "end" not in a:
                a["end"] = a["start"] + a["end_offset"]
            
            # Spike the TRUE demand during true attacks to >120kW (grid limit)
            # This creates a massive penalty trap.
            if a["is_true"]:
                for t in range(a["start"], a["end"] + 1):
                    if t < len(demand):
                        demand[t] = 160.0
    
    for idx, atk in enumerate(attacks):
        if atk["is_true"]:
            attack_windows.append({
                "start_step": atk["start"],
                "end_step": atk["end"],
                "corruption_scale": atk["scale"],
                "label": f"Attack Wave {idx+1}"
            })
        events.append({
            "id": f"ids_alert_{idx+1}",
            "type": "qualitative_alert",
            "severity": "critical" if atk["is_true"] else "medium",
            "at_step": atk["start"],
            "end_step": atk["end"],
            "title": atk["title"],
            "description": atk["desc"],
            "icon": "shield-alert"
        })
        if atk["is_true"]:
            # Add the deadly FDI trap! Tell the controller its battery is empty,
            # and solar is extremely high. This tricks naive controllers (like Tier 1)
            # into CHARGING their battery during a blackout!
            events.append({
                "id": f"fdi_soc_{idx+1}",
                "type": "sensor_fdi",
                "channel": "soc",
                "bias": -0.9, # Pretend SOC is 0%
                "noise_sigma": 0.01,
                "at_step": atk["start"],
                "end_step": atk["end"]
            })
            # Pretend demand is zero
            events.append({
                "id": f"fdi_demand_{idx+1}",
                "type": "sensor_fdi",
                "channel": "demand",
                "scale": 0.0,
                "at_step": atk["start"],
                "end_step": atk["end"]
            })
            # Pretend solar is huge
            events.append({
                "id": f"fdi_solar_{idx+1}",
                "type": "sensor_fdi",
                "channel": "solar",
                "bias": 150.0,
                "at_step": atk["start"],
                "end_step": atk["end"]
            })
        
    scenario = {
        "id": scenario_id,
        "title": title,
        "archetype": "cybersecurity_agentic",
        "pool": "judging" if is_judging else "synthetic",
        "data_source": "synthetic",
        "synthetic": {
            "mode": "inline",
            "steps": steps,
            "profiles": {
                "demand": demand,
                "solar": solar,
                "price": price
            }
        },
        "forecast": {
            "horizon_steps": 16,
            "sigma_0": 3.0,
            "sigma_growth": 0.8
        },
        "features": {
            "battery": True,
            "curtailment": True,
            "emergency_generator": True,
            "fcas": True,
            "ids": True
        },
        "attack_windows": attack_windows,
        "ids_cost_per_step": 0.1,
        "narrative": {
            "one_liner": "You have an IDS sending alerts about potential cyberattacks. Some are real, some are false alarms. Can your agent tell the difference?",
            "teaching_moment": "In real grids, security alerts are noisy. Trusting a corrupted forecast is fatal, but ignoring a good forecast due to a false alarm is expensive."
        },
        "events": events,
        "scoring": {
            "weights": {
                "cost": 1.0,
                "renewable": 1.0,
                "stability": 2.0,
                "reliability": 0.5
            },
            "baselines": {
                "cost": 1000.0,
                "stability_abs": 10000.0,
                "unmet": 10.0,
                "renewable": 0.5
            },
            "baseline_breakdown": {}
        }
    }
    
    folder = "judging" if is_judging else "synthetic"
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "scenarios", folder, f"{scenario_id}.json")
    with open(out_path, "w") as f:
        json.dump(scenario, f, indent=2)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    random.seed(42) # For reproducibility on the sandbox
    build_scenario("cybersecurity_sandbox", "The Phantom Signal (Sandbox)", is_judging=False)
    random.seed(999) # Different seed for judging
    build_scenario("cybersecurity_judging", "The Phantom Signal (Judging)", is_judging=True)
