# GridLock Engine

The simulation engine for the **Watt The Hack / GridLock** energy grid hackathon (DeepNeuron).

This is the public engine package — controllers, scenarios authoring, and the judging server live in private repos. Participants use this package to develop and test their controllers locally before submitting to the hackathon evaluation server.

## Install

```bash
pip install git+https://github.com/AaronEliasZachariah/GridLock-Engine.git
```

## Quick start

```python
from watt_the_hack.engine.engine import Engine
from watt_the_hack.simulation.runner import run_simulation

engine = Engine()
state = {
    "time": 0,
    "demand": 50.0,
    "solar": 0.0,
    "soc": 0.5,
    "price": 0.20,
    "profiles": {"demand": [50.0] * 96, "solar": [0.0] * 96},
    "price_profile": [0.20] * 96,
}

def controller(state):
    return {
        "battery_flow_kw": 0.0,
        "emergency_generator": 0.0,
        "curtail_solar": 0.0,
        "fcas_reserve_kw": 0.0,
    }

result = run_simulation(engine=engine, controller=controller, initial_state=state, steps=96)
print(result["metrics"])
```

## What's in here

- `watt_the_hack/engine/` — physics + market step
- `watt_the_hack/metrics/` — scoring metrics
- `watt_the_hack/simulation/` — runner glue
- `watt_the_hack/controllers/` — reference controllers (rule-based, parametric)
- `watt_the_hack/data_loaders/` — scenario loading utilities

## License

MIT
