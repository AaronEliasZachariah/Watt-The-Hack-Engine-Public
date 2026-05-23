def rule_based_controller(state: dict) -> dict:
    """
    Template Controller for the Watt The Hack AI Hackathon.

    This function is called by the simulation every 15 minutes (1 timestep).
    Your goal is to return an 'action' dictionary that minimizes costs and avoids blackouts.
    """

    # =========================================================================
    # 1. READ THE STATE (What you know about the world right now)
    # =========================================================================

    # Battery Status
    soc = float(state.get("soc", 0.5))  # State of Charge (0.0 = Empty, 1.0 = Full)

    # City Physics (in kW)
    demand_kw = float(state.get("demand", 0.0))  # How much power the city is consuming
    solar_kw = float(
        state.get("solar", 0.0)
    )  # How much free power the sun is providing

    # Economics ($ per kWh)
    price = float(
        state.get("price", 0.2)
    )  # Current cost to import power from the main grid

    # =========================================================================
    # 2. YOUR AI LOGIC (Decide what to do)
    # =========================================================================

    # Calculate if we have free solar power left over after feeding the city
    solar_surplus_kw = solar_kw - demand_kw

    battery_flow_kw = 0.0
    emergency_generator = 0.0
    curtail_solar = 0.0
    fcas_reserve_kw = 0.0

    # --- Strategy A: Battery Management ---
    # Rule 1: If there is extra solar, and the battery isn't full, charge the battery!
    if solar_surplus_kw > 0.0 and soc < 0.95:
        # Negative value means CHARGE the battery
        battery_flow_kw = -solar_surplus_kw

    # Rule 2: If the city needs power, and grid prices are high, discharge the battery!
    elif solar_surplus_kw < 0.0 and price >= 0.30 and soc > 0.20:
        # Positive value means DISCHARGE the battery to help feed the city
        battery_flow_kw = abs(solar_surplus_kw)

    # Calculate net demand after battery
    net_city_demand = demand_kw - solar_kw - battery_flow_kw

    # --- Strategy B: Overvoltage Management (Too much solar) ---
    # The grid can only absorb 50.0 kW of our exported solar.
    # Exceeding this causes Overvoltage (Massive $5.00/kWh penalty).
    # If we are exporting too much, we must safely disconnect some solar panels.
    if net_city_demand < -50.0:
        curtail_solar = abs(net_city_demand) - 50.0

    # --- Strategy C: Risk Management (The Panic Button) ---
    # The main grid can only supply a maximum of 120.0 kW.
    # If the city pulls more than that, it BLACKS OUT (Massive $10.00/kWh penalty).
    # If the battery can't cover the gap, we must turn on the expensive Diesel Generator.
    elif net_city_demand > 120.0:
        # We are about to overload the grid!
        # Fire up the diesel generator ($1.00/kWh) to cover the exact difference.
        emergency_generator = net_city_demand - 120.0

    # =========================================================================
    # 3. RETURN YOUR ACTION (Tell the engine what to do)
    # =========================================================================

    return {
        # How many kW to push/pull from the battery (Positive = Discharge, Negative = Charge)
        "battery_flow_kw": battery_flow_kw,
        # How many kW to generate from the backup diesel plant (Costs $1.00/kWh to run)
        "emergency_generator": emergency_generator,
        # How many kW of solar power to safely disconnect to avoid frying the grid
        "curtail_solar": curtail_solar,
        # How many kW of inverter capacity to reserve for FCAS revenue
        "fcas_reserve_kw": fcas_reserve_kw,
    }
