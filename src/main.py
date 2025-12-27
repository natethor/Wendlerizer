"""
Main Flask application for the Wendlerizer.
"""

import os

from flask import Flask, Response, jsonify, render_template, request
from pydantic import ValidationError

from src.models import (
    BarType,
    Lift,
    LiftInput,
    LiftType,
    Units,
    Wendler531Cycle,
)
from src.utils import estimate_1rm

app = Flask(__name__)

# Read secret key from environment or use a default for development
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key-please-change")


@app.route("/")
def index() -> str:
    """Render the main page with the lift input form."""
    return render_template("index.html")


@app.route("/program", methods=["POST"])
def generate_program() -> str | tuple[Response, int]:
    """Generate a 5/3/1 program based on form input."""
    try:
        # Convert form data to proper types
        form_data = {
            "name": request.form.get("name", ""),
            "squat": int(request.form.get("squat", 0)),
            "press": int(request.form.get("press", 0)),
            "bench_press": int(request.form.get("bench_press", 0)),
            "deadlift": int(request.form.get("deadlift", 0)),
            "units": Units(request.form.get("units", "pounds")),
            "bar_type": BarType(request.form.get("bar_type", "standard")),
            "calculate_tms": request.form.get("calculate_tms", "maxes"),
            "light": bool(request.form.get("light", False)),
        }

        # Validate with Pydantic
        data = LiftInput(**form_data)

        # Set barbell weight based on units
        if data.units == Units.KILOGRAMS:
            std_bar = 20.0
            womens_bar = 15.0
        else:
            std_bar = 45.0
            womens_bar = 35.0

        bar_weight = womens_bar if data.bar_type == BarType.WOMENS else std_bar

        # Create main lifts with training max option
        main_lifts = [
            Lift(
                lift_type=LiftType.SQUAT,
                personal_record=data.squat,
                training_max=data.squat if data.calculate_tms == "tmaxes" else None,
                increment=10.0 if not data.light else 5.0,
                barbell_weight=bar_weight,
            ),
            Lift(
                lift_type=LiftType.PRESS,
                personal_record=data.press,
                training_max=data.press if data.calculate_tms == "tmaxes" else None,
                increment=5.0 if not data.light else 2.5,
                barbell_weight=bar_weight,
            ),
            Lift(
                lift_type=LiftType.DEADLIFT,
                personal_record=data.deadlift,
                training_max=data.deadlift if data.calculate_tms == "tmaxes" else None,
                increment=10.0 if not data.light else 5.0,
                barbell_weight=bar_weight,
            ),
            Lift(
                lift_type=LiftType.BENCH_PRESS,
                personal_record=data.bench_press,
                training_max=data.bench_press if data.calculate_tms == "tmaxes" else None,
                increment=5.0 if not data.light else 2.5,
                barbell_weight=bar_weight,
            ),
        ]

        # Create accessory lifts
        accessory_lifts = [
            Lift(LiftType.PULL_UP, personal_record=0),
            Lift(LiftType.DB_ROW, personal_record=0),
            Lift(LiftType.BARBELL_CURL, personal_record=0),
            Lift(LiftType.TRICEP_EXT, personal_record=0),
            Lift(LiftType.CORE, personal_record=0),
        ]

        # Generate program
        cycle = Wendler531Cycle(main_lifts, accessory_lifts)
        program = cycle.generate_cycle()

        # Define lift type groupings for template
        bodyweight_lifts = {LiftType.PULL_UP, LiftType.CORE}
        rpe_lifts = {LiftType.DB_ROW, LiftType.BARBELL_CURL, LiftType.TRICEP_EXT}

        # Return partial template with program
        return render_template(
            "partials/program.html",
            program=program,
            name=data.name,
            units=data.units,  # Already a string due to use_enum_values
            bodyweight_lifts=bodyweight_lifts,
            rpe_lifts=rpe_lifts,
        )

    except ValidationError as e:
        return jsonify({"error": str(e)}), 400
    except ValueError as e:
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"Error generating program: {e}")
        return jsonify({"error": "Failed to generate program"}), 500


@app.route("/estimate-1rm", methods=["POST"])
def calculate_1rm() -> Response | tuple[Response, int]:
    """Estimate 1RM from weight and reps."""
    try:
        weight = float(request.form.get("weight", 0))
        reps = int(request.form.get("reps", 0))

        if weight <= 0 or reps <= 0:
            raise ValueError("Weight and reps must be positive numbers")

        estimated = estimate_1rm(weight, reps)
        response: Response = jsonify({"estimated_1rm": estimated})
        return response

    except ValueError as e:
        return jsonify({"error": str(e)}), 400


def main() -> None:
    """Main entry point for the Wendlerizer application."""
    # Ensure templates auto-reload in development
    app.jinja_env.auto_reload = True
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    # Run the app
    app.run(debug=True, port=8080)


if __name__ == "__main__":
    main()
