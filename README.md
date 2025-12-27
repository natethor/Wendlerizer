# The Wendlerizer

A web application for generating Jim Wendler's 5/3/1 strength training programs.

## About This Fork

This is a modernized fork of the original Wendlerizer project. The core concept remains the same - helping athletes generate customized 5/3/1 training programs - but the implementation has been rewritten with modern tools and practices:

- **Modern Python**: Type hints, Pydantic models, dataclasses
- **Flask**: Updated web framework and structure
- **Modern Frontend**: Tailwind CSS, HTMX for dynamic interactions
- **Modern Tooling**: uv for dependency management, pyproject.toml for configuration
- **Improved UX**: Responsive design, print-friendly 2x2 weekly workout grids

## Running Locally

This project uses [uv](https://github.com/astral-sh/uv) for fast, modern Python package management.

1. Install uv (if you haven't already):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and uv sync:

   ```bash
   git clone https://github.com/natethor/Wendlerizer.git
   cd Wendlerizer
   uv sync
   ```

3. Run the application:

   ```bash
   uv run wendlerizer
   ```

4. Visit <http://localhost:8080> in your web browser

The app runs in development mode with auto-reload enabled, so changes to the code will automatically restart the server.

## Project Structure

```text
src/
├── main.py              # Flask application and routes
├── models.py            # Data models, validation schemas, and workout patterns
├── utils.py             # Utility functions (weight rounding, 1RM estimation)
└── templates/           # Jinja2 templates
    ├── base.html
    ├── index.html
    └── partials/
        └── program.html
```

## Credits

Original concept and implementation by the Wendlerizer project contributors.
Modernized fork maintained by [natethor](https://github.com/natethor).

Jim Wendler's 5/3/1 program methodology is detailed in his books "5/3/1: The Simplest and Most Effective Training System for Raw Strength" and "Beyond 5/3/1".
