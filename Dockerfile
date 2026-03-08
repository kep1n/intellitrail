# Use the official uv image with Python 3.12 (change version if needed)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Enable uv's bytecode compilation for faster startup
ENV UV_COMPILE_BYTECODE=1

# Copy dependency files first (better layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (no project install yet, just deps)
RUN uv sync --frozen --no-install-project

# Copy the rest of the application
COPY . .

# Install the project itself
RUN uv sync --frozen

# Install Playwright browsers and their OS dependencies
RUN uv run playwright install chromium --with-deps

# Expose the port FastAPI will run on
EXPOSE 8000

# Run with uvicorn via uv (production mode — no --reload)
CMD ["uv", "run", "uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]