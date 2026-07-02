# Mycelix-Core Development Container
# Provides complete environment for FL, Holochain, and Smart Contract development

FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    pkg-config \
    libssl-dev \
    libclang-dev \
    clang \
    cmake \
    protobuf-compiler \
    libprotobuf-dev \
    git \
    curl \
    wget \
    jq \
    htop \
    tmux \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Holochain toolchain
USER vscode
ENV PATH="/home/vscode/.cargo/bin:/home/vscode/.nix-profile/bin:${PATH}"

# Install Nix (for Holochain)
RUN curl -L https://nixos.org/nix/install | sh -s -- --no-daemon

# Configure Nix for Holochain
RUN mkdir -p ~/.config/nix && \
    echo "experimental-features = nix-command flakes" >> ~/.config/nix/nix.conf

# Install Foundry (for Solidity development)
RUN curl -L https://foundry.paradigm.xyz | bash && \
    ~/.foundry/bin/foundryup

ENV PATH="/home/vscode/.foundry/bin:${PATH}"

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/home/vscode/.local/bin:${PATH}"

# Pre-configure Rust components
RUN rustup component add rustfmt clippy rust-analyzer && \
    rustup target add wasm32-unknown-unknown

# Install cargo tools
RUN cargo install cargo-watch cargo-nextest cargo-expand cargo-audit

# Install Holochain CLI tools (via cargo for now, nix is more reliable but slower)
RUN cargo install holochain_cli --locked || true

# Set up workspace directory
WORKDIR /workspaces/Mycelix-Core

# Create helpful aliases
RUN echo 'alias build="cargo build --release"' >> ~/.bashrc && \
    echo 'alias test="cargo nextest run"' >> ~/.bashrc && \
    echo 'alias demo="python -m mycelix_cli demo"' >> ~/.bashrc && \
    echo 'alias fl-demo="cargo run --bin fl-demo --release"' >> ~/.bashrc && \
    echo 'alias watch="cargo watch -x check"' >> ~/.bashrc && \
    echo '' >> ~/.bashrc && \
    echo '# Mycelix Welcome' >> ~/.bashrc && \
    echo 'if [ -f .devcontainer/welcome.sh ]; then bash .devcontainer/welcome.sh; fi' >> ~/.bashrc

USER root

# Install additional Python packages globally for convenience
RUN pip3 install --no-cache-dir \
    rich \
    typer \
    pytest \
    pytest-asyncio \
    httpx \
    numpy \
    scipy

USER vscode
