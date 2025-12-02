# Stage 1: Build Environment Setup
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 as builder

RUN apt-get update -y && apt-get install -y wget curl git tar bzip2 && rm -rf /var/lib/apt/lists/*

# Run as root; set HOME and WORKDIR explicitly
ENV HOME_INSTALL=/usr/local
WORKDIR $HOME_INSTALL

ENV ENV_NAME="diffdock-pocket"

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj bin/micromamba
ENV PATH=$HOME_INSTALL/bin:$HOME_INSTALL/.local/bin:$PATH

# Ensure micromamba root prefix exists and is set so envs are created under $HOME_INSTALL/micromamba
ENV MAMBA_ROOT_PREFIX=$HOME_INSTALL/micromamba
RUN mkdir -p $MAMBA_ROOT_PREFIX

# Copy and create Conda environment
ENV ENV_FILE_NAME=environment.yml
COPY ./$ENV_FILE_NAME .
# create the environment under $MAMBA_ROOT_PREFIX
RUN $HOME_INSTALL/bin/micromamba env create --file $ENV_FILE_NAME && $HOME_INSTALL/bin/micromamba clean -afy --quiet

# Copy application code
COPY . $HOME_INSTALL/DiffDock-Pocket

# Stage 2: Runtime Environment
FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

# Use root user; set HOME and WORKDIR explicitly
ENV HOME_INSTALL=/usr/local
WORKDIR $HOME_INSTALL

ENV ENV_NAME="diffdock-pocket"

# Copy the Conda environment and application code from the builder stage (no --chown)
COPY --from=builder $HOME_INSTALL/micromamba $HOME_INSTALL/micromamba
# Make sure copied micromamba envs and binaries are accessible to the container runtime user
# RUN chmod -R a+rx $HOME_INSTALL/micromamba || true
COPY --from=builder $HOME_INSTALL/bin $HOME_INSTALL/bin
COPY --from=builder $HOME_INSTALL/DiffDock-Pocket $HOME_INSTALL/DiffDock-Pocket
WORKDIR $HOME_INSTALL/DiffDock-Pocket

# Set the environment variables
ENV MAMBA_ROOT_PREFIX=$HOME_INSTALL/micromamba
ENV PATH=$HOME_INSTALL/bin:$HOME_INSTALL/.local/bin:$PATH
RUN micromamba shell init -s bash --root-prefix $MAMBA_ROOT_PREFIX

# Expose ports for streamlit and gradio
EXPOSE 7860 8501

# Default command
CMD ["sh", "-c", "micromamba run -n ${ENV_NAME} python utils/print_device.py"]
