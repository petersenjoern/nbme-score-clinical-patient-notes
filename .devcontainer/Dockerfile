FROM nvcr.io/nvidia/pytorch:22.04-py3

# Install packages
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    my-extra-package-1 \
#    my-extra-package-2 \
#    && \
#    rm -rf /var/lib/apt/lists/

# Install linux packages
RUN apt update && apt install --no-install-recommends -y zip htop screen libgl1-mesa-glx

# Install pip packages
RUN python -m pip install --upgrade pip
RUN pip install -U --no-cache wandb \
    --extra-index-url https://download.pytorch.org/whl/cu113

#RUN git clone https://github.com/huggingface/transformers.git && \
#    cd transformers && \
#    pip install -e .

# Set working dir
WORKDIR /workspace


# Set environment variables
ENV OMP_NUM_THREADS=8



#RUN pip install --user transformers[pytorch] -y && \
#    pip install --user wandb -y
#RUN conda install -c huggingface transformers -y && \
#    conda install wandb -y
