FROM nvcr.io/nvidia/pytorch:22.04-py3

# Install packages
#RUN apt-get update && apt-get install -y --no-install-recommends \
#    my-extra-package-1 \
#    my-extra-package-2 \
#    && \
#    rm -rf /var/lib/apt/lists/

RUN conda install -c huggingface transformers
WORKDIR /workspace
