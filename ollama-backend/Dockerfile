FROM ollama/ollama:latest

# Listen on all interfaces, port 8080
ENV OLLAMA_HOST 0.0.0.0:8080

# Store model weight files in /models
ENV OLLAMA_MODELS /models

# Reduce logging verbosity
ENV OLLAMA_DEBUG false

# Never unload model weights from the GPU
ENV OLLAMA_KEEP_ALIVE -1 

# Copy custom parameter model files into the container
COPY ./Modelfile_deepseek-r1:7b_32k Modelfile_deepseek-r1:7b_32k
COPY ./Modelfile_llama3.2:3b_32k Modelfile_llama3.2:3b_32k


# Store the model weights in the container image
ENV EMBED_MODEL nomic-embed-text
ENV LLM_MODEL llama3.2:3b_32k
ENV REASON_MODEL deepseek-r1:7b_32k
RUN ollama serve & sleep 5 && ollama pull $EMBED_MODEL && ollama create $LLM_MODEL -f Modelfile_llama3.2:3b_32k && ollama create $REASON_MODEL -f Modelfile_deepseek-r1:7b_32k

# Start Ollama
ENTRYPOINT ["ollama", "serve"]