FROM ollama/ollama:latest

# Listen on all interfaces, port 8080
ENV OLLAMA_HOST 0.0.0.0:8080

# Store model weight files in /models
ENV OLLAMA_MODELS /models

# Reduce logging verbosity
ENV OLLAMA_DEBUG false

# Never unload model weights from the GPU
ENV OLLAMA_KEEP_ALIVE -1 

ENV EMBED_MODEL nomic-embed-text
ENV LLM_MODEL llama3.2_32k
ENV REASON_MODEL deepseek-r1_32k
ENV IMAGE_MODEL llama3.2-vision_32k

# Copy custom parameter model files into the container
COPY ./$LLM_MODEL $LLM_MODEL
COPY ./$REASON_MODEL $REASON_MODEL
COPY ./$IMAGE_MODEL $IMAGE_MODEL


# Store the model weights in the container image
RUN ollama serve & sleep 5 && ollama pull $EMBED_MODEL && ollama create $LLM_MODEL -f $LLM_MODEL && ollama create $REASON_MODEL -f REASON_MODEL && ollama create $IMAGE_MODEL -f $IMAGE_MODEL

# Start Ollama
ENTRYPOINT ["ollama", "serve"]