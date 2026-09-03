kubectl create secret generic -n llm-system litellm-masterkey \
  --from-literal=masterkey="sk-$(openssl rand -hex 24)"

kubectl create secret generic -n llm-system litellm-db \
  --from-literal=username=litellm \
  --from-literal=password="dbpassword"

kubectl create secret generic -n llm-system litellm-env \
  --from-literal=LITELLM_SALT_KEY="sk-$(openssl rand -hex 24)" \
  --from-literal=REDIS_PASSWORD="redispassword" \
  --from-literal=OPENAI_API_KEY="openaiapikey"
